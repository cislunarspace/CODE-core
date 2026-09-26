//! 星历预采样缓存（仅 `spice` feature 下编译）。
//!
//! Rust 积分内循环里，`GravityField`/`ThirdBody`/`IndirectTerm` 每个 RK 子步
//! 都跨界调 cspice FFI（`spkezr`/`pxform`）。本模块在积分前把要用到的天体
//! 状态与旋转矩阵在均匀网格上预采样、建三次样条存内存；力模型每步查表，不再
//! 调 cspice。
//!
//! Python 侧的 `ephem_cache.py` 只拦 Python 层查询，对 Rust 积分内循环无效
//! （Rust 直接走 `spk_accel`/`gravity_field`→`spice_ffi`，不回 Python）。
//! 故缓存必须做在 Rust 侧。
//!
//! # 为什么用三次样条而非线性插值
//!
//! 线性插值 C⁰ 连续，网格点处导数跳变，让自适应积分器（PD45）疯狂缩步长
//! （实测 93 倍 RHS 调用）。三次样条 C² 连续消除此问题（经验同
//! `ephem_cache.py:10-16` 与 qiao `ephem_table.py`）。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use cspice::common::AberrationCorrection;
use cspice::spk::easier_reader;
use cspice::time::Et;

use crate::spice_ffi::{pxform, SpiceFfiError};

/// 缓存查询失败原因（strict 模式下的硬错误）。
#[derive(Debug, Clone, PartialEq)]
pub enum CacheMissError {
    /// 缓存未启用（未调 `enable`）。
    NotEnabled,
    /// 键缺失：该 (target, observer) / (from, to) / (target, observer, frame)
    /// 未被预采样注册。
    KeyMiss(String),
    /// 查询时刻越出缓存覆盖范围。
    OutOfRange { et: f64, start: f64, end: f64 },
}

impl std::fmt::Display for CacheMissError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheMissError::NotEnabled => write!(f, "ephem cache not enabled"),
            CacheMissError::KeyMiss(key) => write!(f, "ephem cache key not registered: {key}"),
            CacheMissError::OutOfRange { et, start, end } => {
                write!(f, "ephem cache et {et} out of range [{start}, {end}]")
            }
        }
    }
}

impl From<CacheMissError> for SpiceFfiError {
    fn from(e: CacheMissError) -> Self {
        SpiceFfiError::Failed(e.to_string())
    }
}

impl From<CacheMissError> for cspice::Error {
    fn from(e: CacheMissError) -> Self {
        cspice::Error {
            short_message: "EPHEM_CACHE_MISS".to_string(),
            explanation: e.to_string(),
            long_message: String::new(),
            traceback: String::new(),
        }
    }
}

/// 采样体状态向量（位置 + 速度，共 6 个分量）。
type BodySample = [f64; 6];
/// 3×3 旋转矩阵。
type FrameMatrix = [[f64; 3]; 3];
/// 6×6 状态变换矩阵。
type SxformMatrix = [[f64; 6]; 6];

/// (key, 采样数据) 类型别名（Vec，用于 build 内部暂存）。
type BodyGrids = Vec<((String, String), Vec<BodySample>)>;
type FrameGrids = Vec<((String, String), Vec<FrameMatrix>)>;
type SxformGrids = Vec<((String, String), Vec<SxformMatrix>)>;

/// 单个采样点类型（用于公共 API 参数切片）。
type BodyEntry = ((String, String), Vec<BodySample>);
type FrameEntry = ((String, String), Vec<FrameMatrix>);
type SxformEntry = ((String, String), Vec<SxformMatrix>);

/// 自然三次样条：预解二阶导数，查询 O(log N) 二分定位 + O(1) 求值。
///
/// 构造时用追赶法解三对角系统求各节点二阶导数 `m`（自然边界 m₀=mₙ=0）。
struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// 各节点二阶导数（自然样条首末为 0）
    m: Vec<f64>,
}

impl CubicSpline {
    /// 由 (xs, ys) 构造自然三次样条。`xs` 须严格递增、长度 ≥ 2。
    fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let n = xs.len();
        debug_assert!(n >= 2, "spline needs >= 2 points");
        debug_assert!(
            xs.windows(2).all(|w| w[0] < w[1]),
            "xs must be strictly increasing"
        );

        // 自然边界：二阶导数 m[0] = m[n-1] = 0；中间节点解三对角系统。
        // 标准方程：h[idx-1]·m[idx-1] + 2(h[idx-1]+h[idx])·m[idx]
        //          + h[idx]·m[idx+1]
        //          = 6((y[idx+1]-y[idx])/h[idx] - (y[idx]-y[idx-1])/h[idx-1])
        // 用 Thomas 算法（追赶法）解 m[1..n-1]。
        let mut m = vec![0.0_f64; n];
        if n >= 3 {
            let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
            let nm = n - 2; // 未知数个数（m[1..n-1]）
            let mut a = vec![0.0_f64; nm]; // 下对角 a[0] 不用
            let mut b = vec![0.0_f64; nm]; // 主对角
            let mut c = vec![0.0_f64; nm]; // 上对角 c[nm-1] 不用
            let mut d = vec![0.0_f64; nm];
            for i in 0..nm {
                let idx = i + 1;
                let h0 = h[idx - 1];
                let h1 = h[idx];
                b[i] = 2.0 * (h0 + h1);
                d[i] = 6.0 * ((ys[idx + 1] - ys[idx]) / h1 - (ys[idx] - ys[idx - 1]) / h0);
                if i > 0 {
                    a[i] = h0;
                }
                if i + 1 < nm {
                    c[i] = h1;
                }
            }
            // 追赶：前消元
            let mut cp = vec![0.0_f64; nm];
            let mut dp = vec![0.0_f64; nm];
            cp[0] = c[0] / b[0];
            dp[0] = d[0] / b[0];
            for i in 1..nm {
                let denom = b[i] - a[i] * cp[i - 1];
                cp[i] = if i + 1 < nm { c[i] / denom } else { 0.0 };
                dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
            }
            // 回代
            let mut x = vec![0.0_f64; nm];
            x[nm - 1] = dp[nm - 1];
            for i in (0..nm - 1).rev() {
                x[i] = dp[i] - cp[i] * x[i + 1];
            }
            m[1..(nm + 1)].copy_from_slice(&x[..nm]);
        }
        // m[0]、m[n-1] 保持 0（自然边界）

        Self { xs, ys, m }
    }

    /// 在 t 处求值。t 须在 [xs[0], xs[n-1]] 内；越界端点钳位（调用方保证覆盖）。
    fn eval(&self, t: f64) -> f64 {
        let n = self.xs.len();
        if t <= self.xs[0] {
            return self.ys[0];
        }
        if t >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        // 二分定位区间 i 使 xs[i] <= t < xs[i+1]
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let i = lo;
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - t) / h;
        let b = (t - self.xs[i]) / h;
        // 三次样条求值（二阶导数形式）
        a * self.ys[i]
            + b * self.ys[i + 1]
            + ((a * a * a - a) * self.m[i] + (b * b * b - b) * self.m[i + 1]) * (h * h) / 6.0
    }

    /// 在 t 处求二阶导数。二阶导数形式的样条里 y'' 在区间内是节点二阶
    /// 导数 `m` 的线性插值（C¹ 连续）；端点越界钳位同 `eval`。
    ///
    /// 供需要天体加速度的使用方（如 HJB 时变会合系的 d̈、ω̇）从同一条
    /// 位置样条取二阶导，避免另建查询路径。
    fn eval_second(&self, t: f64) -> f64 {
        let n = self.xs.len();
        if t <= self.xs[0] {
            return self.m[0];
        }
        if t >= self.xs[n - 1] {
            return self.m[n - 1];
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let i = lo;
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - t) / h;
        let b = (t - self.xs[i]) / h;
        a * self.m[i] + b * self.m[i + 1]
    }
}

/// 单个 (target, observer) 对的位置/速度样条。
struct BodySpline {
    pos: [CubicSpline; 3],
    vel: [CubicSpline; 3],
}

/// 单个 (from, to) 帧对的旋转矩阵 9 分量样条（行优先）。
struct FrameSpline {
    comps: [CubicSpline; 9],
}

/// 单个 (from, to) 帧对的 6×6 状态变换矩阵 36 分量样条（行优先）。
///
/// SPICE `sxform` 返回的 6×6 矩阵分块为 ``[R 0; Rdot R]``——左上 3×3 是
/// 旋转 R，左下 3×3 是 R·ω（R 对时间导数），右下仍为 R。缓存 36 个分量
/// 可直接插值整个矩阵，供 Lense-Thirring 等需要 Rdot 的力模型使用。
struct SxformSpline {
    comps: [CubicSpline; 36],
}

/// ET → UTC 日历量的预采样表。
///
/// 与星历共用同一时间网格（[`EphemCache::build`] 的 `t_grid`），存的是
/// 「UTC 自 2000-01-01T00:00:00Z 起的秒数」这条**连续单调**曲线；查询时在相邻
/// 采样点间线性插值。若改存 (年积日, 日内秒)，年/日边界会回绕而无法插值。
///
/// 精度：区间内 UTC 相对 ET 是斜率为 1 的分段线性函数，插值误差只来自
/// TDB−UTC 周期项（< 2 ms）；仅当某区间内含闰秒跳变时误差可达 1 s（该秒被摊到
/// 整个采样区间）。对阻力应用无可见影响（1 s 的 UT 差对应日侧隆起 0.004°），
/// 且插值是纯数值、确定性——串行与并行逐位一致。
struct UtcTable {
    et: Vec<f64>,
    utc_seconds: Vec<f64>,
}

impl UtcTable {
    /// 由同一时间网格的 UTC 秒采样构造。
    fn new(et: Vec<f64>, utc_seconds: Vec<f64>) -> Self {
        debug_assert_eq!(et.len(), utc_seconds.len());
        Self { et, utc_seconds }
    }

    /// 查 `(年, 年积日, 日内秒)`；越出覆盖区间返回 `None`。
    fn calendar_at(&self, et: f64) -> Option<(i32, u16, f64)> {
        let n = self.et.len();
        if n == 0 || et < self.et[0] || et > self.et[n - 1] {
            return None;
        }
        let i0 = self.et.partition_point(|&t| t <= et).saturating_sub(1);
        let i1 = (i0 + 1).min(n - 1);
        let utc_seconds = if i1 == i0 {
            self.utc_seconds[i0]
        } else {
            let frac = (et - self.et[i0]) / (self.et[i1] - self.et[i0]);
            self.utc_seconds[i0] + frac * (self.utc_seconds[i1] - self.utc_seconds[i0])
        };
        Some(crate::spice_ffi::utc_seconds_to_calendar(utc_seconds))
    }
}

/// 星历预采样缓存。
pub struct EphemCache {
    bodies: HashMap<(String, String), BodySpline>,
    frames: HashMap<(String, String), FrameSpline>,
    sxforms: HashMap<(String, String), SxformSpline>,
    /// ET → UTC 日历量（[`EphemCache::build`] 采样；[`EphemCache::from_raw_grids`]
    /// 构造的缓存无此项）。
    utc: Option<UtcTable>,
    et_start: f64,
    et_end: f64,
}

impl EphemCache {
    /// 预采样构建缓存。
    ///
    /// 对每个 (target, observer) 在 [et_start, et_end] 上以 dt 步长采样位置/速度；
    /// 对每个 (from, to) 帧对采样 pxform 的 9 个分量；
    /// 对每个 (from, to) sxform 对采样 sxform 的 36 个分量。
    ///
    /// 内部先通过 cspice 采集原始网格，再委托给 [`EphemCache::from_raw_grids`]
    /// 做样条拟合——后者可供无 cspice 内核的测试直接构造缓存。
    pub fn build(
        bodies: &[(String, String)],
        frames: &[(String, String)],
        sxform_pairs: &[(String, String)],
        et_start: f64,
        et_end: f64,
        dt: f64,
    ) -> Result<Self, SpiceFfiError> {
        if et_end <= et_start {
            return Err(SpiceFfiError::Failed("et_end must be > et_start".into()));
        }
        if dt <= 0.0 {
            return Err(SpiceFfiError::Failed("dt must be positive".into()));
        }
        // 两端 margin，避免积分器步长越界
        let margin = 5.0 * dt;
        let t0 = et_start - margin;
        let t1 = et_end + margin;
        let n = ((t1 - t0) / dt).ceil() as usize + 1;
        let t_grid: Vec<f64> = (0..n).map(|i| t0 + i as f64 * dt).collect();

        // ── 采集 body 原始网格 ──
        let mut body_grids: BodyGrids = Vec::with_capacity(bodies.len());
        for (target, observer) in bodies {
            let mut states = Vec::with_capacity(n);
            for &et in &t_grid {
                let et_tdb = Et::from(et);
                let (state, _lt) = easier_reader(
                    target,
                    et_tdb,
                    "J2000",
                    AberrationCorrection::NONE,
                    observer,
                )
                .map_err(|e| SpiceFfiError::Failed(format!("cspice read: {e:?}")))?;
                states.push([
                    state.position.x,
                    state.position.y,
                    state.position.z,
                    state.velocity.0[0],
                    state.velocity.0[1],
                    state.velocity.0[2],
                ]);
            }
            body_grids.push(((target.clone(), observer.clone()), states));
        }

        // ── 采集 frame 原始网格 ──
        let mut frame_grids: FrameGrids = Vec::with_capacity(frames.len());
        for (from, to) in frames {
            let mut mats = Vec::with_capacity(n);
            for &et in &t_grid {
                let r = pxform(from, to, et)?;
                mats.push(r);
            }
            frame_grids.push(((from.clone(), to.clone()), mats));
        }

        // ── 采集 sxform 原始网格 ──
        let mut sx_grids: SxformGrids = Vec::with_capacity(sxform_pairs.len());
        for (from, to) in sxform_pairs {
            let mut mats = Vec::with_capacity(n);
            for &et in &t_grid {
                let m = crate::spice_ffi::sxform(from, to, et)?;
                mats.push(m);
            }
            sx_grids.push(((from.clone(), to.clone()), mats));
        }

        // ── 采集 ET→UTC 原始网格 ──
        // 供需要日历量的力模型（NRLMSISE-00）在并行传播区零 cspice 取值：
        // 采样在 enable 时单线程完成，热循环只查内存表（ADR 0016 的并行区
        // 零 cspice 保证因此对这类力模型同样成立）。
        let mut utc_seconds = Vec::with_capacity(n);
        for &et in &t_grid {
            utc_seconds.push(crate::spice_ffi::et_to_utc_seconds(et)?);
        }

        let mut cache = Self::from_raw_grids(&t_grid, &body_grids, &frame_grids, &sx_grids)?;
        cache.utc = Some(UtcTable::new(t_grid, utc_seconds));
        Ok(cache)
    }

    /// 由预采样原始网格直接构造缓存，不经 cspice / 内核。
    ///
    /// 与 [`EphemCache::build`] 共享样条拟合逻辑，但跳过 cspice 采样——供无
    /// 内核依赖的单元测试，以及未来非 SPICE 星历源（如自定义解析星历）使用。
    ///
    /// # 参数
    /// - `t_grid`: 采样时刻，须严格递增、长度 ≥ 2；成为缓存覆盖范围。
    /// - `bodies`: 每个 (target, observer) 在各采样点的 6 维状态 [x,y,z,vx,vy,vz]。
    /// - `frames`: 每个 (from, to) 在各采样点的 3×3 旋转矩阵。
    /// - `sxforms`: 每个 (from, to) 在各采样点的 6×6 状态变换矩阵。
    ///
    /// 各 body/frame/sxform 序列长度须等于 `t_grid.len()`。
    pub fn from_raw_grids(
        t_grid: &[f64],
        bodies: &[BodyEntry],
        frames: &[FrameEntry],
        sxforms: &[SxformEntry],
    ) -> Result<Self, SpiceFfiError> {
        let n = t_grid.len();
        if n < 2 {
            return Err(SpiceFfiError::Failed("t_grid 至少需要 2 个采样点".into()));
        }
        if !t_grid.windows(2).all(|w| w[0] < w[1]) {
            return Err(SpiceFfiError::Failed("t_grid 必须严格递增".into()));
        }
        for ((tgt, obs), states) in bodies {
            if states.len() != n {
                return Err(SpiceFfiError::Failed(format!(
                    "body ({tgt}, {obs}) 状态数 {} != 网格长度 {n}",
                    states.len()
                )));
            }
        }
        for ((from, to), mats) in frames {
            if mats.len() != n {
                return Err(SpiceFfiError::Failed(format!(
                    "frame ({from}, {to}) 矩阵数 {} != 网格长度 {n}",
                    mats.len()
                )));
            }
        }
        for ((from, to), mats) in sxforms {
            if mats.len() != n {
                return Err(SpiceFfiError::Failed(format!(
                    "sxform ({from}, {to}) 矩阵数 {} != 网格长度 {n}",
                    mats.len()
                )));
            }
        }

        let t = t_grid.to_vec();

        // ── 对 body 建样条 ──
        let mut body_map = HashMap::new();
        for ((target, observer), states) in bodies {
            let mut pos_grids = [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]];
            let mut vel_grids = [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]];
            for (i, s) in states.iter().enumerate() {
                pos_grids[0][i] = s[0];
                pos_grids[1][i] = s[1];
                pos_grids[2][i] = s[2];
                vel_grids[0][i] = s[3];
                vel_grids[1][i] = s[4];
                vel_grids[2][i] = s[5];
            }
            let pos = [
                CubicSpline::new(t.clone(), pos_grids[0].clone()),
                CubicSpline::new(t.clone(), pos_grids[1].clone()),
                CubicSpline::new(t.clone(), pos_grids[2].clone()),
            ];
            let vel = [
                CubicSpline::new(t.clone(), vel_grids[0].clone()),
                CubicSpline::new(t.clone(), vel_grids[1].clone()),
                CubicSpline::new(t.clone(), vel_grids[2].clone()),
            ];
            body_map.insert((target.clone(), observer.clone()), BodySpline { pos, vel });
        }

        // ── 对 frame 建样条 ──
        let mut frame_map = HashMap::new();
        for ((from, to), mats) in frames {
            let mut comps_grids: [Vec<f64>; 9] = Default::default();
            for g in comps_grids.iter_mut() {
                *g = vec![0.0_f64; n];
            }
            for (i, r) in mats.iter().enumerate() {
                for k in 0..9 {
                    comps_grids[k][i] = r[k / 3][k % 3];
                }
            }
            let comps = comps_grids.map(|g| CubicSpline::new(t.clone(), g));
            frame_map.insert((from.clone(), to.clone()), FrameSpline { comps });
        }

        // ── 对 sxform 建样条 ──
        let mut sxform_map = HashMap::new();
        for ((from, to), mats) in sxforms {
            let mut comps_grids: [Vec<f64>; 36] = std::array::from_fn(|_| vec![0.0_f64; n]);
            for (i, m) in mats.iter().enumerate() {
                for r in 0..6 {
                    for c in 0..6 {
                        comps_grids[r * 6 + c][i] = m[r][c];
                    }
                }
            }
            let comps = comps_grids.map(|g| CubicSpline::new(t.clone(), g));
            sxform_map.insert((from.clone(), to.clone()), SxformSpline { comps });
        }

        Ok(Self {
            bodies: body_map,
            frames: frame_map,
            sxforms: sxform_map,
            utc: None,
            et_start: t_grid[0],
            et_end: t_grid[n - 1],
        })
    }

    /// 查 (target, observer) 在 et 的位置。未缓存或越界返回 None。
    pub fn body_position(&self, target: &str, observer: &str, et: f64) -> Option<[f64; 3]> {
        let key = (target.to_string(), observer.to_string());
        let bs = self.bodies.get(&key)?;
        if !(self.et_start..=self.et_end).contains(&et) {
            return None;
        }
        Some([bs.pos[0].eval(et), bs.pos[1].eval(et), bs.pos[2].eval(et)])
    }

    /// 查 (target, observer) 在 et 的速度。未缓存或越界返回 None。
    pub fn body_velocity(&self, target: &str, observer: &str, et: f64) -> Option<[f64; 3]> {
        let key = (target.to_string(), observer.to_string());
        let bs = self.bodies.get(&key)?;
        if !(self.et_start..=self.et_end).contains(&et) {
            return None;
        }
        Some([bs.vel[0].eval(et), bs.vel[1].eval(et), bs.vel[2].eval(et)])
    }

    /// 查 (target, observer) 在 et 的加速度——位置样条的二阶导数。
    /// 未缓存或越界返回 None。
    pub fn body_acceleration(&self, target: &str, observer: &str, et: f64) -> Option<[f64; 3]> {
        let key = (target.to_string(), observer.to_string());
        let bs = self.bodies.get(&key)?;
        if !(self.et_start..=self.et_end).contains(&et) {
            return None;
        }
        Some([
            bs.pos[0].eval_second(et),
            bs.pos[1].eval_second(et),
            bs.pos[2].eval_second(et),
        ])
    }

    /// 查 (from, to) 帧旋转矩阵。未缓存或越界返回 None。
    pub fn frame_matrix(&self, from: &str, to: &str, et: f64) -> Option<[[f64; 3]; 3]> {
        let key = (from.to_string(), to.to_string());
        let fs = self.frames.get(&key)?;
        if !(self.et_start..=self.et_end).contains(&et) {
            return None;
        }
        let mut r = [[0.0_f64; 3]; 3];
        for k in 0..9 {
            r[k / 3][k % 3] = fs.comps[k].eval(et);
        }
        Some(r)
    }

    /// 查 (from, to) 帧 6×6 状态变换矩阵。未缓存或越界返回 None。
    pub fn state_transform_matrix(&self, from: &str, to: &str, et: f64) -> Option<[[f64; 6]; 6]> {
        let key = (from.to_string(), to.to_string());
        let ss = self.sxforms.get(&key)?;
        if !(self.et_start..=self.et_end).contains(&et) {
            return None;
        }
        let mut m = [[0.0_f64; 6]; 6];
        for (k, row) in m.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                *val = ss.comps[k * 6 + j].eval(et);
            }
        }
        Some(m)
    }
}

// ---- 进程级单例 ----

/// 缓存实例。`RwLock` 而非 `Mutex`：并行段积分下多线程并发读三次样条
/// （纯数值，无 cspice），读锁并行不互相阻塞；enable/disable 写锁与读锁互斥。
static CACHE: RwLock<Option<EphemCache>> = RwLock::new(None);

/// strict 模式标记（`StrictGuard` RAII 管理，打靶并行区开启）：并行区内即使
/// 缓存未启用也硬失败，保证零 cspice（并行区 cspice 是内核池损坏/panic 的
/// 根源）。缓存已启用后的 miss（区间外/缺 target）不受本标记控制——一律
/// 返回 `Err`（ADR 0020 决策 4）；本标记只额外兜住"未启用缓存"场景。
static STRICT: AtomicBool = AtomicBool::new(false);

/// RAII：作用域内开启 strict 缓存模式，Drop 时恢复原值。
///
/// 语义：strict 下 "未启用缓存" 的 `lookup_*` 查询返回 `Err`（硬失败），
/// 由调用方 `?` 向上传播——杜绝并行区力模型静默回退 cspice。非 strict 下
/// 未启用缓存返回 `Ok(None)`，调用方按既有模式回退 cspice（合法路径）。
pub struct StrictGuard {
    prev: bool,
}

impl StrictGuard {
    pub fn new() -> Self {
        let prev = STRICT.swap(true, Ordering::SeqCst);
        Self { prev }
    }
}

impl Drop for StrictGuard {
    fn drop(&mut self) {
        STRICT.store(self.prev, Ordering::SeqCst);
    }
}

impl Default for StrictGuard {
    fn default() -> Self {
        Self::new()
    }
}

fn strict() -> bool {
    STRICT.load(Ordering::SeqCst)
}

/// 安装缓存（替换已有的）。后续力模型查询优先走它。
pub fn enable(cache: EphemCache) {
    let mut g = CACHE.write().expect("ephem cache rwlock poisoned");
    *g = Some(cache);
}

/// 清除缓存（回到逐次 cspice 查询）。
pub fn disable() {
    let mut g = CACHE.write().expect("ephem cache rwlock poisoned");
    *g = None;
}

/// 当前是否处于 strict 缓存模式（`StrictGuard` 作用域内）。
///
/// 供力模型上层区分「缓存未启用且禁止回退 cspice」（strict 区硬失败）与普通
/// SPICE 失败——两者都不带缓存窗口，只能靠本状态分辨（ADR 0020 决策 4）。
pub fn strict_enabled() -> bool {
    strict()
}

/// 当前已启用缓存的覆盖区间（et 秒）。未启用返回 None。
///
/// 供构造方（如 HJB 星历 Hamiltonian 绑定层）在求解前校验求解窗被
/// 缓存覆盖——越界查询在求解热循环里是硬失败，提前到构造时报错。
pub fn enabled_span() -> Option<(f64, f64)> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    g.as_ref().map(|c| (c.et_start, c.et_end))
}

/// 缓存 key 归一化：NAIF ID 字符串（如 "301"）转名字（"MOON"）。
/// ``to_rust_spec`` 把天体名转成 ID 字符串传给 ``easier_reader``，而
/// ``enable_ephem_cache`` 侧用名字作 key——enable 后 miss 即硬失败
/// （ADR 0020 决策 4），key 不一致必须在查询侧收敛而非静默回退。
fn normalize_body_name(name: &str) -> &str {
    if let Ok(id) = name.parse::<i32>() {
        if let Some(canonical) = crate::spice_ffi::id_to_name(id) {
            return canonical;
        }
    }
    name
}

/// strict-aware 查 ET → UTC 日历量 `(年, 年积日, 日内秒)`。
///
/// 语义同 [`lookup_body_position`]：缓存未启用时 `Ok(None)`（调用方回退 cspice，
/// 合法路径）；strict 模式下未启用即 `Err`；缓存已启用后越界一律 `Err`。
/// 需要日历量的力模型（NRLMSISE-00）靠这条路径在并行传播区零 cspice。
pub fn lookup_utc_calendar(et: f64) -> Result<Option<(i32, u16, f64)>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    // `from_raw_grids` 构造的缓存（无 cspice 采样的测试/外部星历源）没有该项。
    let Some(utc) = cache.utc.as_ref() else {
        return Err(CacheMissError::KeyMiss("utc calendar".into()));
    };
    if let Some(calendar) = utc.calendar_at(et) {
        return Ok(Some(calendar));
    }
    Err(CacheMissError::OutOfRange {
        et,
        start: cache.et_start,
        end: cache.et_end,
    })
}

/// 查天体位置。缓存未启用（``enable_ephem_cache`` 未调用）时返回 `Ok(None)`
/// （调用方回退 cspice，合法路径）；**启用后** miss（区间外 / 目标不在预采样
/// 列表）一律返回 `Err`（ADR 0020 决策 4：enable 是用户要求缓存的信号，
/// enable 后 miss 就是错误，不静默回退 cspice）。``StrictGuard`` 仅对
/// "未启用缓存" 场景额外生效（并行区零 cspice 保险）。
pub fn lookup_body_position(
    target: &str,
    observer: &str,
    et: f64,
) -> Result<Option<[f64; 3]>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    let pos = cache.body_position(normalize_body_name(target), observer, et);
    if pos.is_some() {
        return Ok(pos);
    }
    // 缓存已启用但 miss：一律硬失败（不再区分 strict/非 strict）。
    if !(cache.et_start..=cache.et_end).contains(&et) {
        return Err(CacheMissError::OutOfRange {
            et,
            start: cache.et_start,
            end: cache.et_end,
        });
    }
    Err(CacheMissError::KeyMiss(format!("({target}, {observer})")))
}

/// 查帧旋转矩阵。语义同 `lookup_body_position`。
pub fn lookup_frame_matrix(
    from: &str,
    to: &str,
    et: f64,
) -> Result<Option<[[f64; 3]; 3]>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    let m = cache.frame_matrix(from, to, et);
    if m.is_some() {
        return Ok(m);
    }
    // 缓存已启用但 miss：一律硬失败（不再区分 strict/非 strict）。
    if !(cache.et_start..=cache.et_end).contains(&et) {
        return Err(CacheMissError::OutOfRange {
            et,
            start: cache.et_start,
            end: cache.et_end,
        });
    }
    Err(CacheMissError::KeyMiss(format!("frame ({from}, {to})")))
}

/// strict-aware 查帧 6×6 状态变换矩阵。语义同 `lookup_body_position`。
pub fn lookup_sxform(
    from: &str,
    to: &str,
    et: f64,
) -> Result<Option<[[f64; 6]; 6]>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    let m = cache.state_transform_matrix(from, to, et);
    if m.is_some() {
        return Ok(m);
    }
    // 缓存已启用但 miss：一律硬失败（不再区分 strict/非 strict）。
    if !(cache.et_start..=cache.et_end).contains(&et) {
        return Err(CacheMissError::OutOfRange {
            et,
            start: cache.et_start,
            end: cache.et_end,
        });
    }
    Err(CacheMissError::KeyMiss(format!("sxform ({from}, {to})")))
}

/// 查天体速度。语义同 `lookup_body_position`。
pub fn lookup_body_velocity(
    target: &str,
    observer: &str,
    et: f64,
) -> Result<Option<[f64; 3]>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    let vel = cache.body_velocity(normalize_body_name(target), observer, et);
    if vel.is_some() {
        return Ok(vel);
    }
    // 缓存已启用但 miss：一律硬失败（不再区分 strict/非 strict）。
    if !(cache.et_start..=cache.et_end).contains(&et) {
        return Err(CacheMissError::OutOfRange {
            et,
            start: cache.et_start,
            end: cache.et_end,
        });
    }
    Err(CacheMissError::KeyMiss(format!("({target}, {observer})")))
}

/// 查天体加速度（位置样条二阶导数）。语义同 `lookup_body_position`。
pub fn lookup_body_acceleration(
    target: &str,
    observer: &str,
    et: f64,
) -> Result<Option<[f64; 3]>, CacheMissError> {
    let g = CACHE.read().expect("ephem cache rwlock poisoned");
    let Some(cache) = g.as_ref() else {
        return if strict() {
            Err(CacheMissError::NotEnabled)
        } else {
            Ok(None)
        };
    };
    let acc = cache.body_acceleration(normalize_body_name(target), observer, et);
    if acc.is_some() {
        return Ok(acc);
    }
    // 缓存已启用但 miss：一律硬失败（不再区分 strict/非 strict）。
    if !(cache.et_start..=cache.et_end).contains(&et) {
        return Err(CacheMissError::OutOfRange {
            et,
            start: cache.et_start,
            end: cache.et_end,
        });
    }
    Err(CacheMissError::KeyMiss(format!("({target}, {observer})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 跨年边界的插值不得回绕：存 (年积日, 日内秒) 会在年界出错，存连续 UTC 秒则正确。
    #[test]
    fn utc_table_interpolates_across_year_boundary() {
        // 2000-12-29 .. 2001-01-03（2000 为闰年，doy 366 = 12-31）。
        let et: Vec<f64> = (362..368).map(|i| f64::from(i) * 86400.0).collect();
        let table = UtcTable::new(et.clone(), et.clone());

        // 节点：+364 天是 2000 年 doy 365，+365 天是 doy 366（12-31），
        // +366 天进入 2001 年 doy 1。
        assert_eq!(table.calendar_at(364.0 * 86400.0), Some((2000, 365, 0.0)));
        assert_eq!(table.calendar_at(365.0 * 86400.0), Some((2000, 366, 0.0)));
        assert_eq!(table.calendar_at(366.0 * 86400.0), Some((2001, 1, 0.0)));
        // 年界中点：插值出 2000-12-31T12:00，而非把年积日当连续量回绕。
        assert_eq!(
            table.calendar_at(365.5 * 86400.0),
            Some((2000, 366, 43200.0))
        );
        // 覆盖区间外返回 None。
        assert!(table.calendar_at(361.0 * 86400.0).is_none());
        assert!(table.calendar_at(368.0 * 86400.0).is_none());
    }

    /// UTC 秒原点在 2000-01-01T00:00:00Z：3×86400 落在第 4 天（年积日 4）。
    #[test]
    fn utc_table_maps_days_to_day_of_year() {
        let et: Vec<f64> = (0..5).map(|i| i as f64 * 86400.0).collect();
        let table = UtcTable::new(et.clone(), et);
        let (y, doy, s) = table.calendar_at(3.0 * 86400.0).unwrap();
        assert_eq!((y, doy, s), (2000, 4, 0.0));
    }

    /// `lookup_utc_calendar` 的三态语义：未启用（strict / 非 strict）、无 UTC 表、
    /// 越界。ADR 0016 的并行区零 cspice 就靠"strict 下未启用即硬失败"这一条。
    ///
    /// 本用例读写进程级缓存，故把三种情形合并为一个用例，并依赖同模块其余用例
    /// 都是纯数值（不碰全局）。
    #[test]
    fn lookup_utc_calendar_semantics() {
        disable();
        // 未启用 + 非 strict：Ok(None) —— 调用方回退 cspice（合法路径）。
        assert_eq!(lookup_utc_calendar(0.0).unwrap(), None);
        {
            // 未启用 + strict：硬失败，杜绝并行区静默回退 cspice。
            let _strict = StrictGuard::new();
            assert!(matches!(
                lookup_utc_calendar(0.0),
                Err(CacheMissError::NotEnabled)
            ));
        }
        assert_eq!(lookup_utc_calendar(0.0).unwrap(), None);

        // 由原始网格构造（无 cspice 采样）的缓存没有 UTC 表。
        let t_grid: Vec<f64> = (0..4).map(|i| i as f64 * 10.0).collect();
        let cache = EphemCache::from_raw_grids(&t_grid, &[], &[], &[]).expect("构造缓存");
        enable(cache);
        assert!(matches!(
            lookup_utc_calendar(5.0),
            Err(CacheMissError::KeyMiss(_))
        ));

        // 带 UTC 表：区间内按秒查得日历量，区间外 OutOfRange。
        let mut cache = EphemCache::from_raw_grids(&t_grid, &[], &[], &[]).expect("构造缓存");
        cache.utc = Some(UtcTable::new(
            t_grid,
            (0..4).map(|i| i as f64 * 10.0).collect(),
        ));
        enable(cache);
        assert_eq!(lookup_utc_calendar(15.0).unwrap(), Some((2000, 1, 15.0)));
        assert!(matches!(
            lookup_utc_calendar(100.0),
            Err(CacheMissError::OutOfRange { .. })
        ));
        disable();
    }

    #[test]
    fn test_cubic_spline_reproduces_samples() {
        // 样条在节点处应精确返回原值
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 2.0, 1.5, 0.5];
        let sp = CubicSpline::new(xs, ys);
        for (x, y) in [(0.0, 0.0), (1.0, 2.0), (2.0, 1.5), (3.0, 0.5)] {
            assert!((sp.eval(x) - y).abs() < 1e-10, "at {x}");
        }
    }

    #[test]
    fn test_cubic_spline_smooth() {
        // 三次样条对 sin 应有较高精度（C² 连续）。步长 0.3 时精度 ~1e-4，
        // 边界附近略差；这里验证整体平顺性，容差取 5e-3。
        let n = 20;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let sp = CubicSpline::new(xs, ys);
        let mut max_err = 0.0_f64;
        for x in [0.15_f64, 0.7, 1.3, 2.5, 4.1, 5.2] {
            let expected = x.sin();
            let err = (sp.eval(x) - expected).abs();
            max_err = max_err.max(err);
            assert!(
                err < 5e-3,
                "at {x}: got {} exp {}, err {err}",
                sp.eval(x),
                expected
            );
        }
        // 内部点（远离边界）应明显更精确
        assert!((sp.eval(2.0) - 2.0_f64.sin()).abs() < 1e-4);
        let _ = max_err;
    }

    #[test]
    fn test_cubic_spline_second_derivative() {
        // 对 sin 采样建样条：二阶导应近似 -sin（同阶精度，容差与一阶导同量级）。
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let sp = CubicSpline::new(xs, ys);
        // 远离自然边界（首末节点二阶导强制为 0，边界误差大）。
        for x in [1.0_f64, 2.3, 3.7, 5.1, 6.4] {
            let got = sp.eval_second(x);
            let expected = -x.sin();
            assert!(
                (got - expected).abs() < 5e-3,
                "at {x}: got {got} exp {expected}"
            );
        }
        // 节点处二阶导应精确返回预解的 m。
        assert!((sp.eval_second(2.0) - sp.m[10]).abs() < 1e-12);
    }
}
