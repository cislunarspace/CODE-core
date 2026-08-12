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

/// 星历预采样缓存。
pub struct EphemCache {
    bodies: HashMap<(String, String), BodySpline>,
    frames: HashMap<(String, String), FrameSpline>,
    sxforms: HashMap<(String, String), SxformSpline>,
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

        Self::from_raw_grids(&t_grid, &body_grids, &frame_grids, &sx_grids)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
