//! 平面全星历脉动会合系 Hamiltonian（issue #498，ADR 0034 决策 1）。
//!
//! 会合系由月球瞬时星历定义（旋转 + 脉动）：长度单位取月地距 d(t)，
//! 月球与地球钉在无量纲位置 (1, 0) 与原点；时间单位取秒（求解器 t 即
//! ET 偏移秒，历元映射 et = et0 + t）。帧量 ω(t)、ω̇(t)、ḋ(t)、d̈(t)
//! 与面内基 (ê_r, ê_θ) 全部由缓存的月球位置/速度/加速度导出，每个 t
//! 只查一次星历缓存，全网格节点复用，求解阶段零 cspice。
//!
//! 动力学推导（惯性系 R = d·(x·ê_r + y·ê_θ)，A 为惯性合力，分量取
//! 旋转基投影）：
//!
//! ```text
//! ẋ = ux,  ẏ = uy
//! u̇x = A·ê_r + 2ω·uy + ω̇·y + (ω² − d̈/d)·x − 2(ḋ/d)(ux − ω·y)
//! u̇y = A·ê_θ − 2ω·ux − ω̇·x + (ω² − d̈/d)·y − 2(ḋ/d)(uy + ω·x)
//! ṁ  = −δ·T/(Isp·g0)          （含质量维时）
//! ```
//!
//! 平面口径：太阳第三体取星历位置在瞬时轨道面（月球位置 × 速度法向）
//! 的面内投影（ADR 0034 决策 1），轨道面进动引起的离面修正不在平面
//! 模型内——这是"时间保真度升级、不撞空间保真度墙"的取舍。
//!
//! min-fuel 控制解析消去（û ∈ S¹，δ ∈ [0,1]，L = w·δ）：
//! û* = −p_v/‖p_v‖；开关函数
//! `S = w − (T/1000m)·‖p_v‖ − p_m·T/(Isp·g0)`（T/1000m 为 km/s²），
//! S < 0 时 δ* = 1，否则 0；消去后 `H* = p·f_drift + min(0, S)`。
//!
//! `partial_bound` 包络（耗散安全上界）：
//! ∂H/∂p_x = ux、∂H/∂p_y = uy（精确 |v|）；
//! ∂H/∂p_v = f_drift 的加速度分量 + 推力项，包络 |u̇_d| + T/(1000m)
//! （m 取节点质量，5 维时为质量轴坐标）；
//! ∂H/∂p_m 在 δ* = 1 时为常值 −T/(Isp·g0)，包络取其绝对值。
//!
//! 力一致性（验收 b）：本实现的逐点力复刻 `CompiledForce` 的
//! `PointMass`/`ThirdBody` 公式，第三体位置取缓存查表——与
//! `compute_total_acceleration` 同源同式，对拍见
//! `tests/ephemeris_force_parity.rs`。唯一差异在天体中心 1e-6 km 郱域：
//! HJB 网格会精确命中主星坐标（会合系钉死位置），直接项按物理极限取
//! 0 而非 `MIN_DISTANCE` 截断（截断会产生 ~μ/ε² 伪加速度压垮 CFL），
//! 见 `inertial_accel` 内注释。

use e2m2e_forces::forces::compiled::CompiledForce;
use e2m2e_levelset::grid::Grid;
use e2m2e_levelset::hamiltonian::Hamiltonian;
use e2m2e_spice::ephem_cache;
use ndarray::ArrayD;

/// 与 e2m2e-forces / spk_accel 一致的最小距离截断（km）。
const MIN_DISTANCE: f64 = 1e-6;
/// 定义会合系的次星（帧天体）。
const FRAME_BODY: &str = "MOON";
/// 惯性系原点（传播系 observer）。
const OBSERVER: &str = "EARTH";

/// 一个 t 时刻的全部星历派生量。每个 RK 子步构造一次，全节点复用。
#[derive(Debug, Clone)]
pub struct FrameState {
    /// 月地距 d(t)（km）。
    pub d: f64,
    /// ḋ/d（1/s）。
    pub d_ratio_dot: f64,
    /// d̈/d（1/s²）。
    pub d_ratio_ddot: f64,
    /// 会合系角速度 ω(t)（1/s）。
    pub omega: f64,
    /// ω̇(t)（1/s²）。
    pub omega_dot: f64,
    /// 面内径向单位矢（惯性系分量）。
    pub e_r: [f64; 3],
    /// 面内横向单位矢（惯性系分量）。
    pub e_theta: [f64; 3],
    /// 各第三体的 (body, mu, 面内投影位置 km)，与 forces 列表中 ThirdBody 一一对应。
    pub third_bodies: Vec<(String, f64, [f64; 3])>,
}

/// 解析星历（常值），供退化对拍等测试注入。
#[derive(Debug, Clone)]
pub struct SyntheticEphemeris {
    /// 月球相对地球的位置（km，惯性系）。
    pub moon_pos: [f64; 3],
    /// 月球相对地球的速度（km/s）。
    pub moon_vel: [f64; 3],
    /// 月球相对地球的加速度（km/s²）。
    pub moon_accel: [f64; 3],
    /// 各第三体 (body, 惯性位置 km)，位置应取面内（测试自己保证）。
    pub bodies: Vec<(String, [f64; 3])>,
}

enum EphemerisSource {
    /// 进程级 EphemCache 单例（构造方负责 enable 覆盖求解窗）。
    Cache,
    /// 测试注入的常值解析星历。
    Synthetic(SyntheticEphemeris),
}

/// 平面全星历脉动会合系 Hamiltonian。
///
/// 状态 `(x, y, vx, vy)`（`with_mass` 时追加质量维 m，单位 kg）。
/// 构造参数：力模型列表（限 `PointMass` 与 `ThirdBody`）、历元映射
/// et0（求解器 t = 0 对应的 SPICE et）、求解窗（求解器 t，用于构造时
/// 对照已启用缓存区间校验）与发动机参数。线程安全：求值只读缓存
/// （EphemCache 读锁），无内部可变状态。
pub struct EphemerisPlanar {
    forces: Vec<CompiledForce>,
    /// 求解器 t = 0 对应的 SPICE et（秒）；et = et0 + t。
    pub et0: f64,
    /// 推力 T（N）。
    pub thrust: f64,
    /// 比冲 Isp（s）。
    pub isp: f64,
    /// 标准重力 g0（m/s²）。
    pub g0: f64,
    /// 燃料权重 w：运行代价 L = w·δ。
    pub fuel_weight: f64,
    /// 是否含质量维（5 维状态）。
    pub with_mass: bool,
    /// 4 维模式的固定质量（kg）。
    pub fixed_mass: f64,
    source: EphemerisSource,
}

impl EphemerisPlanar {
    /// 按真实缓存构造。调用前须 `ephem_cache::enable` 覆盖
    /// [et0 + t_span.0, et0 + t_span.1]，否则构造期即失败（把求解
    /// 热循环里的硬失败提前到配置期）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forces: Vec<CompiledForce>,
        et0: f64,
        t_span: (f64, f64),
        thrust: f64,
        isp: f64,
        g0: f64,
        fuel_weight: f64,
        with_mass: bool,
        fixed_mass: f64,
    ) -> Self {
        Self::validate(&forces, thrust, isp, g0, fuel_weight, with_mass, fixed_mass);
        let (et_lo, et_hi) = (et0 + t_span.0, et0 + t_span.1);
        match ephem_cache::enabled_span() {
            None => panic!("EphemCache 未启用：求解前须 enable 覆盖 [{et_lo}, {et_hi}]"),
            Some((start, end)) => assert!(
                start <= et_lo && et_hi <= end,
                "EphemCache 覆盖 [{start}, {end}] 不含求解窗 [{et_lo}, {et_hi}]"
            ),
        }
        Self {
            forces,
            et0,
            thrust,
            isp,
            g0,
            fuel_weight,
            with_mass,
            fixed_mass,
            source: EphemerisSource::Cache,
        }
    }

    /// 按常值解析星历构造（对拍测试用；不做缓存校验）。
    #[allow(clippy::too_many_arguments)]
    pub fn with_synthetic_ephemeris(
        forces: Vec<CompiledForce>,
        synthetic: SyntheticEphemeris,
        et0: f64,
        thrust: f64,
        isp: f64,
        g0: f64,
        fuel_weight: f64,
        with_mass: bool,
        fixed_mass: f64,
    ) -> Self {
        Self::validate(&forces, thrust, isp, g0, fuel_weight, with_mass, fixed_mass);
        Self {
            forces,
            et0,
            thrust,
            isp,
            g0,
            fuel_weight,
            with_mass,
            fixed_mass,
            source: EphemerisSource::Synthetic(synthetic),
        }
    }

    fn validate(
        forces: &[CompiledForce],
        thrust: f64,
        isp: f64,
        g0: f64,
        fuel_weight: f64,
        with_mass: bool,
        fixed_mass: f64,
    ) {
        assert!(!forces.is_empty(), "力模型列表不能为空");
        for f in forces {
            assert!(
                matches!(
                    f,
                    CompiledForce::PointMass { .. } | CompiledForce::ThirdBody { .. }
                ),
                "EphemerisPlanar 只接受 PointMass 与 ThirdBody"
            );
        }
        assert!(
            thrust.is_finite() && thrust > 0.0,
            "thrust 必须为正的有限值"
        );
        assert!(isp.is_finite() && isp > 0.0, "isp 必须为正的有限值");
        assert!(g0.is_finite() && g0 > 0.0, "g0 必须为正的有限值");
        assert!(
            fuel_weight.is_finite() && fuel_weight >= 0.0,
            "fuel_weight 必须为非负有限值"
        );
        if !with_mass {
            assert!(
                fixed_mass.is_finite() && fixed_mass > 0.0,
                "4 维模式 fixed_mass 必须为正的有限值"
            );
        }
    }

    /// 状态维数（4 或 5）。
    pub fn dim(&self) -> usize {
        if self.with_mass {
            5
        } else {
            4
        }
    }

    fn mass_of(&self, state: &[f64]) -> f64 {
        if self.with_mass {
            state[4]
        } else {
            self.fixed_mass
        }
    }

    /// 推力加速度上界（km/s²）：T/m 的 m/s² → km/s² 换算。
    fn max_accel(&self, mass: f64) -> f64 {
        self.thrust / (1000.0 * mass)
    }

    /// 燃料流量上界 T/(Isp·g0)（kg/s）。
    fn mass_flow(&self) -> f64 {
        self.thrust / (self.isp * self.g0)
    }

    /// 构造 t（求解器秒）时刻的帧量：查一次缓存（或解析星历），
    /// 导出 d、ḋ/d、d̈/d、ω、ω̇ 与面内基。
    pub fn frame(&self, t: f64) -> FrameState {
        let et = self.et0 + t;
        // （月球位置、速度、加速度，第三体原始位置表）。
        #[allow(clippy::type_complexity)]
        let (r_m, v_m, a_m, body_pos): (
            [f64; 3],
            [f64; 3],
            [f64; 3],
            Vec<(String, [f64; 3])>,
        ) = match &self.source {
            EphemerisSource::Cache => {
                let look = |target: &str| -> [f64; 3] {
                    match ephem_cache::lookup_body_position(target, OBSERVER, et) {
                        Ok(Some(p)) => p,
                        Ok(None) => {
                            panic!("EphemCache 未启用：查询 ({target}, {OBSERVER}) et={et} 失败")
                        }
                        Err(e) => {
                            panic!("EphemCache 查 ({target}, {OBSERVER}) et={et} 失败：{e}")
                        }
                    }
                };
                let r_m = look(FRAME_BODY);
                let v_m = match ephem_cache::lookup_body_velocity(FRAME_BODY, OBSERVER, et) {
                    Ok(Some(v)) => v,
                    Ok(None) => panic!("EphemCache 未启用：查询速度 {FRAME_BODY} et={et} 失败"),
                    Err(e) => panic!("EphemCache 查速度 {FRAME_BODY} et={et} 失败：{e}"),
                };
                let a_m = match ephem_cache::lookup_body_acceleration(FRAME_BODY, OBSERVER, et) {
                    Ok(Some(a)) => a,
                    Ok(None) => panic!("EphemCache 未启用：查加速度 {FRAME_BODY} et={et} 失败"),
                    Err(e) => panic!("EphemCache 查加速度 {FRAME_BODY} et={et} 失败：{e}"),
                };
                let mut body_pos = Vec::new();
                for f in &self.forces {
                    if let CompiledForce::ThirdBody { body, .. } = f {
                        if body != FRAME_BODY {
                            body_pos.push((body.clone(), look(body)));
                        }
                    }
                }
                (r_m, v_m, a_m, body_pos)
            }
            EphemerisSource::Synthetic(s) => {
                let body_pos = s
                    .bodies
                    .iter()
                    .filter(|(b, _)| b != FRAME_BODY)
                    .cloned()
                    .collect();
                (s.moon_pos, s.moon_vel, s.moon_accel, body_pos)
            }
        };

        let d = norm(&r_m);
        assert!(d > 0.0, "月地距为零，无法定义会合系");
        let e_r = [r_m[0] / d, r_m[1] / d, r_m[2] / d];
        let h_vec = cross(&r_m, &v_m);
        let h = norm(&h_vec);
        assert!(h > 0.0, "月球速度与位置共线，轨道面退化");
        let e_h = [h_vec[0] / h, h_vec[1] / h, h_vec[2] / h];
        let e_theta = cross(&e_h, &e_r);

        let d_dot = dot(&r_m, &v_m) / d;
        let d_ddot = (dot(&v_m, &v_m) + dot(&r_m, &a_m)) / d - d_dot * d_dot / d;
        let omega = h / (d * d);
        // ω = |h|/d²，dω/dt = (ê_h·(r×a))/d² − 2ω·ḋ/d。
        let omega_dot = dot(&cross(&r_m, &a_m), &e_h) / (d * d) - 2.0 * omega * d_dot / d;

        // 第三体位置投影到瞬时轨道面（ADR 0034 决策 1 的面内口径）；
        // 月球本身在面内（e_r 由它定义），直接用 r_m。
        let third_bodies = self
            .forces
            .iter()
            .filter_map(|f| match f {
                CompiledForce::ThirdBody { body, mu } => {
                    let raw = if body == FRAME_BODY {
                        r_m
                    } else {
                        body_pos
                            .iter()
                            .find(|(b, _)| b == body)
                            .map(|(_, p)| *p)
                            .unwrap_or_else(|| panic!("第三体 {body} 的星历位置缺失"))
                    };
                    let s = dot(&raw, &e_h);
                    let proj = [
                        raw[0] - s * e_h[0],
                        raw[1] - s * e_h[1],
                        raw[2] - s * e_h[2],
                    ];
                    Some((body.clone(), *mu, proj))
                }
                _ => None,
            })
            .collect();

        FrameState {
            d,
            d_ratio_dot: d_dot / d,
            d_ratio_ddot: d_ddot / d,
            omega,
            omega_dot,
            e_r,
            e_theta,
            third_bodies,
        }
    }

    /// 惯性合力在面内基下的分量（km/s²）与惯性位置 R（km）。
    ///
    /// 公式与 `CompiledForce::acceleration` 的 PointMass/ThirdBody 分支
    /// 逐项一致（含 MIN_DISTANCE 截断）；第三体用帧量中的面内投影位置。
    pub fn inertial_accel(&self, fs: &FrameState, x: f64, y: f64) -> ([f64; 3], [f64; 2]) {
        let r_scale = fs.d;
        let r_vec = [
            r_scale * (x * fs.e_r[0] + y * fs.e_theta[0]),
            r_scale * (x * fs.e_r[1] + y * fs.e_theta[1]),
            r_scale * (x * fs.e_r[2] + y * fs.e_theta[2]),
        ];
        let mut acc = [0.0_f64; 3];
        for f in &self.forces {
            match f {
                CompiledForce::PointMass { mu } => {
                    let r_norm = norm(&r_vec);
                    // HJB 网格节点可能恰落在大体位置（会合系把两主星钉在
                    // 固定坐标，奇数节点网格必命中）：中心点处点质量引力
                    // 物理奇异，取 0 贡献保证 CFL 步长不塌（传播场景轨迹
                    // 不会精确穿过天心，与此正则化无交集）。
                    if r_norm >= MIN_DISTANCE {
                        let inv_r3 = 1.0 / r_norm.powi(3);
                        for k in 0..3 {
                            acc[k] -= mu * r_vec[k] * inv_r3;
                        }
                    }
                }
                CompiledForce::ThirdBody { mu, .. } => {
                    let rb = self.third_body_pos(fs, f);
                    let mut diff = [0.0_f64; 3];
                    for k in 0..3 {
                        diff[k] = r_vec[k] - rb[k];
                    }
                    let d_norm = norm(&diff);
                    let b_norm = norm(&rb).max(MIN_DISTANCE);
                    // 节点落在第三体中心时（网格命中月球 (1,0)），
                    // 直接项的物理极限是 0（保留 indirect 项）；
                    // `third_body_acceleration` 的 MIN_DISTANCE 截断在此处
                    // 会产生 ~μ/ε² 的伪加速度压垮 CFL，必须取极限而非截断。
                    if d_norm >= MIN_DISTANCE {
                        let inv_d3 = 1.0 / d_norm.powi(3);
                        for k in 0..3 {
                            acc[k] -= mu * (diff[k] * inv_d3 + rb[k] / b_norm.powi(3));
                        }
                    } else {
                        for k in 0..3 {
                            acc[k] -= mu * rb[k] / b_norm.powi(3);
                        }
                    }
                }
                _ => unreachable!("构造期已校验力模型类型"),
            }
        }
        let a_rot = [dot(&acc, &fs.e_r), dot(&acc, &fs.e_theta)];
        (r_vec, a_rot)
    }

    fn third_body_pos(&self, fs: &FrameState, force: &CompiledForce) -> [f64; 3] {
        let CompiledForce::ThirdBody { body, .. } = force else {
            unreachable!("非 ThirdBody 分支");
        };
        fs.third_bodies
            .iter()
            .find(|(b, _, _)| b == body)
            .map(|(_, _, p)| *p)
            .unwrap_or_else(|| panic!("第三体 {body} 的帧量缺失"))
    }

    /// 无控漂移向量场 f_drift(t, state) = (ux, uy, u̇x, u̇y)（质量维不含）。
    pub fn drift_field(&self, t: f64, state: &[f64]) -> [f64; 4] {
        let fs = self.frame(t);
        self.drift_field_at(&fs, state)
    }

    fn drift_field_at(&self, fs: &FrameState, state: &[f64]) -> [f64; 4] {
        let [x, y, ux, uy] = [state[0], state[1], state[2], state[3]];
        // R = d·q，被 R̈ = A 除以脉动尺度 d 得无量纲坐标加速度。
        let (_, a_rot) = self.inertial_accel(fs, x, y);
        let ar = [a_rot[0] / fs.d, a_rot[1] / fs.d];
        let w = fs.omega;
        let wd = fs.omega_dot;
        let cent = w * w - fs.d_ratio_ddot;
        let puls = 2.0 * fs.d_ratio_dot;
        let adx = ar[0] + 2.0 * w * uy + wd * y + cent * x - puls * (ux - w * y);
        let ady = ar[1] - 2.0 * w * ux - wd * x + cent * y - puls * (uy + w * x);
        [ux, uy, adx, ady]
    }

    /// 单点 H\*(t, x, p)。
    pub fn hamiltonian_at(&self, t: f64, state: &[f64], p: &[f64]) -> f64 {
        let fs = self.frame(t);
        self.hamiltonian_at_frame(&fs, state, p)
    }

    fn hamiltonian_at_frame(&self, fs: &FrameState, state: &[f64], p: &[f64]) -> f64 {
        let f = self.drift_field_at(fs, state);
        let pv_norm = p[2].hypot(p[3]);
        let m = self.mass_of(state);
        let a_max = self.max_accel(m);
        let mut s = self.fuel_weight - a_max * pv_norm;
        if self.with_mass {
            s -= p[4] * self.mass_flow();
        }
        p[0] * f[0] + p[1] * f[1] + p[2] * f[2] + p[3] * f[3] + s.min(0.0)
    }

    /// 第 `dim` 维的耗散包络（推导见模块文档）。
    pub fn partial_bound_at(&self, t: f64, state: &[f64], dim: usize) -> f64 {
        let fs = self.frame(t);
        match dim {
            0 => state[2].abs(),
            1 => state[3].abs(),
            2 | 3 => {
                let f = self.drift_field_at(&fs, state);
                f[dim].abs() + self.max_accel(self.mass_of(state))
            }
            4 if self.with_mass => self.mass_flow(),
            _ => panic!("dim 超出状态维数 {}", self.dim()),
        }
    }
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Hamiltonian for EphemerisPlanar {
    fn hamiltonian(
        &self,
        t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        assert_eq!(grid.dim(), self.dim(), "EphemerisPlanar 状态维数不匹配");
        // 每个 t 只构造一次帧量（一次缓存查询），全节点复用。
        let fs = self.frame(t);
        let mut out = ArrayD::zeros(grid.shape());
        for (idx, o) in out.indexed_iter_mut() {
            let pos = [
                grid.axis(0)[idx[0]],
                grid.axis(1)[idx[1]],
                grid.axis(2)[idx[2]],
                grid.axis(3)[idx[3]],
            ];
            let mut state = [0.0_f64; 5];
            state[..4].copy_from_slice(&pos);
            if self.with_mass {
                state[4] = grid.axis(4)[idx[4]];
            }
            let mut pvec = [0.0_f64; 5];
            for d in 0..self.dim() {
                pvec[d] = p[d][idx.clone()];
            }
            *o = self.hamiltonian_at_frame(&fs, &state, &pvec);
        }
        out
    }

    fn partial_bound(
        &self,
        t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        _p_min: &[ArrayD<f64>],
        _p_max: &[ArrayD<f64>],
        dim: usize,
    ) -> ArrayD<f64> {
        assert_eq!(grid.dim(), self.dim(), "EphemerisPlanar 状态维数不匹配");
        let fs = self.frame(t);
        let ax: Vec<Vec<f64>> = (0..self.dim()).map(|d| grid.axis(d).to_vec()).collect();
        // 位置维：∂H/∂p_r = v，精确取速度轴坐标。速度维：|无控加速度| +
        // 推力项——T/m 取节点质量而非质量轴区间上界（逐点包络更紧，
        // 正确性与保守性不减）。质量维：∂H/∂p_m 在 δ*=1 时为常值
        // −T/(Isp·g0)，包络取其绝对值。
        match dim {
            0 => ArrayD::from_shape_fn(grid.shape(), |idx| ax[2][idx[2]].abs()),
            1 => ArrayD::from_shape_fn(grid.shape(), |idx| ax[3][idx[3]].abs()),
            2 | 3 => ArrayD::from_shape_fn(grid.shape(), |idx| {
                let mut state = [0.0_f64; 5];
                for d in 0..self.dim() {
                    state[d] = ax[d][idx[d]];
                }
                let f = self.drift_field_at(&fs, &state);
                f[dim].abs() + self.max_accel(self.mass_of(&state))
            }),
            4 if self.with_mass => ArrayD::from_shape_fn(grid.shape(), |_| self.mass_flow()),
            _ => panic!("dim 超出状态维数 {}", self.dim()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cr3bpSynodic;

    /// 地月点质量（km³/s²），与 e2m2e 数据常量同源量级。
    const MU_EARTH: f64 = 398600.435436;
    const MU_MOON: f64 = 4902.800066;
    const L: f64 = 384400.0;
    const MU_N: f64 = MU_MOON / (MU_EARTH + MU_MOON);

    /// 圆轨道角速度：ω² = (μE+μM)/L³（使退化极限严格满足 CR3BP 量纲）。
    fn omega() -> f64 {
        ((MU_EARTH + MU_MOON) / L.powi(3)).sqrt()
    }

    /// 圆化定常合成星历：月理圆轨道（d、ω 恒定，d̈=ω̇=0），太阳取面内
    /// 定点（μ 取 0，不影响动力学，只覆盖查询路径）。
    fn circular_ephemeris() -> SyntheticEphemeris {
        let moon_pos = [L, 0.0, 0.0];
        let moon_vel = [0.0, L * omega(), 0.0];
        let moon_accel = [-L * omega() * omega(), 0.0, 0.0];
        SyntheticEphemeris {
            moon_pos,
            moon_vel,
            moon_accel,
            bodies: vec![("SUN".to_string(), [1.5e8, 0.0, 0.0])],
        }
    }

    fn forces() -> Vec<CompiledForce> {
        vec![
            CompiledForce::PointMass { mu: MU_EARTH },
            CompiledForce::ThirdBody {
                body: "MOON".to_string(),
                mu: MU_MOON,
            },
            CompiledForce::ThirdBody {
                body: "SUN".to_string(),
                mu: 0.0,
            },
        ]
    }

    fn ham4() -> EphemerisPlanar {
        EphemerisPlanar::with_synthetic_ephemeris(
            forces(),
            circular_ephemeris(),
            0.0,
            1.0,
            300.0,
            9.80665,
            0.1,
            false,
            1000.0,
        )
    }

    /// 帧量退化：圆轨道下 d、ω 恒定，ḋ、d̈、ω̇ 全部为零。
    #[test]
    fn frame_circular_limit_is_constant() {
        let h = ham4();
        let fs = h.frame(123.4);
        assert!((fs.d - L).abs() < 1e-6);
        assert!((fs.omega - omega()).abs() < 1e-18);
        assert!(fs.d_ratio_dot.abs() < 1e-15);
        assert!(fs.d_ratio_ddot.abs() < 1e-15);
        assert!(fs.omega_dot.abs() < 1e-15);
    }

    /// 退化对拍（验收 a）：圆化定常星历下，向量场按量纲换算逐项还原
    /// Cr3bpSynodic——v_n = u/ω，a_n = u̇/ω²（推导见模块文档）。
    #[test]
    fn degenerates_to_cr3bp_synodic() {
        let h = ham4();
        let cr3bp = Cr3bpSynodic::new(MU_N, 0.5, 0.1);
        for k in 0..50 {
            let s = |i: usize| ((k * 11 + i * 17) % 89) as f64 / 44.0 - 1.0;
            let state = [s(0) + 0.5, s(1), s(2), s(3)];
            let f = h.drift_field(0.0, &state);
            // 同一物理轨迹的坐标换算：cr3bp 是质心系（地球在 −μ），
            // 速度维 v_n = u/ω。
            let g = cr3bp.vector_field([
                state[0] - MU_N,
                state[1],
                state[2] / omega(),
                state[3] / omega(),
            ]);
            for d in 0..4 {
                let expect = if d < 2 {
                    f[d] / omega()
                } else {
                    f[d] / (omega() * omega())
                };
                // 采样速度为 O(1)（对应会合速度数万），断言用相对容差。
                assert!(
                    (g[d] - expect).abs() < 1e-9 * (g[d].abs() + 1.0),
                    "状态 {state:?} 第 {d} 维：cr3bp {} vs 星历版换算 {expect}",
                    g[d]
                );
            }
        }
    }

    /// 开关函数含质量协态项（5 维）：p_v = 0、p_m 足够大时 S < 0，
    /// 控制贡献为 w − p_m·T/(Isp·g0)。
    #[test]
    fn switching_function_includes_mass_costate() {
        let h = EphemerisPlanar::with_synthetic_ephemeris(
            forces(),
            circular_ephemeris(),
            0.0,
            300.0,
            300.0,
            9.80665,
            0.5,
            true,
            1000.0,
        );
        let t = 0.0;
        let state = [0.8, 0.1, 0.2, -0.3, 900.0];
        let p = [1.0, -1.0, 0.0, 0.0, 1000.0];
        let hv = h.hamiltonian_at(t, &state, &p);
        let f = h.drift_field(t, &state);
        let mdot = h.mass_flow();
        let expect = p[0] * f[0] + p[1] * f[1] + p[2] * f[2] + p[3] * f[3] + (0.5 - p[4] * mdot);
        assert!((hv - expect).abs() < 1e-12, "{hv} vs {expect}");
        assert!(0.5 - p[4] * mdot < 0.0, "该 p_m 下应满推力");
    }

    /// partial_bound_at 覆盖 |∂H/∂p_dim|（4 维，随机采样，避开开关面）。
    #[test]
    fn partial_bound_covers_numerical_derivative() {
        let h = ham4();
        let eps = 1e-6;
        for k in 0..40 {
            let s = |i: usize| ((k * 7 + i * 13) % 97) as f64 / 48.0 - 1.0;
            let state = [s(0) + 0.5, s(1), s(2), s(3)];
            let p = [s(4), s(5), s(6) + 2.0, s(7) + 2.0];
            for dim in 0..4 {
                let mut p_up = p;
                let mut p_dn = p;
                p_up[dim] += eps;
                p_dn[dim] -= eps;
                let deriv = (h.hamiltonian_at(0.0, &state, &p_up)
                    - h.hamiltonian_at(0.0, &state, &p_dn))
                    / (2.0 * eps);
                let bound = h.partial_bound_at(0.0, &state, dim);
                assert!(
                    deriv.abs() <= bound + 1e-9,
                    "dim {dim}：|∂H/∂p| = {} 超过包络 {bound}",
                    deriv.abs()
                );
            }
        }
    }

    /// 5 维质量维包络：∂H/∂p_m = −δ*·T/(Isp·g0)，绝对值不超过流量上界。
    #[test]
    fn mass_partial_bound_covers_derivative() {
        let h = EphemerisPlanar::with_synthetic_ephemeris(
            forces(),
            circular_ephemeris(),
            0.0,
            300.0,
            300.0,
            9.80665,
            0.5,
            true,
            1000.0,
        );
        let eps = 1e-6;
        for p_m in [-5.0_f64, -0.1, 0.0, 0.1, 500.0] {
            let state = [0.8, 0.1, 0.2, -0.3, 900.0];
            let p = [0.3, -0.2, 1.5, -0.7, p_m];
            let mut p_up = p;
            let mut p_dn = p;
            p_up[4] += eps;
            p_dn[4] -= eps;
            let deriv = (h.hamiltonian_at(0.0, &state, &p_up)
                - h.hamiltonian_at(0.0, &state, &p_dn))
                / (2.0 * eps);
            assert!(deriv.abs() <= h.partial_bound_at(0.0, &state, 4) + 1e-6);
        }
    }

    /// 网格节点命中天体中心（月球在 (1, 0)）：第三体直接项取物理极限 0，
    /// 加速度与耗散包络保持有限（否则 CFL 步长塌为零、求解死循环）。
    #[test]
    fn grid_node_at_body_center_stays_finite() {
        let h = ham4();
        let fs = h.frame(0.0);
        // 月球中心：direct 项 0，剩地球引力与 indirect 项。
        let (_, a_rot) = h.inertial_accel(&fs, 1.0, 0.0);
        for d in 0..2 {
            assert!(
                a_rot[d].abs().is_finite() && a_rot[d].abs() < 1e-3,
                "{a_rot:?}"
            );
        }
        let f = h.drift_field_at(&fs, &[1.0, 0.0, 0.0, 0.0]);
        for d in 0..4 {
            assert!(f[d].abs() < 1.0, "向量场爆掉：{f:?}");
        }
    }

    /// 构造期拒绝非平面力模型与非法参数。
    #[test]
    #[should_panic]
    fn rejects_non_planar_force() {
        EphemerisPlanar::with_synthetic_ephemeris(
            vec![
                CompiledForce::PointMass { mu: MU_EARTH },
                CompiledForce::SRP {
                    area: 1.0,
                    mass: 100.0,
                    cr: 1.3,
                    shadow_bodies: vec![],
                },
            ],
            circular_ephemeris(),
            0.0,
            1.0,
            300.0,
            9.80665,
            0.5,
            false,
            1000.0,
        );
    }

    #[test]
    #[should_panic]
    fn rejects_nonpositive_thrust() {
        EphemerisPlanar::with_synthetic_ephemeris(
            forces(),
            circular_ephemeris(),
            0.0,
            -1.0,
            300.0,
            9.80665,
            0.5,
            false,
            1000.0,
        );
    }
}
