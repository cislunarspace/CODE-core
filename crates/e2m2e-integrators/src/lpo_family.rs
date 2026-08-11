//! LPO 族设计的纯 Rust 数值原子：初猜生成 + 振幅测量。
//!
//! 对齐 Python：
//! - `compute_lpo_initial_guess`（`family/lpo_initial_guess.py`）：从 L4/L5
//!   长周期线性化模态构造平面初猜；
//! - `_l45_distance`（`family/cr3bp_orbits.py`）：传播一个周期，返回距 L4/L5
//!   径向距离的最小/最大值。
//!
//! 复特征分解（`_triangular_modes` 的 `np.linalg.eig`）**不在 Rust 实现**：
//! L4/L5 模态（`omega_l`、`v_l`、`x_L`）是系统常量，不随候选点变，由 Python
//! 预计算后传参（见 `docs/plans/lpo-rust-rayon.md` 第五节）。两个函数只做
//! 纯数值组合，为 M3 网格搜索并行化提供候选点生成与评估。
//!
//! 传播复用 `e2m2e_forces::cr3bp::propagate_cr3bp`（无 STM，**不重写**）。

use e2m2e_forces::cr3bp::propagate_cr3bp;

/// 采样点数（对齐 Python `_l45_distance` 默认 `n_points=2000`）。
const N_POINTS: usize = 2000;
/// 传播最大步长（对齐 Python `Dynamics.DEFAULT_MAX_STEP = 0.01`，无量纲）。
const MAX_STEP: f64 = 0.01;

/// L4/L5 编号校验（4=L4，5=L5）。
fn is_triangular_point(point: u8) -> bool {
    point == 4 || point == 5
}

/// 构造 L4/L5 LPO 初猜状态（仅长周期模态）。
///
/// # 语义（对齐 `compute_lpo_initial_guess`）
///
/// 从 L4/L5 线性化矩阵的长周期模态 v_l（低频）构造仅含长周期分量的初猜，
/// 不含短周期和垂直模态——它们导致拟周期运动，阻碍全周期闭合修正的收敛。
///
/// ```text
/// alpha_l        = (amplitude_km / l_c) / ‖v_l[:3]‖
/// mode_contrib   = alpha_l · Re(v_l)          （phi = 0）
/// state0[0..3]   = x_L + mode_contrib[0..3]
/// state0[3..6]   = mode_contrib[3..6]
/// nominal_period = 2π / omega_l
/// ```
///
/// `phi = 0` 时 `Re(v_l)·cos0 - Im(v_l)·sin0 = Re(v_l)`，模态贡献只剩实部；
/// `‖v_l[:3]‖` 是 v_l 前 3 个复分量（位置部分）的模范数
/// `sqrt(Σ|v[i]|²)`，`|z|² = re² + im²`——与 `np.linalg.norm` 同一数学。
///
/// # 参数
/// - `point`：4（L4）或 5（L5）。模态已由调用方按该点取好，此处仅校验
/// - `amplitude_km`：面内振幅（km）
/// - `char_length_km`：`system.characteristic_length`（km）
/// - `omega_l`：长周期频率（无量纲，`_triangular_modes` 的 `omega_l`）
/// - `v_l`：长周期复特征向量（6 维）实部/虚部交错展平，长度 12：
///   `[re0, im0, re1, im1, ..., re5, im5]`
/// - `x_l`：L4/L5 的完整 3 维位置（等边三角形顶点，设计文档记作 `x_L`）
///
/// # 返回
/// `Ok((state0, nominal_period))`；`point` 非法、`v_l` 长度不为 12、特征长度 /
/// 频率 / 振幅非有限或非正，返回 `Err`。
pub fn lpo_initial_guess(
    point: u8,
    amplitude_km: f64,
    char_length_km: f64,
    omega_l: f64,
    v_l: &[f64],
    x_l: &[f64; 3],
) -> Result<([f64; 6], f64), String> {
    if !is_triangular_point(point) {
        return Err(format!("point 必须为 4（L4）或 5（L5），当前 {point}"));
    }
    if v_l.len() != 12 {
        return Err(format!(
            "v_l 须为 12 个 f64（6 维复向量实部/虚部交错展平），当前 {}",
            v_l.len()
        ));
    }
    if !char_length_km.is_finite() || char_length_km <= 0.0 {
        return Err(format!(
            "特征长度无效（char_length_km={char_length_km:.3}）"
        ));
    }
    if !omega_l.is_finite() || omega_l <= 0.0 {
        return Err(format!("长周期频率无效（omega_l={omega_l:.6}）"));
    }
    if !amplitude_km.is_finite() {
        return Err(format!("振幅无效（amplitude_km={amplitude_km:.3}）"));
    }

    // ‖v_l[:3]‖ = sqrt(Σ re² + im²)，i ∈ 0..3（np.linalg.norm 的前 3 复分量）
    let mut v_norm_sq = 0.0;
    for i in 0..3 {
        let re = v_l[2 * i];
        let im = v_l[2 * i + 1];
        v_norm_sq += re * re + im * im;
    }
    let alpha_l = (amplitude_km / char_length_km) / v_norm_sq.sqrt();

    // mode_contrib[i] = alpha_l · Re(v_l[i]) = alpha_l · v_l[2*i]
    let mut state0 = [0.0_f64; 6];
    for i in 0..3 {
        state0[i] = x_l[i] + alpha_l * v_l[2 * i];
    }
    for i in 3..6 {
        state0[i] = alpha_l * v_l[2 * i];
    }
    let nominal_period = 2.0 * std::f64::consts::PI / omega_l;
    Ok((state0, nominal_period))
}

/// 传播一个周期，返回距 L4/L5 径向距离的最小/最大值（无量纲）。
///
/// # 语义（对齐 `_l45_distance`）
///
/// 1. L4/L5 的平面投影点：`lp_x = 0.5 - mu`，`lp_y = ±√3/2`（L4 正、L5 负）；
/// 2. `t_eval = linspace(0, period, 2000)`（numpy 语义：含两端点，末点=period）；
/// 3. `propagate_cr3bp` 无 STM 传播一个周期，取 2000 个采样状态；
/// 4. 逐点 xy 径向距离 `sqrt((x-lp_x)² + (y-lp_y)²)`，返回 min / max。
///
/// 传播参数对齐 Python `CR3BP_Dynamics`：`max_step = 0.01`
/// （`DEFAULT_MAX_STEP`）；容差由调用方传入（Python 默认 `rtol=atol=1e-12`）。
///
/// # 参数
/// - `mu`：CR3BP 质量参数 μ
/// - `point`：4（L4）或 5（L5）
/// - `state0`：初始状态（已收敛轨道首点）
/// - `period`：全周期（无量纲）
/// - `rtol` / `atol`：传播容差（传给 `propagate_cr3bp`）
///
/// # 返回
/// `Ok((d_min, d_max))`；`point` 非法、周期 / 容差非正或传播失败返回 `Err`。
pub fn l45_distance(
    mu: f64,
    point: u8,
    state0: &[f64; 6],
    period: f64,
    rtol: f64,
    atol: f64,
) -> Result<(f64, f64), String> {
    if !is_triangular_point(point) {
        return Err(format!("point 必须为 4（L4）或 5（L5），当前 {point}"));
    }
    if !period.is_finite() || period <= 0.0 {
        return Err(format!("周期无效（period={period:.6}）"));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(format!("传播容差无效（rtol={rtol:.1e}, atol={atol:.1e}）"));
    }

    let lp_x = 0.5 - mu;
    let lp_y = if point == 4 {
        3.0_f64.sqrt() / 2.0
    } else {
        -3.0_f64.sqrt() / 2.0
    };

    // t_eval = linspace(0, period, 2000)：i·step，末点强制为 period（numpy 端点含闭）
    let step = period / (N_POINTS as f64 - 1.0);
    let mut t_eval = Vec::with_capacity(N_POINTS);
    for i in 0..N_POINTS {
        t_eval.push(i as f64 * step);
    }
    t_eval[N_POINTS - 1] = period;

    let result = propagate_cr3bp(
        mu,
        (0.0, period),
        &t_eval,
        state0,
        rtol,
        atol,
        Some(MAX_STEP),
        None,
    )?;

    let mut d_min = f64::INFINITY;
    let mut d_max = 0.0_f64;
    for state in &result.states {
        let dx = state[0] - lp_x;
        let dy = state[1] - lp_y;
        let d = (dx * dx + dy * dy).sqrt();
        if d < d_min {
            d_min = d;
        }
        if d > d_max {
            d_max = d;
        }
    }
    Ok((d_min, d_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 地月系质量参数（对齐 `earth_moon_system()` 的 `EARTH_MOON_MU`）。
    const MU_EARTH_MOON: f64 = 0.0121506683;
    /// 特征长度（km，`CHAR_LENGTH_KM`）。
    const CHAR_LENGTH_KM: f64 = 384400.0;

    /// L4 长周期模态常量（Python `_triangular_modes(earth_moon_system, 4)` 预计算）：
    /// 频率 `omega_l`、复特征向量 `v_l`（实/虚交错 12 个 f64）、锚点 `x_L`。
    const OMEGA_L_L4: f64 = 0.2982092837049938;
    const V_L_L4: [f64; 12] = [
        -0.8221319912182276,
        0.0,
        0.4455148161239162,
        -0.20964075061607065,
        6.795037248874498e-17,
        -1.327803042151104e-17,
        8.062878806512508e-16,
        -0.24516739221214762,
        0.06251681807659477,
        0.13285665419627535,
        -7.726763846400462e-17,
        1.509869654199918e-17,
    ];
    const X_L_L4: [f64; 3] = [0.4878493317, 0.8660254037844386, 0.0];

    /// L5 长周期模态常量（Python 预计算）：x/z 分量不变，y 分量取负（镜像）。
    const V_L_L5: [f64; 12] = [
        -0.8221319912182276,
        0.0,
        -0.4455148161239162,
        -0.20964075061607065,
        6.795037248874498e-17,
        1.327803042151104e-17,
        -8.062878806512508e-16,
        -0.24516739221214762,
        0.06251681807659477,
        -0.13285665419627535,
        7.726763846400462e-17,
        1.509869654199918e-17,
    ];
    const X_L_L5: [f64; 3] = [0.4878493317, -0.8660254037844386, 0.0];

    /// Python `compute_lpo_initial_guess(earth_moon_system, 4, 1000.0)` 的
    /// 参考输出（state0, nominal_period）。
    const REF_STATE4: [f64; 6] = [
        0.48561751816786186,
        0.8672348275825809,
        1.8446254671283525e-19,
        2.188802068351565e-18,
        0.00016971226282389845,
        -2.0975580924030585e-19,
    ];
    const REF_PERIOD4: f64 = 21.069717310998552;

    /// Python `compute_lpo_initial_guess(earth_moon_system, 5, 1000.0)` 参考输出。
    const REF_STATE5: [f64; 6] = [
        0.48561751816786186,
        -0.8672348275825809,
        1.8446254671283525e-19,
        -2.188802068351565e-18,
        0.00016971226282389845,
        2.0975580924030585e-19,
    ];

    /// M1 收敛参考轨道（`lpo_correction.rs` 的 `REF_STATE`/`REF_PERIOD`），
    /// 供 `l45_distance` 对齐 Python `_l45_distance`。
    const CORRECTED_STATE: [f64; 6] = [
        0.48561751816786186,
        0.8672314929275382,
        0.0,
        -3.4846216971678926e-07,
        0.00016886669249480745,
        0.0,
    ];
    const CORRECTED_PERIOD: f64 = 21.069729764938614;

    /// L4 初猜轨道（`lpo_initial_guess` 输出，z₀=ż₀≈0 平面）。
    const GUESS_STATE: [f64; 6] = [
        0.48561751816786186,
        0.8672348275825809,
        0.0,
        0.0,
        0.00016971226282389845,
        0.0,
    ];
    const GUESS_PERIOD: f64 = 21.069717310998552;

    /// 初猜复现 Python：L4 状态与名义周期逐分量对齐。
    #[test]
    fn lpo_initial_guess_matches_python_l4() {
        let (state0, period) =
            lpo_initial_guess(4, 1000.0, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4, &X_L_L4)
                .expect("L4 初猜应成功");
        for i in 0..6 {
            assert!(
                (state0[i] - REF_STATE4[i]).abs() < 1e-14,
                "state0[{}] = {} vs Python 参考 {}",
                i,
                state0[i],
                REF_STATE4[i]
            );
        }
        assert!(
            (period - REF_PERIOD4).abs() < 1e-14,
            "period {} vs Python 参考 {}",
            period,
            REF_PERIOD4
        );
    }

    /// 初猜复现 Python：L5 是 L4 的 xy 镜像（y 取负、速度分量符号翻转）。
    #[test]
    fn lpo_initial_guess_matches_python_l5() {
        let (state0, period) = lpo_initial_guess(
            5,
            1000.0,
            CHAR_LENGTH_KM,
            OMEGA_L_L4, // L4/L5 频率相同
            &V_L_L5,
            &X_L_L5,
        )
        .expect("L5 初猜应成功");
        for i in 0..6 {
            assert!(
                (state0[i] - REF_STATE5[i]).abs() < 1e-14,
                "state0[{}] = {} vs Python 参考 {}",
                i,
                state0[i],
                REF_STATE5[i]
            );
        }
        assert!(
            (period - REF_PERIOD4).abs() < 1e-14,
            "L5 周期应与 L4 一致（{} vs {}）",
            period,
            REF_PERIOD4
        );
    }

    /// 非法输入：point、v_l 长度、特征长度 / 频率 / 振幅非正均报 Err。
    #[test]
    fn lpo_initial_guess_rejects_bad_input() {
        assert!(
            lpo_initial_guess(3, 1000.0, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4, &X_L_L4).is_err()
        );
        assert!(
            lpo_initial_guess(6, 1000.0, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4, &X_L_L4).is_err()
        );
        assert!(
            lpo_initial_guess(4, 1000.0, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4[..6], &X_L_L4)
                .is_err()
        );
        assert!(lpo_initial_guess(4, 1000.0, 0.0, OMEGA_L_L4, &V_L_L4, &X_L_L4).is_err());
        assert!(lpo_initial_guess(4, 1000.0, CHAR_LENGTH_KM, 0.0, &V_L_L4, &X_L_L4).is_err());
        assert!(
            lpo_initial_guess(4, f64::NAN, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4, &X_L_L4).is_err()
        );
        // 负振幅 Python 不报错（镜像初猜），Rust 与之对齐，允许
        assert!(
            lpo_initial_guess(4, -1000.0, CHAR_LENGTH_KM, OMEGA_L_L4, &V_L_L4, &X_L_L4).is_ok()
        );
    }

    /// 振幅测量的正确轨道：对齐 Python `_l45_distance(dynamics, orbit, 4)`。
    #[test]
    fn l45_distance_matches_python_corrected_l4() {
        let (d_min, d_max) = l45_distance(
            MU_EARTH_MOON,
            4,
            &CORRECTED_STATE,
            CORRECTED_PERIOD,
            1e-12,
            1e-12,
        )
        .expect("L4 距离测量应成功");
        // Python 参考：(0.0004965659002128372, 0.002551843909532946)
        assert!(
            (d_min - 0.0004965659002128372).abs() < 1e-12,
            "d_min {} vs Python 参考 0.0004965659002128372",
            d_min
        );
        assert!(
            (d_max - 0.002551843909532946).abs() < 1e-12,
            "d_max {} vs Python 参考 0.002551843909532946",
            d_max
        );
        assert!(d_min > 0.0 && d_max > d_min, "min/max 应为正且有序");
    }

    /// 振幅测量的初猜轨道：同样对齐 Python（未修正轨道也应有意义的值）。
    #[test]
    fn l45_distance_matches_python_guess() {
        let (d_min, d_max) =
            l45_distance(MU_EARTH_MOON, 4, &GUESS_STATE, GUESS_PERIOD, 1e-12, 1e-12)
                .expect("L4 初猜距离测量应成功");
        // Python 参考：(0.00048452589905836587, 0.0025602697423579806)
        assert!(
            (d_min - 0.00048452589905836587).abs() < 1e-12,
            "d_min {} vs Python 参考 0.00048452589905836587",
            d_min
        );
        assert!(
            (d_max - 0.0025602697423579806).abs() < 1e-12,
            "d_max {} vs Python 参考 0.0025602697423579806",
            d_max
        );
    }

    /// L5 镜像轨道：距离应接近 L4（验证 lp_y 取负）。
    #[test]
    fn l45_distance_l5_symmetry() {
        // L4 收敛轨道做 xy 镜像（y→-y、vx→-vx），对 L5 用同一测量
        let l5_state = [
            CORRECTED_STATE[0],
            -CORRECTED_STATE[1],
            0.0,
            -CORRECTED_STATE[3],
            CORRECTED_STATE[4],
            0.0,
        ];
        let (d_min, d_max) =
            l45_distance(MU_EARTH_MOON, 5, &l5_state, CORRECTED_PERIOD, 1e-12, 1e-12)
                .expect("L5 距离测量应成功");
        // Python 参考：(0.0004965659002153902, 0.0025518439092240297)
        assert!(
            (d_min - 0.0004965659002153902).abs() < 1e-12,
            "d_min {} vs Python 参考 0.0004965659002153902",
            d_min
        );
        assert!(
            (d_max - 0.0025518439092240297).abs() < 1e-12,
            "d_max {} vs Python 参考 0.0025518439092240297",
            d_max
        );
    }

    /// 非法输入：point、周期 / 容差非正均报 Err。
    #[test]
    fn l45_distance_rejects_bad_input() {
        assert!(l45_distance(
            MU_EARTH_MOON,
            3,
            &CORRECTED_STATE,
            CORRECTED_PERIOD,
            1e-12,
            1e-12
        )
        .is_err());
        assert!(l45_distance(MU_EARTH_MOON, 4, &CORRECTED_STATE, 0.0, 1e-12, 1e-12).is_err());
        assert!(l45_distance(
            MU_EARTH_MOON,
            4,
            &CORRECTED_STATE,
            CORRECTED_PERIOD,
            0.0,
            1e-12
        )
        .is_err());
        assert!(l45_distance(
            MU_EARTH_MOON,
            4,
            &CORRECTED_STATE,
            CORRECTED_PERIOD,
            1e-12,
            0.0
        )
        .is_err());
    }
}
