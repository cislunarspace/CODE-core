//! J2000 惯性系 N 体力模型 + STM 变分方程的 Rust 实现。
//!
//! 从 Python `EphemerisDynamics` 迁移，实现：
//! 1. `compute_nbody_acceleration_and_jacobian`：单次遍历所有天体，同时计算
//!    加速度 a 和雅可比 ∂a/∂r（从 `_compute_acc_and_jacobian` 迁移）
//! 2. `propagate_with_stm`：42 维增广状态传播（6 状态 + 36 STM 展平），
//!    使用 `solve_ivp_capped` + STM 变分方程（dΦ/dt = A·Φ）
//!
//! ## 物理模型
//! 以 `origin` 为坐标原点的受限 N 体问题：
//! ```text
//! a = -μ₀·r/|r|³ - Σᵢ μᵢ·[(r-rᵢ)/|r-rᵢ|³ + rᵢ/|rᵢ|³]
//! ```
//!
//! ## STM 变分方程
//! 增广状态：`[r(3), v(3), Φ(36)]`，共 42 维。
//! ```text
//! dΦ/dt = A · Φ,  A = [0₃ₓ₃  I₃ₓ₃; ∂a/∂r  0₃ₓ₃]
//! ```

use e2m2e_spice::spk_accel;

/// 最小距离钳位（km），防止除零。
pub const MIN_DISTANCE: f64 = 1e-6;

/// N 体力模型配置：描述天体列表和原点天体。
pub struct NBodyConfig {
    /// 天体名称列表（如 `["EARTH", "MOON", "SUN"]`）。
    pub bodies: Vec<String>,
    /// 原点天体名称（如 `"EARTH"`）。
    pub origin: String,
    /// 各天体的 GM（km³/s²），与 `bodies` 一一对应。
    pub gm_values: Vec<f64>,
}

/// 单次遍历所有天体，同时计算加速度和雅可比矩阵。
///
/// 从 Python `EphemerisDynamics._compute_acc_and_jacobian` 迁移。
///
/// # 参数
/// - `config`: N 体力模型配置（天体列表 + 原点 + GM）
/// - `et`: SPICE et 秒
/// - `r_sc`: 航天器位置 [x, y, z] km（相对原点天体）
///
/// # 返回
/// `(acc, jacobian)`：加速度 [f64; 3] 和雅可比 [[f64; 3]; 3]。
pub fn compute_nbody_acceleration_and_jacobian(
    config: &NBodyConfig,
    et: f64,
    r_sc: &[f64; 3],
) -> Result<([f64; 3], [[f64; 3]; 3]), String> {
    let mut acc = [0.0_f64; 3];
    let mut jac = [[0.0_f64; 3]; 3];

    for (body, gm) in config.bodies.iter().zip(config.gm_values.iter()) {
        if body == &config.origin {
            // 中心天体：a = -μ·r/|r|³
            let r_norm_sq = r_sc[0] * r_sc[0] + r_sc[1] * r_sc[1] + r_sc[2] * r_sc[2];
            let r_norm = r_norm_sq.sqrt();
            let r_safe = if r_norm < MIN_DISTANCE {
                MIN_DISTANCE
            } else {
                r_norm
            };
            let inv_r3 = 1.0 / (r_safe * r_safe * r_safe);
            let inv_r5 = inv_r3 / (r_safe * r_safe);

            for i in 0..3 {
                acc[i] -= gm * r_sc[i] * inv_r3;
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    jac[i][j] -= gm * (delta * inv_r3 - 3.0 * r_sc[i] * r_sc[j] * inv_r5);
                }
            }
        } else {
            // 摄动天体：用 spk_accel 计算加速度 + 雅可比
            let (a_body, jac_body) = spk_accel::third_body_acceleration_and_jacobian(
                et,
                body,
                &config.origin,
                r_sc,
                *gm,
                MIN_DISTANCE,
            )
            .map_err(|e| format!("SPICE query failed for {}: {:?}", body, e))?;

            for i in 0..3 {
                acc[i] += a_body[i];
                for j in 0..3 {
                    jac[i][j] += jac_body[i][j];
                }
            }
        }
    }

    Ok((acc, jac))
}

/// 计算状态方程的雅可比矩阵 A(t)。
///
/// A = | 0₃ₓ₃  I₃ₓ₃ |
///     | U₃ₓ₃  0₃ₓ₃ |
///
/// 其中 U = ∂a/∂r。
pub fn compute_jacobian_A(jac_da_dr: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut a = [[0.0_f64; 6]; 6];
    // A[0:3, 3:6] = I
    for i in 0..3 {
        a[i][i + 3] = 1.0;
    }
    // A[3:6, 0:3] = ∂a/∂r
    for i in 0..3 {
        for j in 0..3 {
            a[i + 3][j] = jac_da_dr[i][j];
        }
    }
    a
}

/// STM 变分方程的右端项：dΦ/dt = A · Φ。
///
/// # 参数
/// - `stm`: 6×6 状态转移矩阵（展平为 36 维）
/// - `jac_da_dr`: 加速度对位置的偏导数（3×3）
///
/// # 返回
/// dΦ/dt（展平为 36 维）
pub fn stm_derivative(stm: &[f64; 36], jac_da_dr: &[[f64; 3]; 3]) -> [f64; 36] {
    let mut dstm = [0.0_f64; 36];
    // dΦ/dt = A · Φ
    // A = [0, I; U, 0]，其中 U = ∂a/∂r
    //
    // 对于 Φ 的第 j 列（col_j），有：
    //   d(col_j)/dt = A · col_j
    //   前3行：= col_j[3:6]（来自 I 块）
    //   后3行：= U · col_j[0:3]（来自 U 块）
    for col in 0..6 {
        // 前 3 行：dstm[row][col] = stm[row+3][col]（即 I·Φ 的上半部分）
        for row in 0..3 {
            dstm[row * 6 + col] = stm[(row + 3) * 6 + col];
        }
        // 后 3 行：dstm[row+3][col] = Σ_k U[row][k] * stm[k][col]
        for row in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += jac_da_dr[row][k] * stm[k * 6 + col];
            }
            dstm[(row + 3) * 6 + col] = sum;
        }
    }
    dstm
}

/// 42 维增广状态（6 状态 + 36 STM 展平）的右端项。
///
/// 供 `solve_ivp_capped` 使用的回调函数。
///
/// # 增广状态布局
/// - `[0:3]`：位置 r（km）
/// - `[3:6]`：速度 v（km/s）
/// - `[6:42]`：STM Φ（6×6 展平，行优先）
///
/// # 返回
/// `[v(3), a(3), dΦ/dt(36)]`，共 42 维。
pub fn augmented_eom(
    config: &NBodyConfig,
    et: f64,
    augmented_state: &[f64; 42],
) -> Result<[f64; 42], String> {
    let r_sc = [augmented_state[0], augmented_state[1], augmented_state[2]];
    let v = [augmented_state[3], augmented_state[4], augmented_state[5]];
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&augmented_state[6..42]);

    // 计算加速度和雅可比
    let (acc, jac_da_dr) = compute_nbody_acceleration_and_jacobian(config, et, &r_sc)?;

    // STM 变分方程
    let dstm = stm_derivative(&stm, &jac_da_dr);

    // 组装增广状态导数
    let mut result = [0.0_f64; 42];
    // dr/dt = v
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    // dv/dt = a
    result[3] = acc[0];
    result[4] = acc[1];
    result[5] = acc[2];
    // dΦ/dt
    result[6..42].copy_from_slice(&dstm);

    Ok(result)
}

/// 传播结果：状态轨迹 + STM 序列。
pub struct PropagationResult {
    /// 各 `t_eval` 时刻的状态向量 `[x, y, z, vx, vy, vz]`（km, km/s）。
    pub states: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的 STM 矩阵（6×6 展平，行优先）。
    pub stms: Vec<[f64; 36]>,
    /// 实际输出的时间点。
    pub times: Vec<f64>,
}

/// 42 维增广状态传播（状态 + STM）。
///
/// 初始 STM 设为单位矩阵，拼接为 42 维增广状态后用 DOP853 积分。
/// 步长误差控制只统计前 6 维（`error_dim = 6`），避免 STM 分量主导步长选择。
///
/// # 参数
/// - `config`: N 体力模型配置
/// - `t_span`: 积分区间 `(t_start, t_end)`（SPICE et 秒）
/// - `t_eval`: 输出时间点（必须在 `t_span` 内且单调递增）
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`（km, km/s）
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长（秒），`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// `PropagationResult`：每个 `t_eval` 对应的状态和 STM。
///
/// # 错误
/// - 初值处右端项求值失败（如 SPICE 内核缺失导致第三体位置查询失败）；
/// - 积分提前退出导致输出点数少于 `t_eval.len()`（如步长塌缩、中途力模型失败）。
pub fn propagate_with_stm(
    config: &NBodyConfig,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<PropagationResult, String> {
    use e2m2e_propagation::solve_ivp::solve_ivp_capped;

    // 构造 42 维增广状态：[r(3), v(3), Φ(36)]
    let mut augmented0 = [0.0_f64; 42];
    augmented0[..6].copy_from_slice(initial_state);
    // 单位 STM
    for i in 0..6 {
        augmented0[6 + i * 6 + i] = 1.0;
    }

    // 预检：初值处右端项必须可求值。SPICE 内核缺失时第三体位置查询
    // 在此处确定性失败，避免积分静默返回截断结果（issue #246）。
    augmented_eom(config, t_span.0, &augmented0)
        .map_err(|e| format!("initial RHS evaluation failed at t={}: {}", t_span.0, e))?;

    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    // 积分
    let sol = solve_ivp_capped(
        |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            let mut arr = [0.0_f64; 42];
            arr.copy_from_slice(y);
            let result = augmented_eom(config, t, &arr)?;
            Ok(result.to_vec())
        },
        t_span,
        &augmented0,
        t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        Some(6), // 只统计前 6 维误差
    );

    // 完整性校验：solve_ivp_capped 在力模型失败/步长塌缩时会提前退出，
    // 输出点数不足即视为传播失败，不允许静默截断（issue #246）。
    if sol.len() != t_eval.len() {
        return Err(format!(
            "propagation truncated: got {} of {} time points (t_span=({:.3}, {:.3})); \
             likely cause: SPICE kernels not loaded or step size collapsed",
            sol.len(),
            t_eval.len(),
            t_span.0,
            t_span.1,
        ));
    }

    // 分离状态和 STM
    let mut states = Vec::with_capacity(sol.len());
    let mut stms = Vec::with_capacity(sol.len());
    let mut times = Vec::with_capacity(sol.len());

    for (i, y) in sol.iter().enumerate() {
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(&y[..6]);
        states.push(s);

        let mut stm = [0.0_f64; 36];
        stm.copy_from_slice(&y[6..42]);
        stms.push(stm);

        times.push(t_eval[i]);
    }

    Ok(PropagationResult {
        states,
        stms,
        times,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_kernels() {
        let kernel_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("kernels");
        for name in [
            "naif0012.tls",
            "pck00010.tpc",
            "de430.bsp",
            "de440s.bsp",
            "earth_latest_high_prec.bpc",
            "SPICEEarthPredictedKernel.bpc",
            "SPICELunaFrameKernel.tf",
            "SPICELunaCurrentKernel.bpc",
        ] {
            let path = kernel_dir.join(name);
            if path.exists() {
                let _ = cspice::data::furnish(path.to_string_lossy().to_string());
            }
        }
    }

    /// 地月系配置：EARTH 为中心，MOON 为摄动体。
    fn earth_moon_config() -> NBodyConfig {
        NBodyConfig {
            bodies: vec!["EARTH".to_string(), "MOON".to_string()],
            origin: "EARTH".to_string(),
            // GM 值（km³/s²）：JPL DE430
            gm_values: vec![398600.435436, 4902.800066],
        }
    }

    /// 地月日配置。
    fn earth_moon_sun_config() -> NBodyConfig {
        NBodyConfig {
            bodies: vec!["EARTH".to_string(), "MOON".to_string(), "SUN".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436, 4902.800066, 132712440041.9394],
        }
    }

    // =========================================================
    // 测试 1：中心天体引力 + 雅可比数值验证
    // =========================================================

    /// 纯中心引力测试：只用 EARTH，验证对称性。
    #[test]
    fn central_body_acceleration_basic() {
        load_kernels();
        // 只用 EARTH 作为中心天体（无摄动体），验证纯中心引力的对称性
        let config = NBodyConfig {
            bodies: vec!["EARTH".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436],
        };
        let et = 0.0;
        let r = [7000.0, 0.0, 0.0]; // LEO

        let (acc, jac) = compute_nbody_acceleration_and_jacobian(&config, et, &r).unwrap();

        // a_x ≈ -μ/r² = -398600/7000² ≈ -8.13e-3 km/s²
        assert!(acc[0] < 0.0, "中心引力应指向 -x");
        assert!(acc[0].abs() > 1e-3, "LEO 处引力应有量级 1e-3 km/s²");
        assert!(acc[1].abs() < 1e-14, "对称性：y 方向应为 0");
        assert!(acc[2].abs() < 1e-14, "对称性：z 方向应为 0");

        // 雅可比数值验证：用有限差分
        let h = 1e-5; // km
        for dim in 0..3 {
            let mut r_plus = r;
            let mut r_minus = r;
            r_plus[dim] += h;
            r_minus[dim] -= h;

            let (acc_plus, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_plus).unwrap();
            let (acc_minus, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_minus).unwrap();

            for i in 0..3 {
                let fd = (acc_plus[i] - acc_minus[i]) / (2.0 * h);
                let analytical = jac[i][dim];
                let err = (fd - analytical).abs();
                let scale = fd.abs().max(analytical.abs()).max(1e-20);
                assert!(
                    err / scale < 1e-6,
                    "jac[{}][{}] 数值={}, 解析={}, 相对误差={}",
                    i,
                    dim,
                    fd,
                    analytical,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 2：第三体摄动 + 雅可比数值验证
    // =========================================================

    /// 第三体摄动雅可比数值验证（含 EARTH+MOON+SUN）。
    #[test]
    fn third_body_jacobian_numerical() {
        load_kernels();
        let config = earth_moon_sun_config();
        let et = 0.0; // J2000
        let r = [7000.0, 0.0, 0.0]; // LEO

        let (_, jac) = compute_nbody_acceleration_and_jacobian(&config, et, &r).unwrap();

        // 有限差分验证（中心天体 + 第三体一起测）
        let h = 1e-3; // km
        for dim in 0..3 {
            let mut r_plus = r;
            let mut r_minus = r;
            r_plus[dim] += h;
            r_minus[dim] -= h;

            let (acc_plus, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_plus).unwrap();
            let (acc_minus, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_minus).unwrap();

            for i in 0..3 {
                let fd = (acc_plus[i] - acc_minus[i]) / (2.0 * h);
                let analytical = jac[i][dim];
                let err = (fd - analytical).abs();
                let scale = fd.abs().max(analytical.abs()).max(1e-20);
                // 容差放宽到 5e-2（有限差分精度 + SPICE 插值误差）
                assert!(
                    err / scale < 5e-2,
                    "jac[{}][{}] 数值={}, 解析={}, 相对误差={}",
                    i,
                    dim,
                    fd,
                    analytical,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 3：STM 导数矩阵维度和对称性
    // =========================================================

    #[test]
    fn stm_derivative_dimensions() {
        // 构造单位 STM
        let mut stm = [0.0_f64; 36];
        for i in 0..6 {
            stm[i * 6 + i] = 1.0;
        }

        // 构造一个简单的雅可比
        let jac_da_dr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let dstm = stm_derivative(&stm, &jac_da_dr);

        // 验证：dΦ/dt = A·Φ = A·I = A
        // A = [0₃ₓ₃  I₃ₓ₃]
        //     [U₃ₓ₃  0₃ₓ₃]
        //
        // 前 3 行：dstm[row][col] = δ(col, row+3)（来自 I 块）
        for row in 0..3 {
            for col in 0..6 {
                let expected = if col == row + 3 { 1.0 } else { 0.0 };
                assert!(
                    (dstm[row * 6 + col] - expected).abs() < 1e-14,
                    "dstm[{}][{}] = {} vs {}",
                    row,
                    col,
                    dstm[row * 6 + col],
                    expected
                );
            }
        }

        // 后 3 行：dstm[row+3][col] = U[row][col]（因为 Φ=I，U·I = U）
        // 注意：A 的后 3 行前 3 列是 U，后 3 列是 0
        for row in 0..3 {
            for col in 0..6 {
                let expected = if col < 3 {
                    jac_da_dr[row][col] // U 块
                } else {
                    0.0 // 0 块
                };
                assert!(
                    (dstm[(row + 3) * 6 + col] - expected).abs() < 1e-14,
                    "dstm[{}][{}] = {} vs {}",
                    row + 3,
                    col,
                    dstm[(row + 3) * 6 + col],
                    expected
                );
            }
        }
    }

    // =========================================================
    // 测试 4：42 维增广右端项基本正确性
    // =========================================================

    #[test]
    fn augmented_eom_basic() {
        load_kernels();
        let config = earth_moon_sun_config();
        let et = 0.0;

        // 构造增广状态：LEO 位置 + 圆轨道速度 + 单位 STM
        let mut state = [0.0_f64; 42];
        state[0] = 7000.0; // x = 7000 km
        state[4] = 7.5; // vy ≈ 圆轨道速度 km/s
                        // 单位 STM
        for i in 0..6 {
            state[6 + i * 6 + i] = 1.0;
        }

        let result = augmented_eom(&config, et, &state).unwrap();

        // dr/dt = v: vx=0, vy=7.5, vz=0
        assert!((result[0] - 0.0).abs() < 1e-10, "dx/dt = vx = 0");
        assert!((result[1] - 7.5).abs() < 1e-10, "dy/dt = vy = 7.5");
        assert!((result[2] - 0.0).abs() < 1e-10, "dz/dt = vz = 0");

        // dv/dt = a
        assert!(result[3] < 0.0, "ax 应为负（指向地心）");
        assert!(result[3].abs() > 1e-3, "LEO 加速度量级");

        // dΦ/dt 应非零（因为 A ≠ 0）
        let dstm_norm: f64 = result[6..42].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(dstm_norm > 0.0, "dΦ/dt 不应全为零");
    }

    // =========================================================
    // 测试 5：STM 传播与有限差分对比（端到端）
    // =========================================================

    /// STM 传播与有限差分对比（端到端）。
    ///
    /// 传播 1 个轨道周期（~5400 秒），让 STM 明显偏离单位矩阵，
    /// 然后用有限差分验证 STM 的 ∂r(T)/∂r(0) 和 ∂r(T)/∂v(0)。
    #[test]
    fn stm_propagation_vs_finite_difference() {
        load_kernels();
        let config = earth_moon_sun_config();

        let et0 = 0.0;
        let r0 = [7000.0, 0.0, 0.0];
        let v0 = [0.0, 7.5, 0.0];
        let state0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let dt = 5400.0;
        let et1 = et0 + dt;
        let t_eval = vec![et0, et1];

        // 正常传播（带 STM）
        let result = propagate_with_stm(
            &config,
            (et0, et1),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        assert_eq!(result.states.len(), 2, "应输出 2 个时间点");
        let r1 = result.states[1];
        let stm_flat = &result.stms[1];

        // 有限差分验证
        let h = 1.0; // km（位置扰动）
        let h_vel = 0.001; // km/s（速度扰动）

        // 验证 ∂r(T)/∂r(0)
        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[dim] += h;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();
            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm_flat[row * 6 + dim];
                let fd_val = (r1_pert[row] - r1[row]) / h;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 5e-2,
                    "STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }

        // 验证 ∂r(T)/∂v(0)
        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[3 + dim] += h_vel;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();
            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm_flat[row * 6 + (dim + 3)];
                let fd_val = (r1_pert[row] - r1[row]) / h_vel;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 0.5,
                    "STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim + 3,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 6：propagate_with_stm 高层接口
    // =========================================================

    /// propagate_with_stm 端到端测试：传播 LEO 一个周期，验证 STM 与有限差分一致。
    #[test]
    fn propagate_with_stm_leo_one_period() {
        load_kernels();
        let config = earth_moon_sun_config();

        let et0 = 0.0;
        let r0 = [7000.0, 0.0, 0.0];
        let v0 = [0.0, 7.5, 0.0];
        let state0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];

        let dt = 5400.0; // 一个轨道周期
        let et1 = et0 + dt;
        let t_eval = vec![et0, et1];

        let result = propagate_with_stm(
            &config,
            (et0, et1),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.states.len(), 2, "应输出 2 个时间点");
        assert_eq!(result.stms.len(), 2);

        // 初始 STM 应为单位矩阵
        let stm0 = &result.stms[0];
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (stm0[i * 6 + j] - expected).abs() < 1e-14,
                    "初始 STM[{}][{}] = {} vs {}",
                    i,
                    j,
                    stm0[i * 6 + j],
                    expected
                );
            }
        }

        // STM 传播后应明显偏离单位矩阵
        let stm1 = &result.stms[1];
        let stm_deviation: f64 = stm1
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let i = k / 6;
                let j = k % 6;
                let expected = if i == j { 1.0 } else { 0.0 };
                (v - expected).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            stm_deviation > 0.01,
            "STM 传播后应明显偏离单位矩阵，偏差={}",
            stm_deviation
        );

        // 用 propagate_with_stm 做有限差分验证 STM
        let r1 = result.states[1];
        let h = 1.0; // km

        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[dim] += h;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();

            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm1[row * 6 + dim];
                let fd_val = (r1_pert[row] - r1[row]) / h;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 5e-2,
                    "propagate_with_stm STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 7：无星历天体必须返回 Err，不允许静默截断（issue #246）
    // =========================================================

    /// 第三体无星历数据时，propagate_with_stm 必须返回带上下文的 Err，
    /// 而不是静默返回只有 t0 的截断结果。
    #[test]
    fn propagate_with_stm_unknown_body_returns_err() {
        load_kernels();
        let config = NBodyConfig {
            bodies: vec!["EARTH".to_string(), "FAKEBODY".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436, 1.0],
        };

        let state0 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let t_eval = vec![0.0, 100.0];

        let result = propagate_with_stm(
            &config,
            (0.0, 100.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        );

        let err = match result {
            Ok(_) => panic!("无星历天体必须返回 Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("FAKEBODY") || err.contains("SPICE"),
            "错误信息应指名失败的天体或 SPICE 查询，实际: {}",
            err
        );
    }
}
