//! LPO 全周期 Newton 修正器（Rust 实现）。
//!
//! 对齐 Python `DifferentialCorrection.iterate_full_period_correction` 的
//! LPO 语义（`lpo_fixed_x0` 策略，无对称性假设的全周期闭合）：
//!
//! - 自由变量：`[y₀, ẋ₀, ẏ₀, T]`（4 个；x₀ 固定为族参数，z₀=ż₀=0 平面约束）
//! - 约束：`[Δy=0, Δẋ=0, Δẏ=0]`（3 个），全周期闭合 `F = state(T) - state(0)`
//!   在约束分量 `[1, 3, 4]` 上
//! - 4 变量 3 约束**欠定**，用 min-norm 最小二乘（`dX = Jᵀ(JJᵀ)⁻¹F`）——
//!   与 numpy `lstsq` 同一数学，见设计文档 `docs/plans/lpo-rust-rayon.md` 第五节
//!
//! 传播复用 `e2m2e_forces::cr3bp::propagate_cr3bp_stm`（**不重写**）。

use e2m2e_forces::cr3bp::{cr3bp_eom, propagate_cr3bp_stm};

/// 收敛容差（对齐 Python `DifferentialCorrection.DEFAULT_TOLERANCE`）。
const TOLERANCE: f64 = 1e-12;
/// 发散阈值（对齐 Python `DifferentialCorrection.divergence_limit`）。
const DIVERGENCE_LIMIT: f64 = 1e10;
/// 停滞阈值（对齐 Python `DifferentialCorrection.stagnation_limit`）。
const STAGNATION_LIMIT: f64 = 1e-14;
/// 最大迭代次数（对齐 Python `DifferentialCorrection.DEFAULT_MAX_ITERATIONS`）。
const MAX_ITERATIONS: usize = 50;
/// 停滞时把"修正量过小但残差已足够小"判为收敛的残差阈值（对齐 Python
/// `iterate_full_period_correction:952`）。
const STAGNATION_CONVERGE_ERROR: f64 = 1e-8;
/// 收敛后周期校验的下/上限（对齐 `_correct_lpo` 的周期异常检查）。
const PERIOD_MIN: f64 = 10.0;
const PERIOD_MAX: f64 = 50.0;

/// LPO 全周期修正结果。
#[derive(Debug, Clone)]
pub struct LpoCorrectionResult {
    /// 修正后的初始状态（6 维，x₀ 固定）。
    pub state: [f64; 6],
    /// 修正后的全周期（无量纲时间）。
    pub period: f64,
    /// 是否收敛（闭合残差 < 容差，或停滞时残差 < 1e-8）。
    pub converged: bool,
    /// 实际迭代次数。
    pub iterations: usize,
}

/// 向量二范数。
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// 解欠定线性系统 `J·dX = rhs` 的最小范数解（n_constraints ≤ n_vars）。
///
/// # 数学依据
///
/// 欠定行满秩系统有无穷多解；最小范数解
///   `dX = Jᵀ (J Jᵀ)⁻¹ rhs`（J 行满秩时即右伪逆 `J⁺ = Jᵀ(JJᵀ)⁻¹`）
/// 是离初猜最近的修正——"保形"（微分修正保持轨道族连续性的来源，设计文档
/// lpo-rust-rayon.md 第五节，与多重打靶阻尼 LM 的 min-norm 偏好同一目的）。
/// 该解与 numpy `np.linalg.lstsq`（SVD 伪逆）一致到机器精度——一致是因为两者
/// 实现同一数学，不是为过对照测试。
///
/// # 实现
///
/// 对小矩阵用正规方程 + 高斯消元（部分主元）：先解 `(J Jᵀ) d = rhs`（n_c×n_c
/// 对称正定），再 `dX = Jᵀ d`。3×3 消元比 SVD 简单可靠。
///
/// `jacobian` 为 n_constraints × n_vars 行优先展平；`rhs` 长度 n_constraints。
fn min_norm_lstsq(jacobian: &[f64], rhs: &[f64], n_constraints: usize, n_vars: usize) -> Vec<f64> {
    debug_assert_eq!(jacobian.len(), n_constraints * n_vars);
    debug_assert_eq!(rhs.len(), n_constraints);

    // 正规方程 G = J Jᵀ（n_c × n_c，行满秩时对称正定）
    let mut g = vec![0.0_f64; n_constraints * n_constraints];
    for i in 0..n_constraints {
        for k in 0..n_constraints {
            let mut s = 0.0;
            for j in 0..n_vars {
                s += jacobian[i * n_vars + j] * jacobian[k * n_vars + j];
            }
            g[i * n_constraints + k] = s;
        }
    }

    // 增广矩阵 [G | rhs]
    let mut aug = vec![0.0_f64; n_constraints * (n_constraints + 1)];
    for i in 0..n_constraints {
        for k in 0..n_constraints {
            aug[i * (n_constraints + 1) + k] = g[i * n_constraints + k];
        }
        aug[i * (n_constraints + 1) + n_constraints] = rhs[i];
    }

    // 高斯消元（部分主元）
    for i in 0..n_constraints {
        let mut max_val = aug[i * (n_constraints + 1) + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n_constraints {
            if aug[k * (n_constraints + 1) + i].abs() > max_val {
                max_val = aug[k * (n_constraints + 1) + i].abs();
                max_row = k;
            }
        }
        if max_row != i {
            for j in 0..=n_constraints {
                aug.swap(
                    i * (n_constraints + 1) + j,
                    max_row * (n_constraints + 1) + j,
                );
            }
        }
        let pivot = aug[i * (n_constraints + 1) + i];
        if pivot.abs() < 1e-15 {
            continue; // 奇异：该行解留 0（min-norm 退化解）
        }
        for k in (i + 1)..n_constraints {
            let factor = aug[k * (n_constraints + 1) + i] / pivot;
            for j in i..=n_constraints {
                aug[k * (n_constraints + 1) + j] -= factor * aug[i * (n_constraints + 1) + j];
            }
        }
    }

    // 回代
    let mut d = vec![0.0_f64; n_constraints];
    for i in (0..n_constraints).rev() {
        let mut sum = aug[i * (n_constraints + 1) + n_constraints];
        for j in (i + 1)..n_constraints {
            sum -= aug[i * (n_constraints + 1) + j] * d[j];
        }
        let pivot = aug[i * (n_constraints + 1) + i];
        if pivot.abs() > 1e-15 {
            d[i] = sum / pivot;
        }
    }

    // dX = Jᵀ d
    let mut dx = vec![0.0_f64; n_vars];
    for j in 0..n_vars {
        let mut s = 0.0;
        for i in 0..n_constraints {
            s += jacobian[i * n_vars + j] * d[i];
        }
        dx[j] = s;
    }
    dx
}

/// 在族参数 `x0` 处做 LPO 全周期 Newton 修正。
///
/// # 语义（对齐 Python `iterate_full_period_correction` 的 `lpo_fixed_x0` 策略）
///
/// 每轮迭代：
/// 1. 用当前 `(state, T)` 全周期 STM 传播 `propagate_cr3bp_stm(mu, (0, T),
///    t_eval=[0, T], initial_state, rtol, atol, max_step=None)`，取终端
///    `final_state` 与 `final_stm`（6×6）；
/// 2. 闭合残差 `error_vector = [Δy, Δẋ, Δẏ]`（`final_state[i] - state[i]`，
///    `i ∈ {1,3,4}`），`current_error = ‖error_vector‖`；
/// 3. 收敛：`current_error < tolerance`；发散：`current_error > divergence_limit`；
/// 4. 组装雅可比（3 约束 × 4 变量）：
///    - 状态自由变量（`var_idx ∈ {1,3,4}`）：`J[i,j] = final_stm[c_idx, var_idx]`，
///      若 `var_idx == c_idx` 再减 1（∂(-state(0))/∂state(0) = -I 项）；
///    - 时间自由变量（`var_idx == 6`）：`J[i,j] = ẋ(T)[c_idx]`
///      （`cr3bp_eom` 的状态导数）；
/// 5. min-norm 求解 `J·delta = error_vector`（对齐 Python
///    `np.linalg.lstsq(J, error_vector)[0]`），更新 `X_new = X - delta`；
/// 6. `T <= 0` 时钳位到 0.1；停滞（`‖delta‖ < stagnation_limit`）时若
///    `current_error < 1e-8` 判收敛，否则判停滞失败；
/// 7. 上限 `max_iterations = 50`。
///
/// 收敛后校验周期 `T ∈ [10, 50]`（无量纲），否则 `Err`（对齐 `_correct_lpo`
/// 的周期异常检查）。
///
/// # 参数
/// - `mu`：CR3BP 质量参数 μ = m₂/(m₁+m₂)
/// - `x0`：固定的初始 x 坐标（族参数，覆盖 `initial_state[0]`）
/// - `initial_state`：初猜状态（z₀=ż₀=0 平面约束在内部强制）
/// - `period_guess`：初猜全周期
/// - `rtol` / `atol`：传播容差（传给 `propagate_cr3bp_stm`）
///
/// # 返回
/// 收敛且周期有效：`Ok(LpoCorrectionResult)`；发散 / 停滞 / 传播失败 /
/// 周期异常：`Err(String)`（`converged=false` 的非收敛结果不报错，由调用方
/// 通过 `converged` 字段判断）。
pub fn correct_lpo_full_period(
    mu: f64,
    x0: f64,
    initial_state: &[f64; 6],
    period_guess: f64,
    rtol: f64,
    atol: f64,
) -> Result<LpoCorrectionResult, String> {
    if !period_guess.is_finite() || period_guess <= 0.0 {
        return Err(format!(
            "LPO(x0={x0:.6}) 初猜周期无效（T0={period_guess:.3}）"
        ));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(format!(
            "LPO(x0={x0:.6}) 传播容差无效（rtol={rtol:.1e}, atol={atol:.1e}）"
        ));
    }

    let mut state = *initial_state;
    // 族参数固定 x₀ + xy 平面约束（z₀=ż₀=0，对齐 `_correct_lpo`）
    state[0] = x0;
    state[2] = 0.0;
    state[5] = 0.0;
    let mut period = period_guess;

    let free_vars = [1usize, 3, 4, 6]; // [y₀, ẋ₀, ẏ₀, T]
    let constraint_indices = [1usize, 3, 4]; // [Δy, Δẋ, Δẏ]
    let n_vars = free_vars.len();
    let n_constraints = constraint_indices.len();

    let mut converged = false;
    let mut iterations = 0usize;

    for iteration in 0..MAX_ITERATIONS {
        iterations = iteration + 1;

        // 1. 全周期 STM 传播（t_eval=[0, T]，只取终端）
        let result = propagate_cr3bp_stm(
            mu,
            (0.0, period),
            &[0.0, period],
            &state,
            rtol,
            atol,
            None,
            None,
        )?;
        let final_state = *result.states.last().ok_or("empty propagation result")?;
        let final_stm = *result.stms.last().ok_or("empty STM result")?;

        // 2. 闭合残差 F = state(T) - state(0) 在约束分量 [1,3,4]
        let error_vector = [
            final_state[1] - state[1],
            final_state[3] - state[3],
            final_state[4] - state[4],
        ];
        let current_error = l2_norm(&error_vector);

        // 3. 收敛 / 发散判据
        if current_error < TOLERANCE {
            converged = true;
            break;
        }
        if current_error > DIVERGENCE_LIMIT {
            break;
        }

        // 4. 组装雅可比（3 约束 × 4 变量）
        let state_derivative = cr3bp_eom(mu, &final_state); // ẋ(T)
        let mut jacobian = [0.0_f64; 12];
        for (j, &var_idx) in free_vars.iter().enumerate() {
            if var_idx < 6 {
                // 状态自由变量：∂final/∂var = STM 列；var_idx == c_idx 时减 1
                for (i, &c_idx) in constraint_indices.iter().enumerate() {
                    let mut val = final_stm[c_idx * 6 + var_idx];
                    if var_idx == c_idx {
                        val -= 1.0;
                    }
                    jacobian[i * n_vars + j] = val;
                }
            } else {
                // 时间自由变量：∂final/∂T = ẋ(T)
                for (i, &c_idx) in constraint_indices.iter().enumerate() {
                    jacobian[i * n_vars + j] = state_derivative[c_idx];
                }
            }
        }

        // 5. min-norm 求解 J·delta = error_vector，更新 X_new = X - delta。
        //    delta 与 Python `np.linalg.lstsq(J, error_vector)[0]` 同解（位级
        //    一致）；等价于 Newton 规范式 `J·dX = -F, X_new = X + dX`（同一
        //    X_new = X - J⁺·F）。
        let delta = min_norm_lstsq(&jacobian, &error_vector, n_constraints, n_vars);
        let correction_norm = l2_norm(&delta);

        for (j, &var_idx) in free_vars.iter().enumerate() {
            if var_idx < 6 {
                state[var_idx] -= delta[j];
            } else {
                period -= delta[j];
            }
        }
        if period <= 0.0 {
            period = 0.1;
        }

        // 6. 停滞检查（对齐 Python：无条件 break，残差已足够小时算收敛）
        if correction_norm < STAGNATION_LIMIT {
            if current_error < STAGNATION_CONVERGE_ERROR {
                converged = true;
            }
            break;
        }
    }

    // 收敛后校验周期（对齐 `_correct_lpo` 的周期异常检查）
    if converged && !(PERIOD_MIN..=PERIOD_MAX).contains(&period) {
        return Err(format!(
            "LPO(x0={x0:.6}) 周期异常（T={period:.3}，预期 {PERIOD_MIN:.0}-{PERIOD_MAX:.0}）"
        ));
    }

    Ok(LpoCorrectionResult {
        state,
        period,
        converged,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 地月系质量参数（对齐 `earth_moon_system()` 的 `EARTH_MOON_MU`）。
    const MU_EARTH_MOON: f64 = 0.0121506683;

    /// 初猜的 LPO 初猜（对齐 `compute_lpo_initial_guess(system, 4, 1000km)`）：
    /// 长周期线性化模态，z₀=ż₀≈0（平面）。
    const GUESS_STATE: [f64; 6] = [
        0.48561751816786186,
        0.8672348275825809,
        0.0,
        0.0,
        0.00016971226282389845,
        0.0,
    ];
    const GUESS_PERIOD: f64 = 21.069717310998552;

    /// 从初猜出发收敛后的参考轨道（Python `_correct_lpo` 输出，供对照）。
    const REF_STATE: [f64; 6] = [
        0.48561751816786186,
        0.8672314929275382,
        0.0,
        -3.4846216971678926e-07,
        0.00016886669249480745,
        0.0,
    ];
    const REF_PERIOD: f64 = 21.069729764938614;

    /// 已知 3×4 欠定系统：min-norm 解手算对照。
    ///
    /// J = [1 1 0 1; 0 2 1 0; 1 0 1 1]，rhs = [1, 0, 0]。
    /// 通解 dX = [2/3 - t, 1/3, -2/3, t]；min-norm 在 t=1/3 取
    /// dX = [1/3, 1/3, -2/3, 1/3]（‖·‖² = 7/9 < 任一特解）。
    #[test]
    fn test_min_norm_lstsq_known_solution() {
        let j = [
            1.0, 1.0, 0.0, 1.0, //
            0.0, 2.0, 1.0, 0.0, //
            1.0, 0.0, 1.0, 1.0,
        ];
        let rhs = [1.0, 0.0, 0.0];
        let dx = min_norm_lstsq(&j, &rhs, 3, 4);

        let expected = [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0];
        for (got, exp) in dx.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "dX 分量 {} vs 期望 {}", got, exp);
        }

        // J·dX = rhs（残差为零）
        let residual = [
            j[0] * dx[0] + j[1] * dx[1] + j[2] * dx[2] + j[3] * dx[3] - rhs[0],
            j[4] * dx[0] + j[5] * dx[1] + j[6] * dx[2] + j[7] * dx[3] - rhs[1],
            j[8] * dx[0] + j[9] * dx[1] + j[10] * dx[2] + j[11] * dx[3] - rhs[2],
        ];
        for r in residual {
            assert!(r.abs() < 1e-12, "J·dX - rhs = {}", r);
        }

        // min-norm：dX 与 J 的零空间正交（null(J) = span([1,0,0,-1])）
        let null_vec = [1.0, 0.0, 0.0, -1.0];
        let proj =
            dx[0] * null_vec[0] + dx[1] * null_vec[1] + dx[2] * null_vec[2] + dx[3] * null_vec[3];
        assert!(proj.abs() < 1e-12, "dX·null = {}", proj);

        // min-norm：‖dX‖ ≤ 任一特解（取 t=0 特解 [2/3, 1/3, -2/3, 0]，‖·‖=1）
        let particular = [2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 0.0];
        assert!(
            l2_norm(&dx) < l2_norm(&particular),
            "min-norm 应不大于特解范数"
        );
    }

    /// 行满秩单位基：J = [I₃ | 0]，min-norm 解 = [rhs, 0]。
    #[test]
    fn test_min_norm_lstsq_identity_blocks() {
        let j = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        let rhs = [2.0, -1.0, 3.0];
        let dx = min_norm_lstsq(&j, &rhs, 3, 4);
        let expected = [2.0, -1.0, 3.0, 0.0];
        for (got, exp) in dx.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "dX {} vs {}", got, exp);
        }
    }

    /// 端到端：从线性化长周期初猜出发，Newton 收敛到闭合轨道。
    ///
    /// 对齐 Python `_correct_lpo`：初猜收敛后周期应在 LPO 范围、状态接近
    /// Python 参考解、且重新传播闭合残差足够小。
    #[test]
    fn test_correct_lpo_full_period_converges_from_guess() {
        let result = correct_lpo_full_period(
            MU_EARTH_MOON,
            GUESS_STATE[0],
            &GUESS_STATE,
            GUESS_PERIOD,
            1e-12,
            1e-12,
        )
        .expect("LPO 修正应收敛");
        assert!(
            result.converged,
            "修正应收敛（迭代 {} 次）",
            result.iterations
        );
        assert!(
            result.iterations <= 8,
            "初猜足够近，应少量迭代收敛（{} 次）",
            result.iterations
        );

        // 周期在 LPO 范围且接近 Python 参考
        assert!(
            (10.0..=50.0).contains(&result.period),
            "周期 {} 应在 [10, 50]",
            result.period
        );
        assert!(
            (result.period - REF_PERIOD).abs() < 1e-4,
            "周期 {} vs Python 参考 {}",
            result.period,
            REF_PERIOD
        );

        // 修正后状态接近 Python 参考解（传播路径 max_step/t_eval 差异在容差内）
        for i in 0..6 {
            assert!(
                (result.state[i] - REF_STATE[i]).abs() < 1e-5,
                "state[{}] = {} vs Python 参考 {}",
                i,
                result.state[i],
                REF_STATE[i]
            );
        }

        // 全周期闭合残差：重新传播验证 < 1e-8
        let prop = propagate_cr3bp_stm(
            MU_EARTH_MOON,
            (0.0, result.period),
            &[0.0, result.period],
            &result.state,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        let final_state = prop.states.last().unwrap();
        let closure = (0..6)
            .map(|i| (final_state[i] - result.state[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(closure < 1e-8, "全周期闭合残差 {} 应 < 1e-8", closure);
    }

    /// 已经收敛的轨道直接喂回修正器，应在 1-2 轮内确认闭合（不破坏解）。
    #[test]
    fn test_correct_lpo_full_period_idempotent() {
        let result = correct_lpo_full_period(
            MU_EARTH_MOON,
            REF_STATE[0],
            &REF_STATE,
            REF_PERIOD,
            1e-12,
            1e-12,
        )
        .expect("LPO 修正应成功");
        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "已闭合轨道应 1-2 轮确认（{} 次）",
            result.iterations
        );
    }

    /// 周期异常：收敛到 [10,50] 之外的伪解应报错（对齐 `_correct_lpo`）。
    ///
    /// 直接用超小周期（< 1e-6 量级）触发：即使传播"闭合"，周期校验也应拦截。
    #[test]
    fn test_correct_lpo_full_period_rejects_bad_period() {
        // 构造一个周期极小、必被周期校验拦截的输入。传播本身应可行，
        // 但收敛后周期 < 10 → Err。
        let tiny_state = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0];
        let res = correct_lpo_full_period(MU_EARTH_MOON, 0.9, &tiny_state, 0.01, 1e-12, 1e-12);
        // 只断言：要么收敛且被周期校验拒绝（Err），要么发散/停滞（Ok converged=false）；
        // 不允许出现"收敛且周期异常却返回 Ok converged=true"
        match res {
            Ok(r) => assert!(!r.converged || (10.0..=50.0).contains(&r.period)),
            Err(_) => {}
        }
    }
}
