//! 编译型力模型 + STM 变分方程的 PD45 传播。
//!
//! 镜像 `nbody_stm.rs` 的 42 维增广状态布局，但用 PD45 循环
//! （与 `propagate_compiled` 同一 RK 表/控制器），使 with_stm=True/False
//! 的 states 逐位一致。
//!
//! 复用 `nbody_stm` 的 `stm_derivative`。
//!
//! 参数敏感列（ASSIST 式变分方程对力模型参数的一阶偏导）：在 42 维之后
//! 每条参数追加 6 维 `S_p = ∂[r,v]/∂p`，满足 `Ṡ_p = A·S_p + [0; ∂a/∂p]`，
//! 初值 `S_p(0) = 0`。`sens` 为空时行为与旧 42 维路径逐位一致。

use super::compiled::{
    compute_total_acceleration_and_jacobian, next_force_discontinuity, param_accel_derivative,
    prev_force_discontinuity, CompiledForce, SensParam,
};
use super::nbody_stm;
use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// 最小步长（秒），防止步长坍缩。
const MIN_STEP: f64 = 1e-12;

/// 传播结果：状态轨迹 + STM 序列 + 参数敏感列序列。
pub struct CompiledStmResult {
    pub states: Vec<[f64; 6]>,
    pub stms: Vec<[f64; 36]>,
    /// 每个输出点的敏感列展平（`6*n_params`，按 `sens` 参数顺序排列）；
    /// 无敏感参数时为空。
    pub sensitivities: Vec<Vec<f64>>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 增广右端项：`[v(3), a(3), dΦ/dt(36), dS/dt(6*n_params)]`。
fn augmented_eom(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    augmented: &[f64],
    sens: &[(usize, SensParam)],
) -> Result<Vec<f64>, String> {
    let state6 = [
        augmented[0],
        augmented[1],
        augmented[2],
        augmented[3],
        augmented[4],
        augmented[5],
    ];
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&augmented[6..42]);

    let (acc, jac_da_dr, dadv) =
        compute_total_acceleration_and_jacobian(forces, et, &state6, observer)?;
    let dstm = nbody_stm::stm_derivative(&stm, &jac_da_dr, &dadv);

    let n_aug = 42 + 6 * sens.len();
    let mut result = vec![0.0_f64; n_aug];
    result[0] = state6[3];
    result[1] = state6[4];
    result[2] = state6[5];
    result[3] = acc[0];
    result[4] = acc[1];
    result[5] = acc[2];
    result[6..42].copy_from_slice(&dstm);

    // 每条敏感列：Ṡ_r = S_v，Ṡ_v = ∂a/∂r·S_r + ∂a/∂v·S_v + ∂a/∂p
    for (j, &(force_idx, param)) in sens.iter().enumerate() {
        let base = 42 + 6 * j;
        let s = &augmented[base..base + 6];
        let da_dp = param_accel_derivative(&forces[force_idx], param, et, &state6, observer)?;
        for i in 0..3 {
            let mut dsv = da_dp[i];
            for k in 0..3 {
                dsv += jac_da_dr[i][k] * s[k] + dadv[i][k] * s[3 + k];
            }
            result[base + i] = s[3 + i];
            result[base + 3 + i] = dsv;
        }
    }
    Ok(result)
}

/// 预检：在 t0 做一次加速度 + 雅可比计算，SPICE 配置错误立即显式失败。
pub fn precheck(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &[f64; 6],
) -> Result<(), String> {
    compute_total_acceleration_and_jacobian(forces, et, state, observer)?;
    Ok(())
}

/// PD45 42 维增广状态传播（状态 + STM）。
///
/// 与 `nbody_stm::propagate_with_stm` 物理等价，但用 PD45 循环
/// （与 `propagate_compiled` 同一 RK 表/控制器），
/// 保证 with_stm=True/False 的 states 逐位一致。
///
/// `t_span` 方向即传播方向；反向（`t_span.1 < t_span.0`）时 `t_eval` 须
/// 同向单调递减。
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_stm(
    forces: &[CompiledForce],
    observer: &str,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    method: RkMethod,
) -> Result<CompiledStmResult, String> {
    propagate_compiled_stm_sens(
        forces,
        observer,
        t_span,
        t_eval,
        initial_state,
        rtol,
        atol,
        max_step,
        max_steps,
        method,
        &[],
    )
}

/// PD45 增广状态传播（状态 + STM + 参数敏感列）。
///
/// `sens` 的每项是 ``(force_index, SensParam)``：`force_index` 是
/// `forces` 切片中的下标，`SensParam` 指定对哪个参数求偏导。每条参数在
/// 增广状态尾部追加 6 维敏感列（初值零），维度 `42 + 6·sens.len()`。
/// `sens` 为空时与 [`propagate_compiled_stm`] 逐位一致。
///
/// `t_span` 方向即传播方向；反向（`t_span.1 < t_span.0`）时 `t_eval` 须
/// 同向单调递减。
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_stm_sens(
    forces: &[CompiledForce],
    observer: &str,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    method: RkMethod,
    sens: &[(usize, SensParam)],
) -> Result<CompiledStmResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }
    for &(force_idx, _) in sens {
        if force_idx >= forces.len() {
            return Err(format!(
                "sens force index {force_idx} out of range ({} forces)",
                forces.len()
            ));
        }
    }

    precheck(forces, observer, t_span.0, initial_state)?;

    // 传播方向：t_span.1 < t_span.0 时反向积分（h 取负、t_eval 单调递减）。
    let dir = (t_span.1 - t_span.0).signum();
    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let n_aug = 42 + 6 * sens.len();
    let mut augmented0 = vec![0.0_f64; n_aug];
    augmented0[..6].copy_from_slice(initial_state);
    for i in 0..6 {
        augmented0[6 + i * 6 + i] = 1.0;
    }
    // 敏感列初值为零（augmented0 已置零）

    let mut y = augmented0;
    let mut t = t_span.0;
    let mut h = dir * (t_span.1 - t_span.0).abs().min(h_max);
    // 输出起点跟随 t_eval：当 t_eval[0]==t_span.0 时记录初始状态/STM、eval_idx
    // 从 1 起步；否则（如逐段积分 patch point 时刻非整数小时、t_eval 整数小时
    // 点严格大于 t0）不预设 t_span.0 到输出、eval_idx 从 0 起步由循环匹配。
    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut stms: Vec<[f64; 36]> = Vec::with_capacity(t_eval.len());
    let mut sensitivities: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    if (t_span.0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t_span.0);
        states.push([
            initial_state[0],
            initial_state[1],
            initial_state[2],
            initial_state[3],
            initial_state[4],
            initial_state[5],
        ]);
        let mut stm0 = [0.0_f64; 36];
        for i in 0..6 {
            stm0[i * 6 + i] = 1.0;
        }
        stms.push(stm0);
        if !sens.is_empty() {
            sensitivities.push(vec![0.0_f64; 6 * sens.len()]);
        }
        eval_idx = 1;
    }

    while dir * (t_eval[t_eval.len() - 1] - t) > 0.0 && n_steps < s_max {
        n_steps += 1;

        let mut t_next = if eval_idx < t_eval.len() {
            t_eval[eval_idx]
        } else {
            t_span.1
        };
        // 推力开关边界作为步长终点：正向取 (t, t_span.1] 内最早者，
        // 反向取 [t_span.1, t) 内最晚者，保证 RK 步不跨越不连续面。
        if dir > 0.0 {
            if let Some(boundary) = next_force_discontinuity(forces, t, t_span.1) {
                t_next = t_next.min(boundary);
            }
        } else if let Some(boundary) = prev_force_discontinuity(forces, t, t_span.1) {
            t_next = t_next.max(boundary);
        }
        if dir * (t + h - t_next) > 0.0 {
            h = t_next - t;
        }
        if h.abs() > h_max {
            h = dir * h_max;
        }
        if h.abs() < MIN_STEP * (t_span.1 - t_span.0).abs() {
            return Err(format!(
                "step size collapsed below minimum after {} steps",
                n_steps
            ));
        }

        let forces_ref = forces;
        let observer_ref = observer;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            augmented_eom(forces_ref, observer_ref, ti, yi, sens)
        };

        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, Some(6))
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            while eval_idx < t_eval.len() && dir * (t - t_eval[eval_idx]) >= -1e-9 {
                times.push(t_eval[eval_idx]);
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(&y[..6]);
                states.push(s);
                let mut stm = [0.0_f64; 36];
                stm.copy_from_slice(&y[6..42]);
                stms.push(stm);
                if !sens.is_empty() {
                    sensitivities.push(y[42..].to_vec());
                }
                eval_idx += 1;
            }

            h = suggest_next_step(h, error, tol, method.embedded_order());
        } else {
            n_rejected += 1;
            h = suggest_next_step(h, error, tol, method.embedded_order());
        }
    }

    if times.len() != t_eval.len() {
        return Err(format!(
            "output length mismatch: got {} time points, expected {}",
            times.len(),
            t_eval.len()
        ));
    }

    Ok(CompiledStmResult {
        states,
        stms,
        sensitivities,
        times,
        n_steps,
        n_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GEO 量级圆轨道初值（km, km/s），PointMass 力下近圆。
    fn geo_y0() -> [f64; 6] {
        [42164.0, 0.0, 0.0, 0.0, 3.0747, 0.0]
    }

    /// 反向传播回归（#640）：以正向末态为初值反向积分应还原初值，
    /// 且 STM_bwd ⊠ STM_fwd ≈ I₆（反向变分方程正确性不变量）。
    #[test]
    fn backward_propagation_roundtrip_point_mass() {
        let forces = vec![CompiledForce::PointMass { mu: 398600.4418 }];
        let y0 = geo_y0();
        let observer = "EARTH";

        // 正向 0 -> 3600 s，三点采样。
        let fwd = propagate_compiled_stm(
            &forces,
            observer,
            (0.0, 3600.0),
            &[0.0, 1800.0, 3600.0],
            &y0,
            1e-12,
            1e-14,
            None,
            None,
            RkMethod::Pd78,
        )
        .unwrap();
        let state1 = fwd.states[2];
        let stm_fwd = fwd.stms[2];

        // 反向 3600 -> 0：t_eval 单调递减。修复前此处报
        // "output length mismatch: got 1 time points, expected 2"。
        let bwd = propagate_compiled_stm(
            &forces,
            observer,
            (3600.0, 0.0),
            &[3600.0, 0.0],
            &state1,
            1e-12,
            1e-14,
            None,
            None,
            RkMethod::Pd78,
        )
        .unwrap();

        assert_eq!(
            bwd.times.len(),
            2,
            "反向传播输出长度不符（#640 回归）：got {}",
            bwd.times.len()
        );
        assert!((bwd.times[0] - 3600.0).abs() < 1e-9);
        assert!(bwd.times[1].abs() < 1e-9);
        for (i, (got, &want)) in bwd.states[1].iter().zip(y0.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "backward roundtrip state[{i}] = {got} vs {want}"
            );
        }
        // 反向 STM ⊠ 正向 STM ≈ I₆。
        let stm_bwd = bwd.stms[1];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = 0.0_f64;
                for k in 0..6 {
                    acc += stm_bwd[i * 6 + k] * stm_fwd[k * 6 + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - expect).abs() < 1e-6,
                    "stm_bwd*stm_fwd[{i}][{j}] = {acc} vs {expect}"
                );
            }
        }
    }
}
