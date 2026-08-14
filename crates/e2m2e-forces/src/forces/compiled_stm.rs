//! 编译型力模型 + STM 变分方程的 PD45 传播。
//!
//! 镜像 `nbody_stm.rs` 的 42 维增广状态布局，但用 PD45 循环
//! （与 `propagate_compiled` 同一 RK 表/控制器），使 with_stm=True/False
//! 的 states 逐位一致。
//!
//! 复用 `nbody_stm` 的 `stm_derivative`。

use super::compiled::{
    compute_total_acceleration_and_jacobian, next_force_discontinuity, CompiledForce,
};
use super::nbody_stm;
use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// 最小步长（秒），防止步长坍缩。
const MIN_STEP: f64 = 1e-12;

/// 传播结果：状态轨迹 + STM 序列。
pub struct CompiledStmResult {
    pub states: Vec<[f64; 6]>,
    pub stms: Vec<[f64; 36]>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 42 维增广右端项：[v(3), a(3), dΦ/dt(36)]。
fn augmented_eom(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    augmented: &[f64],
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

    let mut result = vec![0.0_f64; 42];
    result[0] = state6[3];
    result[1] = state6[4];
    result[2] = state6[5];
    result[3] = acc[0];
    result[4] = acc[1];
    result[5] = acc[2];
    result[6..42].copy_from_slice(&dstm);
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
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_stm(
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
) -> Result<CompiledStmResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }

    precheck(forces, observer, t_span.0, initial_state)?;

    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut augmented0 = vec![0.0_f64; 42];
    augmented0[..6].copy_from_slice(initial_state);
    for i in 0..6 {
        augmented0[6 + i * 6 + i] = 1.0;
    }

    let mut y = augmented0;
    let mut t = t_span.0;
    let mut h = (t_span.1 - t_span.0).min(h_max);
    // 输出起点跟随 t_eval：当 t_eval[0]==t_span.0 时记录初始状态/STM、eval_idx
    // 从 1 起步；否则（如逐段积分 patch point 时刻非整数小时、t_eval 整数小时
    // 点严格大于 t0）不预设 t_span.0 到输出、eval_idx 从 0 起步由循环匹配。
    // 此前硬编码 vec![t_span.0] + eval_idx=1 假设 t_eval[0]==t_span.0，导致
    // t_eval[0]>t_span.0 时首个输出点状态/STM 错置为初值、与后续点错位
    // （与 propagate_compiled 同源 bug）。
    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut stms: Vec<[f64; 36]> = Vec::with_capacity(t_eval.len());
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
        eval_idx = 1;
    }

    while t < t_eval[t_eval.len() - 1] && n_steps < s_max {
        n_steps += 1;

        let mut t_next = if eval_idx < t_eval.len() {
            t_eval[eval_idx]
        } else {
            t_span.1
        };
        if let Some(boundary) = next_force_discontinuity(forces, t, t_span.1) {
            t_next = t_next.min(boundary);
        }
        if t + h > t_next {
            h = t_next - t;
        }
        h = h.min(h_max);
        if h < MIN_STEP * (t_span.1 - t_span.0).abs() {
            return Err(format!(
                "step size collapsed below minimum after {} steps",
                n_steps
            ));
        }

        let forces_ref = forces;
        let observer_ref = observer;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            augmented_eom(forces_ref, observer_ref, ti, yi)
        };

        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, Some(6))
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            while eval_idx < t_eval.len() && t >= t_eval[eval_idx] - 1e-9 {
                times.push(t_eval[eval_idx]);
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(&y[..6]);
                states.push(s);
                let mut stm = [0.0_f64; 36];
                stm.copy_from_slice(&y[6..42]);
                stms.push(stm);
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
        times,
        n_steps,
        n_rejected,
    })
}
