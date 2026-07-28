//! 编译型力模型 + STM 变分方程的 PD45 传播。
//!
//! 镜像 `nbody_stm.rs` 的 42 维增广状态布局，但用 PD45 循环
//! （与 `propagate_compiled` 同一 RK 表/控制器），使 with_stm=True/False
//! 的 states 逐位一致。
//!
//! 复用 `nbody_stm` 的 `compute_jacobian_A` + `stm_derivative`。

use crate::forces::compiled::{compute_total_acceleration_and_jacobian, CompiledForce};
use crate::forces::nbody_stm;
use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::pd45::PD45_TABLE;

/// PD45 嵌入误差估计的阶数（p=4 嵌入, 误差 ~ O(h^5)）。
const PD45_EMBEDDED_ORDER: usize = 4;

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
///
/// 从编译型力模型获取加速度 + 雅可比，组装 A 矩阵，
/// 计算 STM 变分方程。
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

    let (acc, jac_da_dr) = compute_total_acceleration_and_jacobian(forces, et, &state6, observer)?;
    let dstm = nbody_stm::stm_derivative(&stm, &jac_da_dr);

    let mut result = vec![0.0_f64; 42];
    // dr/dt = v
    result[0] = state6[3];
    result[1] = state6[4];
    result[2] = state6[5];
    // dv/dt = a
    result[3] = acc[0];
    result[4] = acc[1];
    result[5] = acc[2];
    // dΦ/dt
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
///
/// # 参数
/// - `forces`: 编译型力模型列表
/// - `observer`: 传播系 origin 天体名
/// - `t_span`: (t_start, t_end) 积分区间（SPICE et 秒）
/// - `t_eval`: 输出时间点（必须在 t_span 内且单调递增）
/// - `initial_state`: 初始状态 [x,y,z,vx,vy,vz]（km, km/s）
/// - `rtol`, `atol`: 积分容差（合并为 tol = rtol）
/// - `max_step`: 最大步长（秒），None 则不限制
/// - `max_steps`: 最大步数
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
) -> Result<CompiledStmResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }

    // 预检
    precheck(forces, observer, t_span.0, initial_state)?;

    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    // 构造 42 维增广状态
    let mut augmented0 = vec![0.0_f64; 42];
    augmented0[..6].copy_from_slice(initial_state);
    // 单位 STM
    for i in 0..6 {
        augmented0[6 + i * 6 + i] = 1.0;
    }

    let mut y = augmented0;
    let mut t = t_span.0;
    let mut h = (t_span.1 - t_span.0).min(h_max);
    let mut eval_idx = 1usize; // t_eval[0] == t0 已记录
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    let mut times = vec![t_span.0];
    let mut states = vec![[
        initial_state[0],
        initial_state[1],
        initial_state[2],
        initial_state[3],
        initial_state[4],
        initial_state[5],
    ]];
    let mut stms = {
        let mut stm0 = [0.0_f64; 36];
        for i in 0..6 {
            stm0[i * 6 + i] = 1.0;
        }
        vec![stm0]
    };

    while t < t_eval[t_eval.len() - 1] && n_steps < s_max {
        n_steps += 1;

        // 限制步长不超过下一个评估点
        if eval_idx < t_eval.len() {
            let t_next_eval = t_eval[eval_idx];
            if t + h > t_next_eval {
                h = t_next_eval - t;
            }
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

        let (y_new, error) = explicit_rk_step(&PD45_TABLE, t, &y, h, callback, Some(6))
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            // 输出落在 t_eval 的点
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

            h = suggest_next_step(h, error, tol, PD45_EMBEDDED_ORDER);
        } else {
            n_rejected += 1;
            h = suggest_next_step(h, error, tol, PD45_EMBEDDED_ORDER);
        }
    }

    // 校验输出长度
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
