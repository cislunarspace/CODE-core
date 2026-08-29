//! 编译型力模型的 IAS15 传播（状态 + 可选 STM + 可选参数敏感列）。
//!
//! 驱动 [`e2m2e_propagation::ias15::propagate_ias15`]：状态/STM/敏感列统一
//! 走增广状态 `[r(3), v(3), Φ(36, 可选), S(6·n_params, 可选)]`，其中 Φ 与
//! S 作为一阶额外分量由 IAS15 单积分处理，位置/速度走二阶双重积分。
//!
//! 与 RK 路径（`compiled_stm.rs`）的差异：IAS15 容差是相对加速度采样量级
//! 的（语义对齐 IAS15 论文），且步长截断到输出点/低推力开关机边界，
//! 不做稠密输出插值。

use super::compiled::{
    compute_total_acceleration, compute_total_acceleration_and_jacobian, next_force_discontinuity,
    param_accel_derivative, CompiledForce, SensParam,
};
use super::nbody_stm;
use e2m2e_propagation::ias15::propagate_ias15;

/// IAS15 传播结果。
pub struct CompiledIas15Result {
    pub states: Vec<[f64; 6]>,
    /// `with_stm=true` 时每个输出点的 STM（展平 36），否则为空。
    pub stms: Vec<[f64; 36]>,
    /// `sens` 非空时每个输出点的敏感列展平（`6·n_params`，按 `sens` 顺序）。
    pub sensitivities: Vec<Vec<f64>>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 编译型力模型的 IAS15 传播。
///
/// - `tol`：相对容差（对加速度采样量级归一），与 RK 路径的 rtol/atol
///   语义不同，见模块文档。
/// - `with_stm`：增广 36 维 STM（初值单位阵）。
/// - `sens`：每项 ``(force_index, SensParam)``，追加 6 维敏感列（初值零）。
///   需要雅可比时自动走 `acceleration_and_jacobian`（同 STM 路径的支持面：
///   不支持的力类型在首次求值时显式报错）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_ias15(
    forces: &[CompiledForce],
    observer: &str,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    tol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    with_stm: bool,
    sens: &[(usize, SensParam)],
) -> Result<CompiledIas15Result, String> {
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

    let n_params = sens.len();
    let n = 6 + if with_stm { 36 } else { 0 } + 6 * n_params;

    let mut y0 = vec![0.0_f64; n];
    y0[..6].copy_from_slice(initial_state);
    if with_stm {
        for i in 0..6 {
            y0[6 + i * 6 + i] = 1.0;
        }
    }

    let need_jac = with_stm || n_params > 0;
    let rhs = |et: f64, y: &[f64]| -> Result<Vec<f64>, String> {
        let state6 = [y[0], y[1], y[2], y[3], y[4], y[5]];
        let mut dy = vec![0.0_f64; n];
        dy[0] = state6[3];
        dy[1] = state6[4];
        dy[2] = state6[5];
        let mut idx = 6;
        if need_jac {
            let (acc, jac_da_dr, dadv) =
                compute_total_acceleration_and_jacobian(forces, et, &state6, observer)?;
            dy[3] = acc[0];
            dy[4] = acc[1];
            dy[5] = acc[2];
            if with_stm {
                let mut stm = [0.0_f64; 36];
                stm.copy_from_slice(&y[6..42]);
                let dstm = nbody_stm::stm_derivative(&stm, &jac_da_dr, &dadv);
                dy[6..42].copy_from_slice(&dstm);
                idx = 42;
            }
            for (j, &(force_idx, param)) in sens.iter().enumerate() {
                let base = idx + 6 * j;
                let s = &y[base..base + 6];
                let da_dp =
                    param_accel_derivative(&forces[force_idx], param, et, &state6, observer)?;
                for i in 0..3 {
                    let mut dsv = da_dp[i];
                    for k in 0..3 {
                        dsv += jac_da_dr[i][k] * s[k] + dadv[i][k] * s[3 + k];
                    }
                    dy[base + i] = s[3 + i];
                    dy[base + 3 + i] = dsv;
                }
            }
        } else {
            let acc = compute_total_acceleration(forces, et, &state6, observer)?;
            dy[3] = acc[0];
            dy[4] = acc[1];
            dy[5] = acc[2];
        }
        Ok(dy)
    };

    // 低推力开关机边界：步长必须落在边界上（同 compiled_stm 的截断逻辑）。
    let mut breaks: Vec<f64> = Vec::new();
    {
        let mut t = t_span.0;
        while let Some(b) = next_force_discontinuity(forces, t, t_span.1) {
            breaks.push(b);
            t = b;
        }
    }

    let result = propagate_ias15(rhs, t_span, t_eval, &y0, tol, max_step, max_steps, &breaks)
        .map_err(|e: String| format!("ias15 propagation failed: {e}"))?;

    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut stms: Vec<[f64; 36]> = Vec::with_capacity(t_eval.len());
    let mut sensitivities: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    for y in &result.states {
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(&y[..6]);
        states.push(s);
        if with_stm {
            let mut stm = [0.0_f64; 36];
            stm.copy_from_slice(&y[6..42]);
            stms.push(stm);
        }
        if n_params > 0 {
            let base = if with_stm { 42 } else { 6 };
            sensitivities.push(y[base..].to_vec());
        }
    }

    Ok(CompiledIas15Result {
        states,
        stms,
        sensitivities,
        times: result.times,
        n_steps: result.n_steps,
        n_rejected: result.n_rejected,
    })
}
