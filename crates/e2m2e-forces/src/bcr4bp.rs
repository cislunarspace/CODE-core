//! BCR4BP（双圆限制性四体问题）运动方程、雅可比与 STM 传播的 Rust 实现。
//!
//! 从 Python `BCR4BP_Dynamics` 迁移，纯数学（无量纲），不依赖 SPICE。在
//! CR3BP 之上叠加太阳质点摄动：太阳在会合系中作共面圆周运动，
//! `r_s(t) = a_s·(cos θ, sin θ, 0)`，`θ = θ0 + ω_s·t`，由解析公式给出。
//!
//! - `bcr4bp_eom`：6 维运动方程右端项（移植 `bcr4bp_dynamics.py` 的
//!   `equations_of_motion` + `sun_acceleration`）
//! - `bcr4bp_jacobian_6x6`：状态方程雅可比 A，CR3BP 伪势 Hessian 叠加太阳
//!   第三体 Hessian，科氏块同 CR3BP（移植 `compute_jacobian_A`）
//! - `propagate_bcr4bp` / `propagate_bcr4bp_stm`：PD78 传播，循环结构与
//!   `cr3bp::propagate_cr3bp(_stm)` 一致，保证 with_stm/纯状态 states 逐位相同
//!
//! 与 CR3BP 的区别：方程显式含时（太阳位置随 t 变化），RK callback 必须把
//! 当前步时间 `t` 传入 EOM；STM 变分方程 `dΦ/dt = A(t)·Φ` 中的 A 显式含时。

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// 最小距离钳位（无量纲），防止在天体/太阳位置处除零（对应
/// `dynamics.py:76` 的 `Dynamics.MIN_DISTANCE`）。
pub const MIN_DISTANCE: f64 = 1e-10;

/// 最小步长（相对于积分区间），防止步长坍缩。与 `cr3bp.rs` 一致。
const MIN_STEP: f64 = 1e-12;

/// 太阳在会合系中的解析位置 `[a_s·cos θ, a_s·sin θ, 0]`，`θ = θ0 + ω_s·t`。
///
/// 双圆近似：太阳绕地月质心作共面圆周运动，半径恒为 `sun_distance`。
/// 与 `BCR4BPSystem.sun_position`（`bcr4bp_system.py:149-169`）逐项一致。
fn sun_position(sun_distance: f64, sun_angular_rate: f64, sun_phase0: f64, t: f64) -> [f64; 3] {
    let theta = sun_phase0 + sun_angular_rate * t;
    [sun_distance * theta.cos(), sun_distance * theta.sin(), 0.0]
}

/// BCR4BP 6 维运动方程右端项 `[vx, vy, vz, ax, ay, az]`。
///
/// CR3BP 加速度（离心 + 双天体引力 + 科氏）叠加太阳摄动加速度
/// （直接项 + 间接项），移植自 `BCR4BP_Dynamics.equations_of_motion` 与
/// `sun_acceleration`（`bcr4bp_dynamics.py:65-114`）：
///
/// ```text
/// a_sun = -m_s · [ (r - r_s)/|r - r_s|³ + r_s/|r_s|³ ]
/// ```
///
/// 第一项为太阳对航天器的引力（直接项），第二项扣除太阳对系统质心的
/// 引力（间接项）。`|r_s| = sun_distance`（圆周运动，常数），间接项分母
/// 用 `sun_distance³`，与 Python 逐项等价。
pub fn bcr4bp_eom(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    state: &[f64; 6],
    t: f64,
) -> [f64; 6] {
    let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);

    // CR3BP 部分（与 cr3bp::cr3bp_eom 完全一致）
    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
    let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
        .sqrt()
        .max(MIN_DISTANCE);

    let inv_r1_3 = 1.0 / r1.powi(3);
    let inv_r2_3 = 1.0 / r2.powi(3);

    let ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) * inv_r1_3 - mu * (x - 1.0 + mu) * inv_r2_3;
    let ay = -2.0 * vx + y - (1.0 - mu) * y * inv_r1_3 - mu * y * inv_r2_3;
    let az = -(1.0 - mu) * z * inv_r1_3 - mu * z * inv_r2_3;

    // 太阳摄动（直接项 + 间接项）
    let rs = sun_position(sun_distance, sun_angular_rate, sun_phase0, t);
    let dx = x - rs[0];
    let dy = y - rs[1];
    let dz = z - rs[2];
    let d_norm = (dx * dx + dy * dy + dz * dz).sqrt().max(MIN_DISTANCE);
    let inv_d3 = 1.0 / d_norm.powi(3);
    // |r_s| ≡ sun_distance（圆周运动），间接项分母用 sun_distance³
    let inv_as3 = 1.0 / sun_distance.powi(3);
    let ax_sun = -mu_sun * (dx * inv_d3 + rs[0] * inv_as3);
    let ay_sun = -mu_sun * (dy * inv_d3 + rs[1] * inv_as3);
    let az_sun = -mu_sun * (dz * inv_d3 + rs[2] * inv_as3);

    [vx, vy, vz, ax + ax_sun, ay + ay_sun, az + az_sun]
}

/// BCR4BP 状态方程的 6×6 雅可比矩阵 A(t)，满足 `dΦ/dt = A(t)·Φ`。
///
/// 结构与 CR3BP 相同，左下块在伪势 Hessian 上叠加太阳第三体 Hessian
/// （移植 `compute_jacobian_A`，`bcr4bp_dynamics.py:116-153`）：
///
/// ```text
/// J_sun = -m_s · ( I/|d|³ - 3·d·dᵀ/|d|⁵ ),   d = r - r_s(t)
/// ```
///
/// 间接项 `-m_s·r_s/|r_s|³` 不依赖航天器位置，偏导为零。科氏块
/// `A[3][4]=2`、`A[4][3]=-2` 同 CR3BP。
pub fn bcr4bp_jacobian_6x6(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    state: &[f64; 6],
    t: f64,
) -> [[f64; 6]; 6] {
    let (x, y, z) = (state[0], state[1], state[2]);

    // --- CR3BP 伪势 Hessian（potential.py: pseudo_potential_hessian）---
    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
    let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
        .sqrt()
        .max(MIN_DISTANCE);

    let inv_r1_3 = 1.0 / r1.powi(3);
    let inv_r2_3 = 1.0 / r2.powi(3);
    let inv_r1_5 = inv_r1_3 / (r1 * r1);
    let inv_r2_5 = inv_r2_3 / (r2 * r2);

    let xm = 1.0 - mu;
    let dx1 = x + mu;
    let dx2 = x - 1.0 + mu;

    let u_xx = 1.0
        - xm * (inv_r1_3 - 3.0 * dx1 * dx1 * inv_r1_5)
        - mu * (inv_r2_3 - 3.0 * dx2 * dx2 * inv_r2_5);
    let u_yy =
        1.0 - xm * (inv_r1_3 - 3.0 * y * y * inv_r1_5) - mu * (inv_r2_3 - 3.0 * y * y * inv_r2_5);
    let u_zz = -xm * (inv_r1_3 - 3.0 * z * z * inv_r1_5) - mu * (inv_r2_3 - 3.0 * z * z * inv_r2_5);
    let u_xy = 3.0 * xm * dx1 * y * inv_r1_5 + 3.0 * mu * dx2 * y * inv_r2_5;
    let u_xz = 3.0 * xm * dx1 * z * inv_r1_5 + 3.0 * mu * dx2 * z * inv_r2_5;
    let u_yz = 3.0 * xm * y * z * inv_r1_5 + 3.0 * mu * y * z * inv_r2_5;

    // --- 太阳第三体 Hessian：J_sun[i][j] = -m_s·(δ_ij/d³ - 3·d_i·d_j/d⁵) ---
    let rs = sun_position(sun_distance, sun_angular_rate, sun_phase0, t);
    let d = [x - rs[0], y - rs[1], z - rs[2]];
    let d_norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        .sqrt()
        .max(MIN_DISTANCE);
    let inv_d3 = 1.0 / d_norm.powi(3);
    let inv_d5 = inv_d3 / (d_norm * d_norm);
    let coef = -mu_sun;
    let js_xx = coef * (inv_d3 - 3.0 * d[0] * d[0] * inv_d5);
    let js_yy = coef * (inv_d3 - 3.0 * d[1] * d[1] * inv_d5);
    let js_zz = coef * (inv_d3 - 3.0 * d[2] * d[2] * inv_d5);
    let js_xy = coef * (0.0 - 3.0 * d[0] * d[1] * inv_d5);
    let js_xz = coef * (0.0 - 3.0 * d[0] * d[2] * inv_d5);
    let js_yz = coef * (0.0 - 3.0 * d[1] * d[2] * inv_d5);

    let mut a = [[0.0_f64; 6]; 6];
    // 上半：∂(v)/∂(r,v) = [0, I]
    a[0][3] = 1.0;
    a[1][4] = 1.0;
    a[2][5] = 1.0;
    // 左下：伪势 Hessian + 太阳第三体 Hessian
    a[3][0] = u_xx + js_xx;
    a[3][1] = u_xy + js_xy;
    a[3][2] = u_xz + js_xz;
    a[4][0] = u_xy + js_xy;
    a[4][1] = u_yy + js_yy;
    a[4][2] = u_yz + js_yz;
    a[5][0] = u_xz + js_xz;
    a[5][1] = u_yz + js_yz;
    a[5][2] = u_zz + js_zz;
    // 右下：科氏块（z 行无科氏）
    a[3][4] = 2.0;
    a[4][3] = -2.0;
    a
}

/// STM 变分方程导数：`dΦ/dt = A·Φ`（完整 6×6 矩阵乘，含科氏块）。
///
/// `stm` 为 6×6 行优先展平（36 维）。与 `cr3bp::stm_derivative` 同一纯数学
/// （`dΦ/dt = A·Φ` 与力模型无关），此处照搬而非跨模块复用，保持两个力
/// 模型模块相互独立（重复远比错误抽象便宜）。
fn stm_derivative(a: &[[f64; 6]; 6], stm: &[f64; 36]) -> [f64; 36] {
    let mut dstm = [0.0_f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i][k] * stm[k * 6 + j];
            }
            dstm[i * 6 + j] = s;
        }
    }
    dstm
}

/// 42 维增广状态右端项 `[v(3), a(3), dΦ/dt(36)]`。A(t) 显式含时，故传入 `t`。
fn augmented_eom(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t: f64,
    augmented: &[f64],
) -> Vec<f64> {
    let mut state6 = [0.0_f64; 6];
    state6.copy_from_slice(&augmented[..6]);
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&augmented[6..42]);

    let deriv = bcr4bp_eom(
        mu,
        mu_sun,
        sun_distance,
        sun_angular_rate,
        sun_phase0,
        &state6,
        t,
    );
    let a = bcr4bp_jacobian_6x6(
        mu,
        mu_sun,
        sun_distance,
        sun_angular_rate,
        sun_phase0,
        &state6,
        t,
    );
    let dstm = stm_derivative(&a, &stm);

    let mut result = vec![0.0_f64; 42];
    result[..6].copy_from_slice(&deriv);
    result[6..42].copy_from_slice(&dstm);
    result
}

/// 传播结果（纯状态）。
pub struct Bcr4bpStateResult {
    /// 各 `t_eval` 时刻的状态 `[x, y, z, vx, vy, vz]`。
    pub states: Vec<[f64; 6]>,
    /// 实际输出的时间点（与 `t_eval` 一一对应）。
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 传播结果（状态 + STM）。
pub struct Bcr4bpStmResult {
    pub states: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的 STM（6×6 行优先展平）。
    pub stms: Vec<[f64; 36]>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// BCR4BP 6 维纯状态传播（PD78）。支持双向积分。
///
/// 循环结构与 `cr3bp::propagate_cr3bp` 一致：t_eval 空守卫、起点跟随
/// `t_eval[0]`、步长不超过下一输出点、按 `t_span` 方向带符号步进。唯一差别
/// 是 RK callback 把当前步时间 `t` 传入 EOM（BCR4BP 显式含时）。控制器用
/// `rtol` 作误差阈值，与 STM 路径一致以保证 states 逐位相同。
///
/// # 错误
/// `t_eval` 为空、步长塌缩、或输出点数不足（不允许静默截断）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_bcr4bp(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<Bcr4bpStateResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }

    let method = RkMethod::Pd78;
    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut y = initial_state.to_vec();
    let mut t = t_span.0;
    let direction = (t_span.1 - t_span.0).signum();
    let span_abs = (t_span.1 - t_span.0).abs();
    let mut h = direction * span_abs.min(h_max);

    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    if (t_span.0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t_span.0);
        states.push(*initial_state);
        eval_idx = 1;
    }

    while n_steps < s_max
        && ((direction > 0.0 && t < t_eval[t_eval.len() - 1] - 1e-12)
            || (direction < 0.0 && t > t_eval[t_eval.len() - 1] + 1e-12))
    {
        n_steps += 1;

        if eval_idx < t_eval.len() {
            let t_next = t_eval[eval_idx];
            if (direction > 0.0 && t + h > t_next) || (direction < 0.0 && t + h < t_next) {
                h = t_next - t;
            }
        }
        if h > h_max {
            h = h_max;
        }
        if h < -h_max {
            h = -h_max;
        }
        if h.abs() < MIN_STEP * span_abs {
            return Err(format!(
                "step size collapsed below minimum after {} steps",
                n_steps
            ));
        }

        // BCR4BP 显式含时：必须把当前步时间 ti 传入 EOM（太阳位置随 t 变化）
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let mut s = [0.0_f64; 6];
            s.copy_from_slice(&yi[..6]);
            Ok(bcr4bp_eom(
                mu,
                mu_sun,
                sun_distance,
                sun_angular_rate,
                sun_phase0,
                &s,
                ti,
            )
            .to_vec())
        };

        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, None)
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            while eval_idx < t_eval.len() && direction * (t - t_eval[eval_idx]) >= -1e-9 {
                times.push(t_eval[eval_idx]);
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(&y[..6]);
                states.push(s);
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

    Ok(Bcr4bpStateResult {
        states,
        times,
        n_steps,
        n_rejected,
    })
}

/// BCR4BP 42 维增广状态传播（状态 + STM，PD78）。
///
/// 初始 STM 设为单位矩阵，拼接为 42 维增广状态后积分。步长误差控制只统计
/// 前 6 维（`error_dim = Some(6)`），避免 STM 分量主导步长选择。循环结构
/// 与 `propagate_bcr4bp` 一致，保证两条路径的 states 逐位相同。
///
/// # 错误
/// `t_eval` 为空、步长塌缩、或输出点数不足（不允许静默截断）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_bcr4bp_stm(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<Bcr4bpStmResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }

    let method = RkMethod::Pd78;
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
    let direction = (t_span.1 - t_span.0).signum();
    let span_abs = (t_span.1 - t_span.0).abs();
    let mut h = direction * span_abs.min(h_max);

    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut stms: Vec<[f64; 36]> = Vec::with_capacity(t_eval.len());
    if (t_span.0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t_span.0);
        states.push(*initial_state);
        let mut stm0 = [0.0_f64; 36];
        for i in 0..6 {
            stm0[i * 6 + i] = 1.0;
        }
        stms.push(stm0);
        eval_idx = 1;
    }

    while n_steps < s_max
        && ((direction > 0.0 && t < t_eval[t_eval.len() - 1] - 1e-12)
            || (direction < 0.0 && t > t_eval[t_eval.len() - 1] + 1e-12))
    {
        n_steps += 1;

        if eval_idx < t_eval.len() {
            let t_next = t_eval[eval_idx];
            if (direction > 0.0 && t + h > t_next) || (direction < 0.0 && t + h < t_next) {
                h = t_next - t;
            }
        }
        if h > h_max {
            h = h_max;
        }
        if h < -h_max {
            h = -h_max;
        }
        if h.abs() < MIN_STEP * span_abs {
            return Err(format!(
                "step size collapsed below minimum after {} steps",
                n_steps
            ));
        }

        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            Ok(augmented_eom(
                mu,
                mu_sun,
                sun_distance,
                sun_angular_rate,
                sun_phase0,
                ti,
                yi,
            ))
        };

        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, Some(6))
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            while eval_idx < t_eval.len() && direction * (t - t_eval[eval_idx]) >= -1e-9 {
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

    Ok(Bcr4bpStmResult {
        states,
        stms,
        times,
        n_steps,
        n_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 地月系质量参数（单一来源：constants.toml → e2m2e_propagation 生成；DE421）
    const MU_EARTH_MOON: f64 = e2m2e_propagation::constants::DATUM_DE421_MU;
    const MU_SUN: f64 = 328900.56; // GM_sun / GM_EMB（DE440）
    const SUN_DISTANCE: f64 = 389.0; // 日地平均距离 / 地月距离
    const SUN_OMEGA: f64 = -0.92520; // 会合系中太阳逆行角速度

    /// EOM 在 m_s=0 时退化为 CR3BP（与 cr3bp::cr3bp_eom 逐项一致）。
    #[test]
    fn eom_zero_sun_matches_cr3bp() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        for t in [0.0_f64, 0.7, 3.3] {
            let d_bcr = bcr4bp_eom(MU_EARTH_MOON, 0.0, SUN_DISTANCE, SUN_OMEGA, 0.0, &state, t);
            // 手工重算 CR3BP EOM（同 cr3bp::cr3bp_eom 公式）
            let (x, y, z, vx, vy, _vz) = (0.5, 0.5, 0.1, 0.1, -0.2, 0.05);
            let mu = MU_EARTH_MOON;
            let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
            let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
                .sqrt()
                .max(MIN_DISTANCE);
            let ax = 2.0 * vy + x
                - (1.0 - mu) * (x + mu) / r1.powi(3)
                - mu * (x - 1.0 + mu) / r2.powi(3);
            let ay = -2.0 * vx + y - (1.0 - mu) * y / r1.powi(3) - mu * y / r2.powi(3);
            let az = -(1.0 - mu) * z / r1.powi(3) - mu * z / r2.powi(3);

            assert!((d_bcr[3] - ax).abs() < 1e-12, "t={} ax", t);
            assert!((d_bcr[4] - ay).abs() < 1e-12, "t={} ay", t);
            assert!((d_bcr[5] - az).abs() < 1e-12, "t={} az", t);
        }
    }

    /// 太阳项加速度：手工重算直接项 + 间接项对照。
    #[test]
    fn eom_sun_term_matches_manual() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let t = 0.7;
        let d_full = bcr4bp_eom(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            &state,
            t,
        );
        let d_zero = bcr4bp_eom(MU_EARTH_MOON, 0.0, SUN_DISTANCE, SUN_OMEGA, 0.0, &state, t);

        // 手工重算太阳加速度
        let rs = sun_position(SUN_DISTANCE, SUN_OMEGA, 0.0, t);
        let dx = state[0] - rs[0];
        let dy = state[1] - rs[1];
        let dz = state[2] - rs[2];
        let d_norm = (dx * dx + dy * dy + dz * dz).sqrt().max(MIN_DISTANCE);
        let inv_as3 = 1.0 / SUN_DISTANCE.powi(3);
        let ax_sun = -MU_SUN * (dx / d_norm.powi(3) + rs[0] * inv_as3);
        let ay_sun = -MU_SUN * (dy / d_norm.powi(3) + rs[1] * inv_as3);
        let az_sun = -MU_SUN * (dz / d_norm.powi(3) + rs[2] * inv_as3);

        assert!((d_full[3] - d_zero[3] - ax_sun).abs() < 1e-6, "ax_sun");
        assert!((d_full[4] - d_zero[4] - ay_sun).abs() < 1e-6, "ay_sun");
        assert!((d_full[5] - d_zero[5] - az_sun).abs() < 1e-6, "az_sun");
    }

    /// 雅可比的加速度块与 EOM 中心差分一致（含太阳项，固定 t）。
    #[test]
    fn jacobian_matches_finite_difference() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let t = 0.7;
        let a = bcr4bp_jacobian_6x6(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            &state,
            t,
        );

        // ∂f/∂state 6 列全部用中心差分（太阳项对速度偏导为零，但统一验证）
        let h = 1e-7;
        for col in 0..6 {
            let mut sp = state;
            let mut sm = state;
            sp[col] += h;
            sm[col] -= h;
            let dp = bcr4bp_eom(MU_EARTH_MOON, MU_SUN, SUN_DISTANCE, SUN_OMEGA, 0.0, &sp, t);
            let dm = bcr4bp_eom(MU_EARTH_MOON, MU_SUN, SUN_DISTANCE, SUN_OMEGA, 0.0, &sm, t);
            for row in 0..6 {
                let fd = (dp[row] - dm[row]) / (2.0 * h);
                assert!(
                    (a[row][col] - fd).abs() < 1e-6,
                    "A[{}][{}] 解析={} 差分={}",
                    row,
                    col,
                    a[row][col],
                    fd
                );
            }
        }

        // 科氏块
        assert!((a[3][4] - 2.0).abs() < 1e-15);
        assert!((a[4][3] - (-2.0)).abs() < 1e-15);
        // 上半单位阵
        assert!((a[0][3] - 1.0).abs() < 1e-15);
        assert!((a[1][4] - 1.0).abs() < 1e-15);
        assert!((a[2][5] - 1.0).abs() < 1e-15);
    }

    /// STM 导数 dΦ/dt = A·Φ 对单位 Φ 等于 A。
    #[test]
    fn stm_derivative_identity() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let t = 0.7;
        let a = bcr4bp_jacobian_6x6(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            &state,
            t,
        );

        let mut stm = [0.0_f64; 36];
        for i in 0..6 {
            stm[i * 6 + i] = 1.0;
        }
        let dstm = stm_derivative(&a, &stm);

        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (dstm[i * 6 + j] - a[i][j]).abs() < 1e-12,
                    "dΦ/dt[{}][{}] = {} vs A = {}",
                    i,
                    j,
                    dstm[i * 6 + j],
                    a[i][j]
                );
            }
        }
    }

    /// 端到端：传播一个短弧，STM 与纯状态路径的有限差分一致。
    #[test]
    fn propagate_stm_vs_finite_difference() {
        let state0 = [0.5, 0.2, 0.1, 0.1, 0.3, 0.05];
        let t_eval = vec![0.0, 1.0];

        let result = propagate_bcr4bp_stm(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            (0.0, 1.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        assert_eq!(result.states.len(), 2);
        let r1 = result.states[1];
        let stm = &result.stms[1];

        // 位置扰动验证 ∂r(T)/∂r(0)。差分用纯状态路径（固定 t0=0，扰动不改起算时刻）
        let h = 1e-7;
        for dim in 0..3 {
            let mut sp = state0;
            sp[dim] += h;
            let rp = propagate_bcr4bp(
                MU_EARTH_MOON,
                MU_SUN,
                SUN_DISTANCE,
                SUN_OMEGA,
                0.0,
                (0.0, 1.0),
                &t_eval,
                &sp,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap()
            .states[1];
            for row in 0..3 {
                let fd = (rp[row] - r1[row]) / h;
                assert!(
                    (stm[row * 6 + dim] - fd).abs() / fd.abs().max(1e-6) < 1e-3,
                    "STM[{}][{}] 解析={} 差分={}",
                    row,
                    dim,
                    stm[row * 6 + dim],
                    fd
                );
            }
        }
    }

    /// with_stm 与纯状态路径的 states 逐位一致。
    #[test]
    fn with_stm_and_state_only_match() {
        let state0 = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0];
        let t_eval = vec![0.0, 0.5, 1.0];

        let r_state = propagate_bcr4bp(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            (0.0, 1.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        let r_stm = propagate_bcr4bp_stm(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            (0.0, 1.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();

        assert_eq!(r_state.states.len(), r_stm.states.len());
        for k in 0..r_state.states.len() {
            for i in 0..6 {
                assert!(
                    (r_state.states[k][i] - r_stm.states[k][i]).abs() < 1e-14,
                    "states[{}][{}] 不一致: {} vs {}",
                    k,
                    i,
                    r_state.states[k][i],
                    r_stm.states[k][i]
                );
            }
        }
    }

    /// 双向积分：向后传播应把 forward 末态还原回初值。
    #[test]
    fn backward_integration_roundtrip() {
        let state0 = [0.5, 0.2, 0.1, 0.1, 0.3, 0.05];
        let fwd = propagate_bcr4bp(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            (0.0, 1.0),
            &[0.0, 1.0],
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        let state1 = fwd.states[1];

        let bwd = propagate_bcr4bp(
            MU_EARTH_MOON,
            MU_SUN,
            SUN_DISTANCE,
            SUN_OMEGA,
            0.0,
            (1.0, 0.0),
            &[1.0, 0.0],
            &state1,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        assert_eq!(bwd.states.len(), 2);
        assert!((bwd.times[0] - 1.0).abs() < 1e-12);
        assert!(bwd.times[1].abs() < 1e-12);
        for (i, (got, &want)) in bwd.states[1].iter().zip(state0.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "backward roundtrip state[{}] = {} vs {}",
                i,
                got,
                want
            );
        }
    }
}
