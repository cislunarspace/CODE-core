//! MEGNO 混沌指标传播内核（Cincotta & Simó 2000；Primer §7.1 式 142）。
//!
//! 在现有 CR3BP / BCR4BP 传播之上叠加**切变分方程**（6+6 维，非完整
//! 36 维 STM）与两个时间积分累加器：
//!
//! ```text
//! Y(t)  = (2/t)·∫₀ᵗ (δ̇/δ)·s ds ,   dI₁/dt = t·(δ·δ̇)/|δ|²
//! Ȳ(t)  = (1/t)·∫₀ᵗ Y(s) ds     ,   dI₂/dt = 2·I₁/t   (t > 0)
//! ```
//!
//! 增广状态 14 维：`[r(3), v(3), δr(3), δv(3), I₁, I₂]`。
//! 切向量 δ̇ = A·δ 的 A 复用 `cr3bp_jacobian_6x6` /
//! `bcr4bp_jacobian_6x6`（`propagate_cr3bp_stm` 同源雅可比）。
//!
//! 归一化：被积函数 (δ·δ̇)/|δ|² 对 δ 的整体缩放不变，故长弧积分中
//! δ 越界（|δ| ∉ [1e-100, 1e100]）时在步边界重归一为单位向量，
//! I₁/I₂ 无需补偿（式 142 的数值稳定化，REBOUND 同款做法）。
//!
//! 步长误差控制只统计前 6 维（与 `propagate_cr3bp_stm` 的
//! `error_dim = Some(6)` 同一口径），避免变分分量主导步长。

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

use crate::bcr4bp::{bcr4bp_eom, bcr4bp_jacobian_6x6};
use crate::cr3bp::{cr3bp_eom, cr3bp_jacobian_6x6};

/// 最小步长（相对于积分区间），与 `cr3bp.rs` 一致。
const MIN_STEP: f64 = 1e-12;

/// δ 模长的重归一化界（越界即在步边界缩回单位向量）。
const DELTA_NORM_LO: f64 = 1e-100;
const DELTA_NORM_HI: f64 = 1e100;

/// MEGNO 传播结果。
pub struct MegnoResult {
    /// 各 `t_eval` 时刻的状态 `[x, y, z, vx, vy, vz]`。
    pub states: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的切向量 `[δr(3), δv(3)]`（含重归一化史）。
    pub deltas: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的 Y(t)（式 142 上式）。
    pub y: Vec<f64>,
    /// 各 `t_eval` 时刻的 Ȳ(t)（式 142 下式；正则轨迹 → 2）。
    pub ybar: Vec<f64>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// CR3BP 14 维 MEGNO 增广状态右端项（t 只进累加器，不进力学）。
fn cr3bp_megno_eom(mu: f64, t: f64, y: &[f64]) -> Vec<f64> {
    let mut state6 = [0.0_f64; 6];
    state6.copy_from_slice(&y[..6]);
    let deriv = cr3bp_eom(mu, &state6);
    let a = cr3bp_jacobian_6x6(mu, &state6);

    let mut out = vec![0.0_f64; 14];
    out[..6].copy_from_slice(&deriv);
    // δ̇ = A·δ（只取 A 的速度行 3..6）
    for i in 0..3 {
        out[6 + i] = y[9 + i]; // δṙ = δv
        let mut s = 0.0;
        for k in 0..6 {
            s += a[3 + i][k] * y[6 + k];
        }
        out[9 + i] = s;
    }
    let delta_sq: f64 = y[6..12].iter().map(|v| v * v).sum();
    // δ·δ̇ 全相空间 6 维：δr·δv + δv·δv̇（δv̇ 已在 out[9..12]）。
    let delta_dot: f64 = y[6..9]
        .iter()
        .zip(y[9..12].iter())
        .map(|(r, v)| r * v)
        .sum::<f64>()
        + y[9..12]
            .iter()
            .zip(out[9..12].iter())
            .map(|(v, a)| v * a)
            .sum::<f64>();
    if delta_sq > 0.0 {
        out[12] = t * delta_dot / delta_sq; // dI₁/dt
        out[13] = if t > 0.0 { 2.0 * y[12] / t } else { 0.0 }; // dI₂/dt
    }
    out
}

/// BCR4BP 14 维 MEGNO 增广状态右端项（A(t) 显式含时）。
#[allow(clippy::too_many_arguments)]
fn bcr4bp_megno_eom(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t: f64,
    y: &[f64],
) -> Vec<f64> {
    let mut state6 = [0.0_f64; 6];
    state6.copy_from_slice(&y[..6]);
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

    let mut out = vec![0.0_f64; 14];
    out[..6].copy_from_slice(&deriv);
    for i in 0..3 {
        out[6 + i] = y[9 + i];
        let mut s = 0.0;
        for k in 0..6 {
            s += a[3 + i][k] * y[6 + k];
        }
        out[9 + i] = s;
    }
    let delta_sq: f64 = y[6..12].iter().map(|v| v * v).sum();
    // δ·δ̇ 全相空间 6 维（同 cr3bp 版）。
    let delta_dot: f64 = y[6..9]
        .iter()
        .zip(y[9..12].iter())
        .map(|(r, v)| r * v)
        .sum::<f64>()
        + y[9..12]
            .iter()
            .zip(out[9..12].iter())
            .map(|(v, a)| v * a)
            .sum::<f64>();
    if delta_sq > 0.0 {
        out[12] = t * delta_dot / delta_sq;
        out[13] = if t > 0.0 { 2.0 * y[12] / t } else { 0.0 };
    }
    out
}

/// 步边界 δ 重归一化（式 142 数值稳定化；被积函数缩放不变 ⇒ 累加器免补偿）。
fn renormalize_delta(y: &mut [f64]) {
    let norm: f64 = y[6..12].iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(DELTA_NORM_LO..=DELTA_NORM_HI).contains(&norm) && norm > 0.0 && norm.is_finite() {
        for v in y[6..12].iter_mut() {
            *v /= norm;
        }
    }
}

/// 由 14 维增广状态在 t 处换算 (Y, Ȳ)。
fn megno_values(t: f64, y: &[f64]) -> (f64, f64) {
    if t > 0.0 {
        (2.0 * y[12] / t, y[13] / t)
    } else {
        (0.0, 0.0)
    }
}

/// CR3BP MEGNO 传播主循环（PD78）。循环结构与 `propagate_cr3bp` 一致；
/// 步长误差只统计前 6 维。
///
/// # 错误
/// `t_eval` 为空、步长塌缩、或输出点数不足时返回 `String` 错误。
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp_megno(
    mu: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    initial_delta: Option<[f64; 6]>,
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<MegnoResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }
    let method = RkMethod::Pd78;
    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut y = vec![0.0_f64; 14];
    y[..6].copy_from_slice(initial_state);
    y[6..12].copy_from_slice(&initial_delta.unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let mut t = t_span.0;
    let direction = (t_span.1 - t_span.0).signum();
    let span_abs = (t_span.1 - t_span.0).abs();
    let mut h = direction * span_abs.min(h_max);

    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut ys: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut ybars: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut deltas: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    if (t_span.0 - t_eval[0]).abs() <= 1e-9 {
        let (yv, ybarv) = megno_values(t, &y);
        times.push(t_span.0);
        states.push(*initial_state);
        let mut d = [0.0_f64; 6];
        d.copy_from_slice(&y[6..12]);
        deltas.push(d);
        ys.push(yv);
        ybars.push(ybarv);
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

        let callback =
            |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> { Ok(cr3bp_megno_eom(mu, ti, yi)) };
        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, Some(6))
            .map_err(|e| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;
            renormalize_delta(&mut y);
            while eval_idx < t_eval.len() && direction * (t - t_eval[eval_idx]) >= -1e-9 {
                let (yv, ybarv) = megno_values(t, &y);
                times.push(t_eval[eval_idx]);
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(&y[..6]);
                states.push(s);
                let mut d = [0.0_f64; 6];
                d.copy_from_slice(&y[6..12]);
                deltas.push(d);
                ys.push(yv);
                ybars.push(ybarv);
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
    Ok(MegnoResult {
        states,
        deltas,
        y: ys,
        ybar: ybars,
        times,
        n_steps,
        n_rejected,
    })
}

/// BCR4BP MEGNO 传播主循环（PD78），太阳参数语义与 `propagate_bcr4bp` 一致。
#[allow(clippy::too_many_arguments)]
pub fn propagate_bcr4bp_megno(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    initial_delta: Option<[f64; 6]>,
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<MegnoResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }
    let method = RkMethod::Pd78;
    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut y = vec![0.0_f64; 14];
    y[..6].copy_from_slice(initial_state);
    y[6..12].copy_from_slice(&initial_delta.unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let mut t = t_span.0;
    let direction = (t_span.1 - t_span.0).signum();
    let span_abs = (t_span.1 - t_span.0).abs();
    let mut h = direction * span_abs.min(h_max);

    let mut eval_idx = 0usize;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    let mut ys: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut ybars: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut deltas: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
    if (t_span.0 - t_eval[0]).abs() <= 1e-9 {
        let (yv, ybarv) = megno_values(t, &y);
        times.push(t_span.0);
        states.push(*initial_state);
        let mut d = [0.0_f64; 6];
        d.copy_from_slice(&y[6..12]);
        deltas.push(d);
        ys.push(yv);
        ybars.push(ybarv);
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
            Ok(bcr4bp_megno_eom(
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
            .map_err(|e| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;
            renormalize_delta(&mut y);
            while eval_idx < t_eval.len() && direction * (t - t_eval[eval_idx]) >= -1e-9 {
                let (yv, ybarv) = megno_values(t, &y);
                times.push(t_eval[eval_idx]);
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(&y[..6]);
                states.push(s);
                let mut d = [0.0_f64; 6];
                d.copy_from_slice(&y[6..12]);
                deltas.push(d);
                ys.push(yv);
                ybars.push(ybarv);
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
    Ok(MegnoResult {
        states,
        deltas,
        y: ys,
        ybar: ybars,
        times,
        n_steps,
        n_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_EM: f64 = 1.21505e-2;

    /// Jacobi 常数（无量纲）。
    fn jacobi(mu: f64, s: &[f64; 6]) -> f64 {
        let (x, y, z) = (s[0], s[1], s[2]);
        let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt();
        let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z).sqrt();
        let v2 = s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
        x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - v2
    }

    #[test]
    fn regular_near_keplerian_orbit_converges_to_two() {
        // 主天体近旁小振幅圆轨道（r ≈ 0.02，v = sqrt(1/r)）：运动近似开普勒，
        // Ȳ → 2（式 142 的正则基准）。
        let state = [-(MU_EM) + 0.02, 0.0, 0.0, 0.0, (1.0f64 / 0.02).sqrt(), 0.0];
        let t_eval: Vec<f64> = (0..=200).map(|i| i as f64 * 1.0).collect();
        let result = propagate_cr3bp_megno(
            MU_EM,
            (0.0, 200.0),
            &t_eval,
            &state,
            None,
            1e-10,
            1e-10,
            None,
            None,
        )
        .unwrap();
        let ybar_final = *result.ybar.last().unwrap();
        assert!((ybar_final - 2.0).abs() < 0.05, "ybar = {}", ybar_final);
        // 守恒律不回退：11000 圈积分的 Jacobi 相对漂移 < 1e-7（theory 口径）。
        let c0 = jacobi(MU_EM, &state);
        let c1 = jacobi(MU_EM, result.states.last().unwrap());
        assert!(
            (c1 - c0).abs() / c0.abs() < 1e-7,
            "dC/C = {}",
            (c1 - c0).abs() / c0.abs()
        );
    }

    #[test]
    fn chaotic_gateway_state_grows_ybar() {
        // L1 附近 gateway 带采样（mu 地月）：已知混沌初值，Ȳ 单调抬升 > 2.5。
        // L1 ≈ x = 0.8369。
        let state = [0.8369, 0.0, 0.0, 0.0, 0.0, 0.35];
        let t_eval: Vec<f64> = (0..=400).map(|i| i as f64 * 0.5).collect();
        let result = propagate_cr3bp_megno(
            MU_EM,
            (0.0, 200.0),
            &t_eval,
            &state,
            None,
            1e-10,
            1e-10,
            None,
            None,
        )
        .unwrap();
        let mid = result.ybar[200];
        let final_v = *result.ybar.last().unwrap();
        assert!(final_v > 2.5, "ybar = {}", final_v);
        assert!(final_v >= mid, "ybar mid {} final {}", mid, final_v);
    }

    #[test]
    fn bcr4bp_megno_runs_and_conserves_cr3bp_part_far_from_sun() {
        // 太阳质量置零时 BCR4BP ≡ CR3BP：Ȳ 与 CR3BP 版一致。
        let state = [-(MU_EM) + 0.02, 0.0, 0.0, 0.0, (1.0f64 / 0.02).sqrt(), 0.0];
        let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 1.0).collect();
        let r1 = propagate_cr3bp_megno(
            MU_EM,
            (0.0, 100.0),
            &t_eval,
            &state,
            None,
            1e-10,
            1e-10,
            None,
            None,
        )
        .unwrap();
        let r2 = propagate_bcr4bp_megno(
            MU_EM,
            0.0,
            1000.0,
            0.0,
            0.0,
            (0.0, 100.0),
            &t_eval,
            &state,
            None,
            1e-10,
            1e-10,
            None,
            None,
        )
        .unwrap();
        for (a, b) in r1.ybar.iter().zip(r2.ybar.iter()) {
            assert!((a - b).abs() < 1e-9, "cr3bp {} vs bcr4bp {}", a, b);
        }
    }
}
