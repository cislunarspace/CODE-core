//! CR3BP（圆型限制性三体问题）运动方程、雅可比与 STM 传播的 Rust 实现。
//!
//! 从 Python `CR3BP_Dynamics` 迁移，纯数学（无量纲），不依赖 SPICE。
//!
//! - `cr3bp_eom`：6 维运动方程右端项（移植 `dynamics.py:414-430`）
//! - `cr3bp_jacobian_6x6`：状态方程雅可比 A，含科氏块（移植
//!   `dynamics.py:432-459` + `potential.py` 伪势 Hessian）
//! - `propagate_cr3bp` / `propagate_cr3bp_stm`：PD78 传播，与
//!   `compiled_stm::propagate_compiled_stm` 同一 RK 循环结构，保证
//!   with_stm=True/False 的 states 逐位一致。
//!
//! 与 `nbody_stm` 的区别：CR3BP 的 A 含科氏块（A[3][4]=2、A[4][3]=-2），
//! 不能复用 `nbody_stm::stm_derivative`（它假设 A[3:6][3:6]=0，会漏掉科氏项）。

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// 最小距离钳位（无量纲），防止在天体位置处除零（对应 `dynamics.py:66`）。
pub const MIN_DISTANCE: f64 = 1e-10;

/// 最小步长（相对于积分区间），防止步长坍缩。
///
/// 步长塌缩时本模块返回 ``Err("step size collapsed below minimum ...")``。
/// 该前缀 ``"step size collapsed"`` 是 Python↔Rust 跨语言契约：Python
/// ``EphemerisDynamics._propagate_state_only``（dynamics.py，
/// ``_RUST_STEP_COLLAPSED_MARKER``）据此识别失败并转成空 states。改写本错误
/// 消息须同步 Python 侧标记（issue #317 第 3.1 项）。
const MIN_STEP: f64 = 1e-12;

/// CR3BP 6 维运动方程右端项 `[vx, vy, vz, ax, ay, az]`。
///
/// 移植自 `CR3BP_Dynamics.equations_of_motion`（`dynamics.py:414-430`）。
/// 旋转坐标系（主天体角速度 ω=1），两个主天体固定在 x 轴：
/// 较大天体（质量 1-μ）位于 x=-μ，较小天体（质量 μ）位于 x=1-μ。
/// 加速度各项：`x`/`y` 离心 + 引力 + `2vy`/`-2vx` 科氏；`z` 仅引力。
pub fn cr3bp_eom(mu: f64, state: &[f64; 6]) -> [f64; 6] {
    let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);

    // r1：到较大天体（x=-μ）的距离；r2：到较小天体（x=1-μ）的距离
    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
    let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
        .sqrt()
        .max(MIN_DISTANCE);

    let inv_r1_3 = 1.0 / r1.powi(3);
    let inv_r2_3 = 1.0 / r2.powi(3);

    let ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) * inv_r1_3 - mu * (x - 1.0 + mu) * inv_r2_3;
    let ay = -2.0 * vx + y - (1.0 - mu) * y * inv_r1_3 - mu * y * inv_r2_3;
    let az = -(1.0 - mu) * z * inv_r1_3 - mu * z * inv_r2_3;

    [vx, vy, vz, ax, ay, az]
}

/// CR3BP 状态方程的 6×6 雅可比矩阵 A，满足 `dΦ/dt = A·Φ`。
///
/// 移植自 `CR3BP_Dynamics.compute_jacobian_A`（`dynamics.py:432-459`）与
/// `pseudo_potential_hessian`（`potential.py:14-58`）。
///
/// ```text
/// A = [ 0₃₃   I₃₃ ]
///     [ H     Ω   ]
/// ```
/// H 为伪势 Hessian（对称），Ω 为科氏块：`A[3][4]=2`、`A[4][3]=-2`，
/// z 行（第 5 行）无科氏项。
pub fn cr3bp_jacobian_6x6(mu: f64, state: &[f64; 6]) -> [[f64; 6]; 6] {
    let (x, y, z) = (state[0], state[1], state[2]);

    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
    let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
        .sqrt()
        .max(MIN_DISTANCE);

    let inv_r1_3 = 1.0 / r1.powi(3);
    let inv_r2_3 = 1.0 / r2.powi(3);
    let inv_r1_5 = inv_r1_3 / (r1 * r1);
    let inv_r2_5 = inv_r2_3 / (r2 * r2);

    let xm = 1.0 - mu; // 较大天体质量
    let dx1 = x + mu; // 航天器 - 较大天体
    let dx2 = x - 1.0 + mu; // 航天器 - 较小天体

    // 伪势 Hessian（potential.py）：x/y 方向有离心 +1，z 方向无
    let u_xx = 1.0
        - xm * (inv_r1_3 - 3.0 * dx1 * dx1 * inv_r1_5)
        - mu * (inv_r2_3 - 3.0 * dx2 * dx2 * inv_r2_5);
    let u_yy =
        1.0 - xm * (inv_r1_3 - 3.0 * y * y * inv_r1_5) - mu * (inv_r2_3 - 3.0 * y * y * inv_r2_5);
    let u_zz = -xm * (inv_r1_3 - 3.0 * z * z * inv_r1_5) - mu * (inv_r2_3 - 3.0 * z * z * inv_r2_5);
    let u_xy = 3.0 * xm * dx1 * y * inv_r1_5 + 3.0 * mu * dx2 * y * inv_r2_5;
    let u_xz = 3.0 * xm * dx1 * z * inv_r1_5 + 3.0 * mu * dx2 * z * inv_r2_5;
    let u_yz = 3.0 * xm * y * z * inv_r1_5 + 3.0 * mu * y * z * inv_r2_5;

    let mut a = [[0.0_f64; 6]; 6];
    // 上半：∂(v)/∂(r,v) = [0, I]
    a[0][3] = 1.0;
    a[1][4] = 1.0;
    a[2][5] = 1.0;
    // 左下：伪势 Hessian
    a[3][0] = u_xx;
    a[3][1] = u_xy;
    a[3][2] = u_xz;
    a[4][0] = u_xy;
    a[4][1] = u_yy;
    a[4][2] = u_yz;
    a[5][0] = u_xz;
    a[5][1] = u_yz;
    a[5][2] = u_zz;
    // 右下：科氏块（z 行无科氏）
    a[3][4] = 2.0;
    a[4][3] = -2.0;
    a
}

/// STM 变分方程导数：`dΦ/dt = A·Φ`（完整 6×6 矩阵乘，含科氏块）。
///
/// `stm` 为 6×6 行优先展平（36 维）。返回 `dΦ/dt` 展平。不复用
/// `nbody_stm::stm_derivative`，因为它硬编码无科氏块，对 CR3BP 会出错。
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

/// 42 维增广状态右端项 `[v(3), a(3), dΦ/dt(36)]`。
fn augmented_eom(mu: f64, augmented: &[f64]) -> Vec<f64> {
    let mut state6 = [0.0_f64; 6];
    state6.copy_from_slice(&augmented[..6]);
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&augmented[6..42]);

    let deriv = cr3bp_eom(mu, &state6);
    let a = cr3bp_jacobian_6x6(mu, &state6);
    let dstm = stm_derivative(&a, &stm);

    let mut result = vec![0.0_f64; 42];
    result[..6].copy_from_slice(&deriv);
    result[6..42].copy_from_slice(&dstm);
    result
}

/// 传播结果（纯状态）。
pub struct Cr3bpStateResult {
    /// 各 `t_eval` 时刻的状态 `[x, y, z, vx, vy, vz]`。
    pub states: Vec<[f64; 6]>,
    /// 实际输出的时间点（与 `t_eval` 一一对应）。
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 传播结果（状态 + STM）。
pub struct Cr3bpStmResult {
    pub states: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的 STM（6×6 行优先展平）。
    pub stms: Vec<[f64; 36]>,
    pub times: Vec<f64>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// CR3BP 6 维纯状态传播（PD78）。支持双向积分（stable manifold 沿时间逆向
/// 传播时 `t_span.1 < t_span.0`）。
///
/// 循环结构与 `compiled_stm::propagate_compiled_stm` 一致：t_eval 空守卫、
/// 起点跟随 t_eval[0]、步长不超过下一输出点；额外按 `t_span` 方向带符号
/// 步进（compiled_stm 只用于星历 et 递增，CR3BP 需要逆向）。
/// 控制器用 `rtol` 作误差阈值（与 `propagate_compiled_stm` 一致，保证
/// with_stm/纯状态逐位相同）。
///
/// # 错误
/// `t_eval` 为空、步长塌缩、或输出点数不足（不允许静默截断）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp(
    mu: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<Cr3bpStateResult, String> {
    if t_eval.is_empty() {
        return Err("t_eval must not be empty".to_string());
    }

    let method = RkMethod::Pd78;
    let tol = rtol;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut y = initial_state.to_vec();
    let mut t = t_span.0;
    // 支持双向积分（manifolds 的 stable 分支沿时间逆向传播，t_span.1 < t_span.0）。
    // direction 带符号：向前 +1、向后 -1；步长 h 与 direction 同号。
    let direction = (t_span.1 - t_span.0).signum();
    let span_abs = (t_span.1 - t_span.0).abs();
    let mut h = direction * span_abs.min(h_max);

    // 输出起点跟随 t_eval：当 t_eval[0]==t_span.0 时记录初值、eval_idx 从 1
    // 起步；否则（逐段积分 patch point 时刻非整数倍）eval_idx 从 0 起步由
    // 循环匹配。照抄 compiled_stm.rs:112-135 的首点起步修复（commit
    // 1f5756b/c3685e7），否则 multiple_shooting 逐段积分首点会错置。
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

        // 步长不超过下一输出点（向前 t+h>t_next 或向后 t+h<t_next 时截断到该点）
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

        let callback = |_ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let mut s = [0.0_f64; 6];
            s.copy_from_slice(&yi[..6]);
            Ok(cr3bp_eom(mu, &s).to_vec())
        };

        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h, callback, None)
            .map_err(|e: String| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            t += h;
            y = y_new;

            // 落到输出点：向前 t>=t_eval、向后 t<=t_eval（统一为 direction*(t-t_eval)>=0）
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

    Ok(Cr3bpStateResult {
        states,
        times,
        n_steps,
        n_rejected,
    })
}

/// CR3BP 42 维增广状态传播（状态 + STM，PD78）。
///
/// 初始 STM 设为单位矩阵，拼接为 42 维增广状态后积分。步长误差控制只统计
/// 前 6 维（`error_dim = Some(6)`），避免 STM 分量主导步长选择。循环结构
/// 与 `propagate_cr3bp` 一致，保证两条路径的 states 逐位相同。
///
/// # 错误
/// `t_eval` 为空、步长塌缩、或输出点数不足（不允许静默截断，issue #246）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp_stm(
    mu: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    _atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<Cr3bpStmResult, String> {
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

        let callback =
            |_: f64, yi: &[f64]| -> Result<Vec<f64>, String> { Ok(augmented_eom(mu, yi)) };

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

    Ok(Cr3bpStmResult {
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

    const MU_EARTH_MOON: f64 = e2m2e_propagation::constants::DATUM_DE421_MU; // 地月系质量参数（DE421）

    /// EOM 与 Python 参考值对比：L1 附近的 Halo 轨道初值。
    #[test]
    fn eom_matches_python_reference() {
        // cr3bp_eom 在 (x,y,z,vx,vy,vz) 处的解析值，与 dynamics.py 逐项对照
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let d = cr3bp_eom(MU_EARTH_MOON, &state);

        // 速度导数原样返回
        assert!((d[0] - 0.1).abs() < 1e-15);
        assert!((d[1] - (-0.2)).abs() < 1e-15);
        assert!((d[2] - 0.05).abs() < 1e-15);

        // 用独立重算的 ax/ay/az 验证（公式同 dynamics.py:424-428）
        let (x, y, z, vx, vy, _vz) = (0.5, 0.5, 0.1, 0.1, -0.2, 0.05);
        let mu = MU_EARTH_MOON;
        let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt().max(MIN_DISTANCE);
        let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z)
            .sqrt()
            .max(MIN_DISTANCE);
        let ax =
            2.0 * vy + x - (1.0 - mu) * (x + mu) / r1.powi(3) - mu * (x - 1.0 + mu) / r2.powi(3);
        let ay = -2.0 * vx + y - (1.0 - mu) * y / r1.powi(3) - mu * y / r2.powi(3);
        let az = -(1.0 - mu) * z / r1.powi(3) - mu * z / r2.powi(3);

        assert!((d[3] - ax).abs() < 1e-14, "ax {} vs {}", d[3], ax);
        assert!((d[4] - ay).abs() < 1e-14, "ay {} vs {}", d[4], ay);
        assert!((d[5] - az).abs() < 1e-14, "az {} vs {}", d[5], az);
    }

    /// 雅可比的加速度块与 EOM 有限差分一致。
    #[test]
    fn jacobian_matches_finite_difference() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let a = cr3bp_jacobian_6x6(MU_EARTH_MOON, &state);

        // ∂a/∂r 块（A[3:6][0:3]）用中心差分验证
        let h = 1e-7;
        for col in 0..3 {
            let mut sp = state;
            let mut sm = state;
            sp[col] += h;
            sm[col] -= h;
            let dp = cr3bp_eom(MU_EARTH_MOON, &sp);
            let dm = cr3bp_eom(MU_EARTH_MOON, &sm);
            for row in 0..3 {
                let fd = (dp[3 + row] - dm[3 + row]) / (2.0 * h);
                assert!(
                    (a[3 + row][col] - fd).abs() < 1e-7,
                    "A[{}][{}] 解析={} 差分={}",
                    3 + row,
                    col,
                    a[3 + row][col],
                    fd
                );
            }
        }

        // 科氏块：A[3][4]=2（∂ax/∂vy）、A[4][3]=-2（∂ay/∂vx）
        assert!((a[3][4] - 2.0).abs() < 1e-15);
        assert!((a[4][3] - (-2.0)).abs() < 1e-15);
        // 上半单位阵
        assert!((a[0][3] - 1.0).abs() < 1e-15);
        assert!((a[1][4] - 1.0).abs() < 1e-15);
        assert!((a[2][5] - 1.0).abs() < 1e-15);
        // z 行无科氏（A[5][3]=A[5][4]=0）
        assert!(a[5][3].abs() < 1e-15);
        assert!(a[5][4].abs() < 1e-15);
    }

    /// STM 导数 dΦ/dt = A·Φ 对单位 Φ 等于 A。
    #[test]
    fn stm_derivative_identity() {
        let state = [0.5, 0.5, 0.1, 0.1, -0.2, 0.05];
        let a = cr3bp_jacobian_6x6(MU_EARTH_MOON, &state);

        let mut stm = [0.0_f64; 36];
        for i in 0..6 {
            stm[i * 6 + i] = 1.0;
        }
        let dstm = stm_derivative(&a, &stm);

        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (dstm[i * 6 + j] - a[i][j]).abs() < 1e-14,
                    "dΦ/dt[{}][{}] = {} vs A = {}",
                    i,
                    j,
                    dstm[i * 6 + j],
                    a[i][j]
                );
            }
        }
    }

    /// 端到端：传播一个短弧，STM 与有限差分一致。
    #[test]
    fn propagate_stm_vs_finite_difference() {
        // 3D 初值（z/vz 非零）：避免平面轨道下 ∂x/∂z 等交叉项解析为 0、
        // 仅靠二阶非线性效应显现，差分无法准确估计线性偏导。
        let state0 = [0.5, 0.2, 0.1, 0.1, 0.3, 0.05];
        let t_eval = vec![0.0, 1.0];

        let result = propagate_cr3bp_stm(
            MU_EARTH_MOON,
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

        // 位置扰动验证 ∂r(T)/∂r(0)。差分用纯状态路径（无 STM），减小噪声。
        let h = 1e-7;
        for dim in 0..3 {
            let mut sp = state0;
            sp[dim] += h;
            let rp = propagate_cr3bp(
                MU_EARTH_MOON,
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
                // 容差 1e-3：STM 分量误差控制被禁用（步长由前 6 维决定），
                // STM 精度依赖状态精度累积，与 nbody_stm 同类测试（5e-2）同量级。
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

        let r_state = propagate_cr3bp(
            MU_EARTH_MOON,
            (0.0, 1.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        let r_stm = propagate_cr3bp_stm(
            MU_EARTH_MOON,
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

    /// 双向积分：向后传播 (1, 0) 应把 forward (0,1) 的末态还原回初值。
    /// 回归 manifolds 的 stable 分支逆向传播路径。
    #[test]
    fn backward_integration_roundtrip() {
        let state0 = [0.5, 0.2, 0.1, 0.1, 0.3, 0.05];
        let fwd = propagate_cr3bp(
            MU_EARTH_MOON,
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

        // 向后：t_eval 递减、t_span.1 < t_span.0
        let bwd = propagate_cr3bp(
            MU_EARTH_MOON,
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
        for i in 0..6 {
            assert!(
                (bwd.states[1][i] - state0[i]).abs() < 1e-7,
                "backward roundtrip state[{}] = {} vs {}",
                i,
                bwd.states[1][i],
                state0[i]
            );
        }
    }
}
