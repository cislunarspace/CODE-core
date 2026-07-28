//! J2000 惯性系 N 体力模型 + STM 变分方程的 Rust 实现。
//!
//! 从 Python `EphemerisDynamics` 迁移，实现：
//! 1. `compute_nbody_acceleration_and_jacobian`：单次遍历所有天体，同时计算
//!    加速度 a 和雅可比 ∂a/∂r
//! 2. `stm_derivative`：STM 变分方程右端项 dΦ/dt = A·Φ
//!
//! 供 `compiled_stm` 模块复用 `stm_derivative`。

use crate::spk_accel;

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
pub fn compute_nbody_acceleration_and_jacobian(
    config: &NBodyConfig,
    et: f64,
    r_sc: &[f64; 3],
) -> Result<([f64; 3], [[f64; 3]; 3]), String> {
    let mut acc = [0.0_f64; 3];
    let mut jac = [[0.0_f64; 3]; 3];

    for (body, gm) in config.bodies.iter().zip(config.gm_values.iter()) {
        if body == &config.origin {
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

/// STM 变分方程的右端项：dΦ/dt = A · Φ。
///
/// A = [0₃ₓ₃  I₃ₓ₃; ∂a/∂r  0₃ₓ₃]
pub fn stm_derivative(stm: &[f64; 36], jac_da_dr: &[[f64; 3]; 3]) -> [f64; 36] {
    let mut dstm = [0.0_f64; 36];
    for col in 0..6 {
        for row in 0..3 {
            dstm[row * 6 + col] = stm[(row + 3) * 6 + col];
        }
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
