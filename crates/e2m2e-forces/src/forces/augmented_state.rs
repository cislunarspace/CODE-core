//! 7D 增广状态向量：[r(3), v(3), m(1)]
//!
//! 参考 GMAT CSALT 的设计，将质量纳入状态向量，支持小推力轨迹优化。
//!
//! ## 动力学方程
//!
//! ```text
//! ṙ = v
//! v̇ = a_gravity + a_thrust
//! ṁ = -u0 * T_max / (Isp * g0)
//! ```
//!
//! 其中：
//! - a_thrust = (u0 * T_max / m) * û
//! - u0 ∈ [0, 1]：推力幅值（throttle）
//! - T_max：最大推力（N）
//! - m：航天器质量（kg）
//! - û：推力方向单位向量

use super::compiled::{compute_total_acceleration, CompiledForce};

/// 7D 增广状态：[r(3), v(3), m(1)]
pub type AugmentedState7 = [f64; 7];

/// 7D 增广动力学方程。
///
/// # 参数
/// - `forces`: 力模型列表（重力等）
/// - `observer`: 传播系 origin 天体名（如 "EARTH"）
/// - `et`: 历元时刻（SPICE et 秒）
/// - `state`: 7D 状态 [x, y, z, vx, vy, vz, m]
/// - `t_max`: 最大推力（N）
/// - `isp`: 比冲（s）
/// - `throttle`: 推力幅值 u0 ∈ [0, 1]
/// - `direction`: 推力方向单位向量（惯性系）
///
/// # 返回
/// 7D 导数 [vx, vy, vz, ax, ay, az, mdot]
pub fn augmented_eom_7d(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &AugmentedState7,
    t_max: f64,
    isp: f64,
    throttle: f64,
    direction: &[f64; 3],
) -> Result<AugmentedState7, String> {
    // 提取状态分量
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let m = state[6];

    // 1. 计算重力加速度（6D）
    let state6 = [r[0], r[1], r[2], v[0], v[1], v[2]];
    let a_gravity = compute_total_acceleration(forces, et, &state6, observer)?;

    // 2. 计算推力加速度
    // T_max 单位为 N (kg·m/s²)，质量单位为 kg，加速度单位为 m/s²
    // 需要转换为 km/s²（除以 1000）
    let a_thrust_mag_m_s2 = (t_max / m) * throttle;
    let a_thrust_mag_km_s2 = a_thrust_mag_m_s2 / 1000.0;
    let a_thrust = [
        a_thrust_mag_km_s2 * direction[0],
        a_thrust_mag_km_s2 * direction[1],
        a_thrust_mag_km_s2 * direction[2],
    ];

    // 3. 计算质量流率
    // mdot = -u0 * T_max / (Isp * g0)
    // 单位：kg/s
    let g0 = 9.81; // m/s²
    let mdot = -throttle * t_max / (isp * g0);

    // 4. 组装 7D 导数
    Ok([
        v[0],                       // dx/dt
        v[1],                       // dy/dt
        v[2],                       // dz/dt
        a_gravity[0] + a_thrust[0], // dvx/dt
        a_gravity[1] + a_thrust[1], // dvy/dt
        a_gravity[2] + a_thrust[2], // dvz/dt
        mdot,                       // dm/dt
    ])
}

/// 42D 增广状态 + STM（用于多重打靶）。
///
/// 状态布局：[r(3), v(3), m(1), Φ(36)]
/// 其中 Φ 是 6×6 状态转移矩阵（对 [r, v] 的偏导数）。
///
/// 注意：质量 m 的 STM 不纳入计算，因为推力加速度与质量相关，
/// 但质量本身不作为优化变量（通过 throttle 控制）。
pub type AugmentedState42 = [f64; 43];

/// 7D 增广动力学方程（带 STM）。
///
/// # 参数
/// - `forces`: 力模型列表
/// - `observer`: 传播系 origin
/// - `et`: 历元时刻
/// - `state`: 43D 状态 [x, y, z, vx, vy, vz, m, Φ(36)]
/// - `t_max`: 最大推力（N）
/// - `isp`: 比冲（s）
/// - `throttle`: 推力幅值
/// - `direction`: 推力方向
///
/// # 返回
/// 43D 导数
pub fn augmented_eom_7d_with_stm(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &AugmentedState42,
    t_max: f64,
    isp: f64,
    throttle: f64,
    direction: &[f64; 3],
) -> Result<AugmentedState42, String> {
    // 提取状态分量
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let m = state[6];
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&state[7..43]);

    // 1. 计算重力加速度 + 雅可比（6D）
    let state6 = [r[0], r[1], r[2], v[0], v[1], v[2]];
    let (a_gravity, jac_da_dr) =
        super::compiled::compute_total_acceleration_and_jacobian(forces, et, &state6, observer)?;

    // 2. 计算推力加速度
    let a_thrust_mag_m_s2 = (t_max / m) * throttle;
    let a_thrust_mag_km_s2 = a_thrust_mag_m_s2 / 1000.0;
    let a_thrust = [
        a_thrust_mag_km_s2 * direction[0],
        a_thrust_mag_km_s2 * direction[1],
        a_thrust_mag_km_s2 * direction[2],
    ];

    // 3. 计算质量流率
    let g0 = 9.81;
    let mdot = -throttle * t_max / (isp * g0);

    // 4. 计算 STM 导数
    // dΦ/dt = A * Φ，其中 A 是雅可比矩阵
    // A = [0, I; da/dr, 0]（6×6）
    let dstm = super::nbody_stm::stm_derivative(&stm, &jac_da_dr);

    // 5. 组装 43D 导数
    let mut result = [0.0_f64; 43];
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    result[3] = a_gravity[0] + a_thrust[0];
    result[4] = a_gravity[1] + a_thrust[1];
    result[5] = a_gravity[2] + a_thrust[2];
    result[6] = mdot;
    result[7..43].copy_from_slice(&dstm);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmented_eom_7d_basic() {
        // 测试 7D 动力学方程
        let forces = vec![CompiledForce::PointMass { mu: 398600.4418 }];

        let state = [
            7000.0, 0.0, 0.0, // position (km)
            0.0, 7.5, 0.0,    // velocity (km/s)
            1500.0,           // mass (kg)
        ];

        let t_max = 1.0; // N
        let isp = 3000.0; // s
        let throttle = 0.8;
        let direction = [1.0, 0.0, 0.0];

        let result = augmented_eom_7d(
            &forces, "EARTH", 0.0, &state, t_max, isp, throttle, &direction,
        )
        .unwrap();

        // 验证位置导数 = 速度
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 7.5).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);

        // 验证质量流率
        let expected_mdot = -0.8 * 1.0 / (3000.0 * 9.81);
        assert!((result[6] - expected_mdot).abs() < 1e-10);
    }

    #[test]
    fn test_augmented_eom_7d_thrust_acceleration() {
        // 测试推力加速度计算
        let forces = vec![]; // 无重力

        let state = [
            7000.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 静止
            1000.0, // kg
        ];

        let t_max = 0.5; // N
        let isp = 3000.0;
        let throttle = 1.0;
        let direction = [1.0, 0.0, 0.0];

        let result = augmented_eom_7d(
            &forces, "EARTH", 0.0, &state, t_max, isp, throttle, &direction,
        )
        .unwrap();

        // 验证推力加速度：a = T/m = 0.5 N / 1000 kg = 5e-4 m/s² = 5e-7 km/s²
        let expected_a = 0.5 / 1000.0 / 1000.0; // km/s²
        assert!((result[3] - expected_a).abs() < 1e-10);
    }
}
