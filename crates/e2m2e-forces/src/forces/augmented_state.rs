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

/// 推力配置参数。
///
/// 将推力相关的 4 个独立参数打包，避免函数签名参数过多。
#[derive(Debug, Clone, Copy)]
pub struct ThrustParams {
    /// 最大推力（N）
    pub t_max: f64,
    /// 比冲（s）
    pub isp: f64,
    /// 推力幅值 u0 ∈ [0, 1]
    pub throttle: f64,
    /// 推力方向单位向量（惯性系）
    pub direction: [f64; 3],
}

/// 7D 增广动力学方程。
///
/// # 参数
/// - `forces`: 力模型列表（重力等）
/// - `observer`: 传播系 origin 天体名（如 "EARTH"）
/// - `et`: 历元时刻（SPICE et 秒）
/// - `state`: 7D 状态 [x, y, z, vx, vy, vz, m]
/// - `thrust`: 推力配置
///
/// # 返回
/// 7D 导数 [vx, vy, vz, ax, ay, az, mdot]
pub fn augmented_eom_7d(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &AugmentedState7,
    thrust: &ThrustParams,
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
    let a_thrust_mag_m_s2 = (thrust.t_max / m) * thrust.throttle;
    let a_thrust_mag_km_s2 = a_thrust_mag_m_s2 / 1000.0;
    let a_thrust = [
        a_thrust_mag_km_s2 * thrust.direction[0],
        a_thrust_mag_km_s2 * thrust.direction[1],
        a_thrust_mag_km_s2 * thrust.direction[2],
    ];

    // 3. 计算质量流率
    // mdot = -u0 * T_max / (Isp * g0)
    // 单位：kg/s
    let g0 = 9.81; // m/s²
    let mdot = -thrust.throttle * thrust.t_max / (thrust.isp * g0);

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
/// - `thrust`: 推力配置
///
/// # 返回
/// 43D 导数
pub fn augmented_eom_7d_with_stm(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &AugmentedState42,
    thrust: &ThrustParams,
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
    let a_thrust_mag_m_s2 = (thrust.t_max / m) * thrust.throttle;
    let a_thrust_mag_km_s2 = a_thrust_mag_m_s2 / 1000.0;
    let a_thrust = [
        a_thrust_mag_km_s2 * thrust.direction[0],
        a_thrust_mag_km_s2 * thrust.direction[1],
        a_thrust_mag_km_s2 * thrust.direction[2],
    ];

    // 3. 计算质量流率
    let g0 = 9.81;
    let mdot = -thrust.throttle * thrust.t_max / (thrust.isp * g0);

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

/// 单位推力方向的角度参数化（Du 2024 式 5）。
///
/// `α(θ₁,θ₂) = [cosθ₁cosθ₂, sinθ₁cosθ₂, sinθ₂]`，球面单位向量。
fn direction_from_angles(theta1: f64, theta2: f64) -> [f64; 3] {
    [
        theta1.cos() * theta2.cos(),
        theta1.sin() * theta2.cos(),
        theta2.sin(),
    ]
}

/// 7D 受控 + 灵敏度 EOM（64D 增广状态）。
///
/// 状态布局 `[r(3), v(3), m(1), Φ(36), S(21)]` = 64D：
/// - x₇ = [r, v, m]
/// - Φ 是 6×6 状态对初值 STM（[r,v] 对 [r,v]，链式接龙用）
/// - S 是 7×3 状态对控制参数 (throttle, θ₁, θ₂) 的灵敏度
///
/// 控制用角度参数化：方向 `α(θ₁,θ₂)`，与 `ThrustParams.throttle` 一起作为
/// 控制参数。S(0)=0、Φ(0)=I。
///
/// 灵敏度方程 `dS/dt = A·S + B`，其中 A 是 7×7 动力学雅可比、B 是 7×3 控制
/// 雅可比。详见 `docs/plans/lowthrust-analytic-jacobian-prd.md`。
pub fn augmented_eom_7d_with_sensitivity(
    forces: &[CompiledForce],
    observer: &str,
    et: f64,
    state: &[f64; 64],
    thrust: &ThrustParams,
    theta1: f64,
    theta2: f64,
) -> Result<[f64; 64], String> {
    // 提取状态分量
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let m = state[6];
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&state[7..43]);
    let mut s = [[0.0_f64; 3]; 7]; // 7×3 灵敏度
    for i in 0..7 {
        for j in 0..3 {
            s[i][j] = state[43 + i * 3 + j];
        }
    }

    let state6 = [r[0], r[1], r[2], v[0], v[1], v[2]];

    // 1. 重力加速度 + 雅可比 ∂a_grav/∂r（3×3）
    let (a_gravity, jac_da_dr) =
        super::compiled::compute_total_acceleration_and_jacobian(forces, et, &state6, observer)?;

    // 2. 推力加速度（与 augmented_eom_7d 一致）
    let t_max = thrust.t_max;
    let isp = thrust.isp;
    let u = thrust.throttle; // 油门 ∈ [0,1]
    let g0 = 9.81;
    let alpha = direction_from_angles(theta1, theta2);
    let a_thrust_mag_km_s2 = (t_max / m) * u / 1000.0;
    let a_thrust = [
        a_thrust_mag_km_s2 * alpha[0],
        a_thrust_mag_km_s2 * alpha[1],
        a_thrust_mag_km_s2 * alpha[2],
    ];
    let mdot = -u * t_max / (isp * g0);

    // 3. 组装 7×7 动力学雅可比 A（行优先）
    //    行序 [r(3), v(3), m(1)]，列序 [r(3), v(3), m(1)]
    //    ṙ=v → A[r,v]=I, 其余 r 行 0
    //    v̇=a → A[v,r]=∂a/∂r, A[v,v]=0, A[v,m]=∂a/∂m
    //    ṁ → A[m,*]=0（ṁ 只依赖 throttle）
    // ∂a_thrust/∂m = -u·T_max·α/m²（km/s²，注意 T_max 是 N→m/s²，/1000 转 km）
    let da_dm = [-(u * t_max / (m * m)) / 1000.0 * alpha[0],
                 -(u * t_max / (m * m)) / 1000.0 * alpha[1],
                 -(u * t_max / (m * m)) / 1000.0 * alpha[2]];
    let mut a_mat = [[0.0_f64; 7]; 7];
    for i in 0..3 {
        a_mat[i][3 + i] = 1.0; // A[r_i, v_i] = 1
        for j in 0..3 {
            a_mat[3 + i][j] = jac_da_dr[i][j]; // A[v_i, r_j] = ∂a/∂r
        }
        a_mat[3 + i][6] = da_dm[i]; // A[v_i, m] = ∂a/∂m
    }

    // 4. 组装 7×3 控制雅可比 B（行优先），列序 [throttle, θ₁, θ₂]
    // ∂v̇/∂throttle = T_max·α/m（/1000 转 km/s²）
    let dv_dthr = [(t_max / m) / 1000.0 * alpha[0],
                   (t_max / m) / 1000.0 * alpha[1],
                   (t_max / m) / 1000.0 * alpha[2]];
    // ∂α/∂θ₁ = [-sinθ₁cosθ₂, cosθ₁cosθ₂, 0]
    let dalpha_dt1 = [-theta1.sin() * theta2.cos(),
                      theta1.cos() * theta2.cos(),
                      0.0];
    // ∂α/∂θ₂ = [-cosθ₁sinθ₂, -sinθ₁sinθ₂, cosθ₂]
    let dalpha_dt2 = [-theta1.cos() * theta2.sin(),
                      -theta1.sin() * theta2.sin(),
                      theta2.cos()];
    let coeff = (t_max * u / m) / 1000.0; // km/s²
    let dv_dt1 = [coeff * dalpha_dt1[0], coeff * dalpha_dt1[1], coeff * dalpha_dt1[2]];
    let dv_dt2 = [coeff * dalpha_dt2[0], coeff * dalpha_dt2[1], coeff * dalpha_dt2[2]];
    // ∂ṁ/∂throttle = -T_max/(Isp·g₀)；∂ṁ/∂θ₁=∂ṁ/∂θ₂=0
    let dmdot_dthr = -t_max / (isp * g0);

    let mut b_mat = [[0.0_f64; 3]; 7]; // 行 [r(3),v(3),m]，列 [thr,θ₁,θ₂]
    for i in 0..3 {
        b_mat[3 + i][0] = dv_dthr[i]; // v 行 × throttle 列
        b_mat[3 + i][1] = dv_dt1[i]; // v 行 × θ₁ 列
        b_mat[3 + i][2] = dv_dt2[i]; // v 行 × θ₂ 列
    }
    b_mat[6][0] = dmdot_dthr; // m 行 × throttle 列

    // 5. dS/dt = A·S + B（7×3）
    let mut ds = [[0.0_f64; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            let mut a_s = 0.0;
            for k in 0..7 {
                a_s += a_mat[i][k] * s[k][j];
            }
            ds[i][j] = a_s + b_mat[i][j];
        }
    }

    // 6. dΦ/dt = ∂f[r,v]/∂[r,v] · Φ（6×6，质量不进 Φ）
    let dstm = super::nbody_stm::stm_derivative(&stm, &jac_da_dr);

    // 7. 组装 64D 导数
    let mut result = [0.0_f64; 64];
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    result[3] = a_gravity[0] + a_thrust[0];
    result[4] = a_gravity[1] + a_thrust[1];
    result[5] = a_gravity[2] + a_thrust[2];
    result[6] = mdot;
    result[7..43].copy_from_slice(&dstm);
    for i in 0..7 {
        for j in 0..3 {
            result[43 + i * 3 + j] = ds[i][j];
        }
    }
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
            1500.0, // mass (kg)
        ];

        let thrust = ThrustParams {
            t_max: 1.0,  // N
            isp: 3000.0, // s
            throttle: 0.8,
            direction: [1.0, 0.0, 0.0],
        };

        let result = augmented_eom_7d(&forces, "EARTH", 0.0, &state, &thrust).unwrap();

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
            7000.0, 0.0, 0.0, 0.0, 0.0, 0.0,    // 静止
            1000.0, // kg
        ];

        let thrust = ThrustParams {
            t_max: 0.5, // N
            isp: 3000.0,
            throttle: 1.0,
            direction: [1.0, 0.0, 0.0],
        };

        let result = augmented_eom_7d(&forces, "EARTH", 0.0, &state, &thrust).unwrap();

        // 验证推力加速度：a = T/m = 0.5 N / 1000 kg = 5e-4 m/s² = 5e-7 km/s²
        let expected_a = 0.5 / 1000.0 / 1000.0; // km/s²
        assert!((result[3] - expected_a).abs() < 1e-10);
    }
}
