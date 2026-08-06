//! 编译型力模型：把所有 force 配置一次性序列化给 Rust，RK 内循环全程 Rust。
//!
//! 设计：Python 侧把每个 force 转成元组，传给 `propagate_compiled`。
//! Rust 侧 dispatch 到对应的 force 实现。每步 RK 不再跨界回 Python。
//!
//! 与 `rk_step + Python callback` 模式相比，本模块消除 30 万次 GIL 跨界。

use crate::forces::drag;
use crate::forces::ecom;
use crate::forces::gravity_field::{self, GravityFieldContext, TideConfig, TideMode};
use crate::forces::relativistic;
use crate::forces::srp;
use e2m2e_spice::spk_accel;

/// 编译后的 force（enum，dispatch 用 match）。
///
/// 每个 variant 持有该 force 的全部配置。Python 侧用元组序列化（见
/// `force_from_tuple`）。
pub enum CompiledForce {
    PointMass {
        mu: f64,
    },
    GravityField {
        c_flat: Vec<f64>,
        s_flat: Vec<f64>,
        mu: f64,
        radius: f64,
        degree: usize,
        order: usize,
        input_frame: String,
        propagation_frame: String,
        body: String,
        propagation_origin: String,
        tide_mode: TideMode,
        k_love_flat: Vec<f64>,
        k_plus_flat: Option<Vec<f64>>,
    },
    ThirdBody {
        body: String,
        mu: f64,
    },
    IndirectTerm {
        body: String,
        mu: f64,
    },
    SRP {
        area: f64,
        mass: f64,
        cr: f64,
        shadow_bodies: Vec<String>,
    },
    EcomSrp {
        dyb: [f64; 9],
        shadow_bodies: Vec<String>,
    },
    Relativistic {
        central_body: String,
        primary_body: Option<String>,
        mu_central: f64,
        mu_primary: Option<f64>,
        enable_schwarzschild: bool,
        enable_lense_thirring: bool,
        enable_de_sitter: bool,
        angular_momentum_vector: Option<[f64; 3]>,
        body_radius_override: Option<f64>,
        gamma: f64,
    },
    /// 小推力（电推进）模型。
    ///
    /// 推力加速度 = (T_max / m) * u * α
    /// 其中 T_max 为最大推力（N），m 为航天器质量（kg），u ∈ [0, 1] 为推力幅值，
    /// α 为单位推力方向向量。
    LowThrust {
        /// 最大推力（N）
        t_max: f64,
        /// 比冲（s）
        isp: f64,
        /// 推力幅值 u ∈ [0, 1]（同伦参数）
        throttle: f64,
        /// 推力方向单位向量（惯性系）
        direction: [f64; 3],
    },
    /// 大气阻力力模型。
    ///
    /// ITRF93 系内计算阻力加速度，含帧旋转变换。
    Drag {
        area: f64,
        mass: f64,
        cd: f64,
        /// 传播系 frame（通常 "J2000"）
        propagation_frame: String,
    },
}

impl CompiledForce {
    /// 计算该 force 在 (et, state) 下的加速度。
    ///
    /// `observer` 是传播系 origin 天体名（如 "EARTH"）。
    /// 错误统一成 String（cspice Error 与 SpiceFfiError 之间不兼容）。
    pub fn acceleration(
        &self,
        et: f64,
        state: &[f64; 6],
        observer: &str,
    ) -> Result<[f64; 3], String> {
        match self {
            Self::PointMass { mu } => {
                let r = [state[0], state[1], state[2]];
                let r_norm_sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                let r_norm = r_norm_sq.sqrt().max(1e-6);
                let inv_r3 = 1.0 / (r_norm * r_norm * r_norm);
                Ok([
                    -mu * r[0] * inv_r3,
                    -mu * r[1] * inv_r3,
                    -mu * r[2] * inv_r3,
                ])
            }
            Self::GravityField {
                c_flat,
                s_flat,
                mu,
                radius,
                degree,
                order,
                input_frame,
                propagation_frame,
                body,
                propagation_origin,
                tide_mode,
                k_love_flat,
                k_plus_flat,
            } => {
                let r_sc = [state[0], state[1], state[2]];
                let tide = TideConfig {
                    mode: tide_mode.clone(),
                    k_love_flat: k_love_flat.clone(),
                    k_plus_flat: k_plus_flat.clone(),
                };
                gravity_field::gravity_field_acceleration(
                    et,
                    &r_sc,
                    c_flat,
                    s_flat,
                    *mu,
                    *radius,
                    *degree,
                    *order,
                    input_frame,
                    propagation_frame,
                    body,
                    propagation_origin,
                    &tide,
                )
                .map_err(|e| format!("{:?}", e))
            }
            Self::ThirdBody { body, mu } => {
                let sc_pos = [state[0], state[1], state[2]];
                spk_accel::third_body_acceleration(et, body, observer, &sc_pos, *mu, 1e-6)
                    .map_err(|e| format!("{:?}", e))
            }
            Self::IndirectTerm { body, mu } => {
                spk_accel::indirect_term_acceleration(et, body, observer, *mu, 1e-6)
                    .map_err(|e| format!("{:?}", e))
            }
            Self::SRP {
                area,
                mass,
                cr,
                shadow_bodies,
            } => {
                let sc_pos = [state[0], state[1], state[2]];
                srp::srp_acceleration(et, &sc_pos, *area, *mass, *cr, shadow_bodies, observer)
                    .map_err(|e| format!("{:?}", e))
            }
            Self::EcomSrp { dyb, shadow_bodies } => {
                let sc_pos = [state[0], state[1], state[2]];
                ecom::ecom_acceleration(et, &sc_pos, dyb, shadow_bodies, observer)
                    .map_err(|e| format!("{:?}", e))
            }
            Self::Relativistic {
                central_body,
                primary_body,
                mu_central,
                mu_primary,
                enable_schwarzschild,
                enable_lense_thirring,
                enable_de_sitter,
                angular_momentum_vector,
                body_radius_override,
                gamma,
            } => {
                let state6 = [state[0], state[1], state[2], state[3], state[4], state[5]];
                relativistic::relativistic_acceleration(
                    et,
                    &state6,
                    central_body,
                    primary_body.as_deref(),
                    *mu_central,
                    *mu_primary,
                    *enable_schwarzschild,
                    *enable_lense_thirring,
                    *enable_de_sitter,
                    angular_momentum_vector.as_ref(),
                    *body_radius_override,
                    *gamma,
                )
                .map_err(|e| format!("{:?}", e))
            }
            Self::LowThrust {
                t_max,
                isp: _,
                throttle,
                direction,
            } => {
                // 小推力加速度 = (T_max / m) * u * α
                // 注意：当前实现假设质量恒定，实际应用中需要扩展状态向量包含质量
                let mass = 1000.0; // kg，默认质量，实际应从状态或参数获取
                                   // T_max 单位为 N (kg·m/s²)，质量单位为 kg，加速度单位为 m/s²
                                   // 需要转换为 km/s²（除以 1000）
                let accel_mag_m_s2 = (*t_max / mass) * throttle;
                let accel_mag_km_s2 = accel_mag_m_s2 / 1000.0;
                Ok([
                    accel_mag_km_s2 * direction[0],
                    accel_mag_km_s2 * direction[1],
                    accel_mag_km_s2 * direction[2],
                ])
            }
            Self::Drag {
                area,
                mass,
                cd,
                propagation_frame,
            } => drag::drag_accel(et, state, *area, *mass, *cd, propagation_frame)
                .map_err(|e| format!("{:?}", e)),
        }
    }
}

/// 计算所有 force 的总加速度。
pub fn compute_total_acceleration(
    forces: &[CompiledForce],
    et: f64,
    state: &[f64; 6],
    observer: &str,
) -> Result<[f64; 3], String> {
    let mut total = [0.0_f64; 3];
    for force in forces {
        let a = force.acceleration(et, state, observer)?;
        total[0] += a[0];
        total[1] += a[1];
        total[2] += a[2];
    }
    Ok(total)
}

/// 单个 force 是否支持解析/Rust FD 雅可比。
pub fn supports_jacobian(force: &CompiledForce) -> bool {
    matches!(
        force,
        CompiledForce::PointMass { .. }
            | CompiledForce::GravityField { .. }
            | CompiledForce::ThirdBody { .. }
            | CompiledForce::IndirectTerm { .. }
            | CompiledForce::SRP { .. }
            | CompiledForce::EcomSrp { .. }
            | CompiledForce::Drag { .. }
    )
}

/// 计算单个 force 的加速度 + 雅可比（3×3）。
///
/// - PointMass：解析 `-μ(I/r³ − 3rrᵀ/r⁵)`
/// - GravityField：用 `GravityFieldContext` 做 body-fixed 有限差分
/// - ThirdBody：spk_accel 解析
/// - IndirectTerm：零矩阵（不依赖航天器位置）
/// - SRP/Relativistic：返回 Err
pub fn acceleration_and_jacobian(
    force: &CompiledForce,
    et: f64,
    state: &[f64; 6],
    observer: &str,
) -> Result<([f64; 3], [[f64; 3]; 3], [[f64; 3]; 3]), String> {
    match force {
        CompiledForce::PointMass { mu } => {
            let r = [state[0], state[1], state[2]];
            let r_norm_sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let r_norm = r_norm_sq.sqrt().max(1e-6);
            let inv_r3 = 1.0 / (r_norm * r_norm * r_norm);
            let inv_r5 = inv_r3 / (r_norm * r_norm);
            let acc = [
                -mu * r[0] * inv_r3,
                -mu * r[1] * inv_r3,
                -mu * r[2] * inv_r3,
            ];
            let mut jac = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    jac[i][j] = -mu * (delta * inv_r3 - 3.0 * r[i] * r[j] * inv_r5);
                }
            }
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc, jac, dadv))
        }
        CompiledForce::GravityField {
            c_flat,
            s_flat,
            mu,
            radius,
            degree,
            order,
            input_frame,
            propagation_frame,
            body,
            propagation_origin,
            tide_mode,
            k_love_flat,
            k_plus_flat,
        } => {
            // 用 GravityFieldContext 做 body-fixed FD。
            // build() 时一次性查 SPICE（旋转矩阵 + 原点偏移 + 潮汐系数），
            // jacobian_fd() 在 body-fixed 系做 FD，零额外 SPICE 调用。
            let tide = TideConfig {
                mode: tide_mode.clone(),
                k_love_flat: k_love_flat.clone(),
                k_plus_flat: k_plus_flat.clone(),
            };
            let ctx = GravityFieldContext::build(
                et,
                c_flat,
                s_flat,
                *mu,
                *radius,
                *degree,
                *order,
                input_frame,
                propagation_frame,
                body,
                propagation_origin,
                &tide,
            )
            .map_err(|e| format!("{:?}", e))?;
            let r_sc = [state[0], state[1], state[2]];
            let acc = ctx.accel(&r_sc).map_err(|e| format!("{:?}", e))?;
            let jac = ctx.jacobian_fd(&r_sc).map_err(|e| format!("{:?}", e))?;
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc, jac, dadv))
        }
        CompiledForce::ThirdBody { body, mu } => {
            let sc_pos = [state[0], state[1], state[2]];
            let (acc, jac) = spk_accel::third_body_acceleration_and_jacobian(
                et, body, observer, &sc_pos, *mu, 1e-6,
            )
            .map_err(|e| format!("{:?}", e))?;
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc, jac, dadv))
        }
        CompiledForce::IndirectTerm { .. } => {
            let acc = force.acceleration(et, state, observer)?;
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc, [[0.0; 3]; 3], dadv))
        }
        CompiledForce::SRP { .. } => {
            // 数值差分（与 GravityField::jacobian_fd 同模式，步长 sqrt(eps)*|r|）。
            // SRP 加速度含阴影几何（太阳/遮挡体位置随 et 变化但本步内固定），
            // 差分自动包含光照份额对位置的贡献；阴影边界处不连续引入的
            // 差分误差只影响边界点，对 STM 积分可接受。
            let r_norm = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
            let h = (f64::EPSILON.sqrt() * r_norm).max(1e-6);
            let acc0 = force.acceleration(et, state, observer)?;
            let mut jac = [[0.0_f64; 3]; 3];
            for dim in 0..3 {
                let mut state_plus = *state;
                let mut state_minus = *state;
                state_plus[dim] += h;
                state_minus[dim] -= h;
                let a_plus = force.acceleration(et, &state_plus, observer)?;
                let a_minus = force.acceleration(et, &state_minus, observer)?;
                for i in 0..3 {
                    jac[i][dim] = (a_plus[i] - a_minus[i]) / (2.0 * h);
                }
            }
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc0, jac, dadv))
        }
        CompiledForce::EcomSrp { .. } => {
            // 数值差分（与 SRP 同模式）。
            let r_norm = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
            let h = (f64::EPSILON.sqrt() * r_norm).max(1e-6);
            let acc0 = force.acceleration(et, state, observer)?;
            let mut jac = [[0.0_f64; 3]; 3];
            for dim in 0..3 {
                let mut state_plus = *state;
                let mut state_minus = *state;
                state_plus[dim] += h;
                state_minus[dim] -= h;
                let a_plus = force.acceleration(et, &state_plus, observer)?;
                let a_minus = force.acceleration(et, &state_minus, observer)?;
                for i in 0..3 {
                    jac[i][dim] = (a_plus[i] - a_minus[i]) / (2.0 * h);
                }
            }
            let dadv = [[0.0_f64; 3]; 3];
            Ok((acc0, jac, dadv))
        }
        CompiledForce::Drag {
            area,
            mass,
            cd,
            propagation_frame,
        } => {
            let result =
                drag::drag_accel_and_jacobian(et, state, *area, *mass, *cd, propagation_frame)
                    .map_err(|e| format!("{:?}", e))?;
            Ok((result.acc, result.jac_da_dr, result.jac_da_dv))
        }
        _ => Err("Jacobian not supported for this force type".to_string()),
    }
}

/// 计算所有 force 的总加速度 + 总雅可比（逐力叠加）。
pub fn compute_total_acceleration_and_jacobian(
    forces: &[CompiledForce],
    et: f64,
    state: &[f64; 6],
    observer: &str,
) -> Result<([f64; 3], [[f64; 3]; 3], [[f64; 3]; 3]), String> {
    let mut total_acc = [0.0_f64; 3];
    let mut total_jac = [[0.0_f64; 3]; 3];
    let mut total_dadv = [[0.0_f64; 3]; 3];
    for force in forces {
        let (acc, jac, dadv) = acceleration_and_jacobian(force, et, state, observer)?;
        for i in 0..3 {
            total_acc[i] += acc[i];
            for j in 0..3 {
                total_jac[i][j] += jac[i][j];
                total_dadv[i][j] += dadv[i][j];
            }
        }
    }
    Ok((total_acc, total_jac, total_dadv))
}
