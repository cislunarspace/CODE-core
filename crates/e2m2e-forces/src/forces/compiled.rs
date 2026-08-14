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

/// 加速度 + 雅可比（∂a/∂r + ∂a/∂v）三元组的返回类型：纯状态/STM 传播共用。
pub type AccelJacobiResult = Result<([f64; 3], [[f64; 3]; 3], [[f64; 3]; 3]), String>;

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
    /// 恒质量连续推力模型。
    ///
    /// 推力加速度 = (T / m) * α。推力可常开，也可仅在时段
    /// ``(t_start, t_end)`` 内开启；端点处关机以避免 RK stage 把单点开关
    /// 误算成有限冲量。方向在惯性、VNB 或 LVLH 系中给出。
    LowThrust {
        /// 航天器恒定质量（kg）
        mass: f64,
        /// 推力幅值（N）
        thrust: f64,
        /// 开机时刻（SPICE et 秒）；None 表示常开
        t_start: Option<f64>,
        /// 关机时刻（SPICE et 秒）；None 表示常开
        t_end: Option<f64>,
        /// 推力方向（由 direction_frame 解释）
        direction: [f64; 3],
        /// None / "VNB" / "LVLH"
        direction_frame: Option<String>,
    },
    /// 大气阻力力模型。
    ///
    /// ITRF93 系内计算阻力加速度，含帧旋转变换。
    Drag {
        area: f64,
        mass: f64,
        cd: f64,
        /// F10.7 太阳射电通量（sfu），来自 `ExponentialAtmosphere.f107`。
        f107: f64,
        /// Ap 地磁指数，来自 `ExponentialAtmosphere.ap`。
        ap: f64,
        /// 传播系 frame（通常 "J2000"）
        propagation_frame: String,
    },
}

fn resolve_thrust_direction(
    direction: &[f64; 3],
    direction_frame: Option<&str>,
    state: &[f64; 6],
) -> Result<[f64; 3], String> {
    let norm = |v: [f64; 3]| -> f64 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() };
    let scale =
        |v: [f64; 3], factor: f64| -> [f64; 3] { [v[0] * factor, v[1] * factor, v[2] * factor] };
    let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let local_to_inertial = match direction_frame {
        None => *direction,
        Some("VNB") => {
            let v_norm = norm(v);
            let h = cross(r, v);
            let h_norm = norm(h);
            if v_norm < 1e-12 {
                return Err("VNB frame requires non-zero velocity".to_string());
            }
            if h_norm < 1e-12 {
                return Err("VNB frame requires non-zero angular momentum".to_string());
            }
            let v_hat = scale(v, 1.0 / v_norm);
            let n_hat = scale(h, 1.0 / h_norm);
            let b_hat = cross(v_hat, n_hat);
            [
                direction[0] * v_hat[0] + direction[1] * n_hat[0] + direction[2] * b_hat[0],
                direction[0] * v_hat[1] + direction[1] * n_hat[1] + direction[2] * b_hat[1],
                direction[0] * v_hat[2] + direction[1] * n_hat[2] + direction[2] * b_hat[2],
            ]
        }
        Some("LVLH") => {
            let r_norm = norm(r);
            let v_norm = norm(v);
            let h = cross(r, v);
            let h_norm = norm(h);
            if r_norm < 1e-12 {
                return Err("LVLH frame requires non-zero position".to_string());
            }
            if v_norm < 1e-12 {
                return Err("LVLH frame requires non-zero velocity".to_string());
            }
            let r_hat = scale(r, 1.0 / r_norm);
            let v_hat = scale(v, 1.0 / v_norm);
            if h_norm < 1e-12 {
                [
                    direction[0] * r_hat[0] + direction[1] * v_hat[0],
                    direction[0] * r_hat[1] + direction[1] * v_hat[1],
                    direction[0] * r_hat[2] + direction[1] * v_hat[2],
                ]
            } else {
                let n_hat = scale(h, 1.0 / h_norm);
                // 与 LVLHAxes 对齐：沿迹轴垂直于径向轴，而非一般椭圆轨道
                // 中含径向分量的速度单位向量。
                let along_track = cross(n_hat, r_hat);
                [
                    direction[0] * r_hat[0]
                        + direction[1] * along_track[0]
                        + direction[2] * n_hat[0],
                    direction[0] * r_hat[1]
                        + direction[1] * along_track[1]
                        + direction[2] * n_hat[1],
                    direction[0] * r_hat[2]
                        + direction[1] * along_track[2]
                        + direction[2] * n_hat[2],
                ]
            }
        }
        Some(frame) => return Err(format!("unsupported thrust direction frame {frame:?}")),
    };
    let direction_norm = norm(local_to_inertial);
    if direction_norm < 1e-15 {
        return Err("thrust direction must be non-zero".to_string());
    }
    Ok(scale(local_to_inertial, 1.0 / direction_norm))
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
                mass,
                thrust,
                t_start,
                t_end,
                direction,
                direction_frame,
            } => {
                if *thrust == 0.0 {
                    return Ok([0.0; 3]);
                }
                if let (Some(start), Some(end)) = (t_start, t_end) {
                    if et <= *start || et >= *end {
                        return Ok([0.0; 3]);
                    }
                }
                let direction =
                    resolve_thrust_direction(direction, direction_frame.as_deref(), state)?;
                // N / kg = m/s²，传播状态使用 km/s，故除以 1000。
                let accel_mag_km_s2 = thrust / mass / 1000.0;
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
                f107,
                ap,
                propagation_frame,
            } => drag::drag_accel(et, state, *area, *mass, *cd, *f107, *ap, propagation_frame)
                .map_err(|e| format!("{:?}", e)),
        }
    }
}

/// 返回 `(t, tf]` 内最早的编译力不连续时刻。
///
/// 恒质量 pulse 推力在开机/关机边界处不连续。积分器必须将其作为步长终点，
/// 否则一个 RK 步跨越边界会使累计冲量依赖输出网格。
pub fn next_force_discontinuity(forces: &[CompiledForce], t: f64, tf: f64) -> Option<f64> {
    forces
        .iter()
        .filter_map(|force| match force {
            CompiledForce::LowThrust {
                t_start: Some(start),
                t_end: Some(end),
                ..
            } => Some([*start, *end]),
            _ => None,
        })
        .flatten()
        .filter(|boundary| *boundary > t && *boundary <= tf)
        .min_by(|left, right| left.total_cmp(right))
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
            | CompiledForce::LowThrust { .. }
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
) -> AccelJacobiResult {
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
        CompiledForce::LowThrust { .. } => {
            // VNB/LVLH 方向取决于状态，统一在 Rust 内中心差分，避免 Python
            // 回调并使 STM 与普通 RHS 使用同一方向定义。
            let r_norm = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
            let v_norm = (state[3] * state[3] + state[4] * state[4] + state[5] * state[5]).sqrt();
            let h_r = (f64::EPSILON.sqrt() * r_norm).max(1e-6);
            let h_v = (f64::EPSILON.sqrt() * v_norm).max(1e-9);
            let acc0 = force.acceleration(et, state, observer)?;
            let mut jac = [[0.0_f64; 3]; 3];
            let mut dadv = [[0.0_f64; 3]; 3];
            for dim in 0..3 {
                let mut plus = *state;
                let mut minus = *state;
                plus[dim] += h_r;
                minus[dim] -= h_r;
                let a_plus = force.acceleration(et, &plus, observer)?;
                let a_minus = force.acceleration(et, &minus, observer)?;
                for row in 0..3 {
                    jac[row][dim] = (a_plus[row] - a_minus[row]) / (2.0 * h_r);
                }
                let mut plus_v = *state;
                let mut minus_v = *state;
                plus_v[dim + 3] += h_v;
                minus_v[dim + 3] -= h_v;
                let a_plus_v = force.acceleration(et, &plus_v, observer)?;
                let a_minus_v = force.acceleration(et, &minus_v, observer)?;
                for row in 0..3 {
                    dadv[row][dim] = (a_plus_v[row] - a_minus_v[row]) / (2.0 * h_v);
                }
            }
            Ok((acc0, jac, dadv))
        }
        CompiledForce::Drag {
            area,
            mass,
            cd,
            f107,
            ap,
            propagation_frame,
        } => {
            let result = drag::drag_accel_and_jacobian(
                et,
                state,
                *area,
                *mass,
                *cd,
                *f107,
                *ap,
                propagation_frame,
            )
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
) -> AccelJacobiResult {
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
