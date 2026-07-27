//! 编译型力模型：把所有 force 配置一次性序列化给 Rust，RK 内循环全程 Rust。
//!
//! 设计：Python 侧把每个 force 转成元组，传给 `propagate_compiled`。
//! Rust 侧 dispatch 到对应的 force 实现。每步 RK 不再跨界回 Python。
//!
//! 与 `rk_step + Python callback` 模式相比，本模块消除 30 万次 GIL 跨界。

use crate::forces::gravity_field::{self, TideConfig, TideMode};
use crate::forces::relativistic;
use crate::forces::srp;
use e2m2e_spice::spk_accel;

/// 编译后的 force（enum，dispatch 用 match）。
///
/// 每个 variant 持有该 force 的全部配置。Python 侧用元组序列化（见
/// `force_from_tuple`）。
pub enum CompiledForce {
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
