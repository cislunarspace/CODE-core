//! Python force 元组解析为 `CompiledForce`（供 compiled/lowthrust/打靶绑定共用）。

use pyo3::prelude::*;
#[cfg(feature = "spice")]
use pyo3::types::PyTuple;

/// 解析单个 Python force 元组为 CompiledForce。
///
/// 元组格式（首元素是 type 标签）：
/// - `("gravity", c_flat, s_flat, mu, radius, degree, order, input_frame,
///     propagation_frame, body, propagation_origin, tide_mode, k_love_flat,
///     k_plus_flat_or_none)`
/// - `("third_body", body, mu)`
/// - `("indirect", body, mu)`
/// - `("srp", area, mass, cr, shadow_bodies_list)`
/// - `("uniform_accel", [aR, aT, aN], direction_frame)`，direction_frame 固定为 `"RTN"`
#[cfg(feature = "spice")]
pub(crate) fn parse_force_tuple(
    item: &Bound<'_, PyAny>,
) -> PyResult<e2m2e_forces::forces::compiled::CompiledForce> {
    use e2m2e_forces::forces::compiled::CompiledForce;
    use e2m2e_forces::forces::drag::DragAtmosphere;
    use e2m2e_forces::forces::gravity_field::TideMode;

    let tuple = item
        .downcast::<PyTuple>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("force must be a tuple"))?;
    let tag: String = tuple
        .get_item(0)?
        .extract()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("force tag must be a string"))?;

    match tag.as_str() {
        "point_mass" => {
            let mu: f64 = tuple.get_item(1)?.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("point_mass mu must be float")
            })?;
            Ok(CompiledForce::PointMass { mu })
        }
        "gravity" => {
            // 14 个元素
            let c_flat: Vec<f64> = tuple.get_item(1)?.extract()?;
            let s_flat: Vec<f64> = tuple.get_item(2)?.extract()?;
            let mu: f64 = tuple.get_item(3)?.extract()?;
            let radius: f64 = tuple.get_item(4)?.extract()?;
            let degree: usize = tuple.get_item(5)?.extract()?;
            let order: usize = tuple.get_item(6)?.extract()?;
            let input_frame: String = tuple.get_item(7)?.extract()?;
            let propagation_frame: String = tuple.get_item(8)?.extract()?;
            let body: String = tuple.get_item(9)?.extract()?;
            let propagation_origin: String = tuple.get_item(10)?.extract()?;
            let tide_mode_int: usize = tuple.get_item(11)?.extract()?;
            let k_love_flat: Vec<f64> = tuple.get_item(12)?.extract()?;
            let k_plus_flat_obj = tuple.get_item(13)?;
            let k_plus_flat: Option<Vec<f64>> = if k_plus_flat_obj.is_none() {
                None
            } else {
                Some(k_plus_flat_obj.extract()?)
            };
            let tide_mode = match tide_mode_int {
                0 => TideMode::None,
                1 => TideMode::Solid,
                2 => TideMode::SolidAndPole,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "tide_mode must be 0/1/2, got {}",
                        tide_mode_int
                    )))
                }
            };
            Ok(CompiledForce::GravityField {
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
            })
        }
        "third_body" => {
            let body: String = tuple.get_item(1)?.extract()?;
            let mu: f64 = tuple.get_item(2)?.extract()?;
            Ok(CompiledForce::ThirdBody { body, mu })
        }
        "indirect" => {
            let body: String = tuple.get_item(1)?.extract()?;
            let mu: f64 = tuple.get_item(2)?.extract()?;
            Ok(CompiledForce::IndirectTerm { body, mu })
        }
        "srp" => {
            let area: f64 = tuple.get_item(1)?.extract()?;
            let mass: f64 = tuple.get_item(2)?.extract()?;
            let cr: f64 = tuple.get_item(3)?.extract()?;
            let shadow_bodies: Vec<String> = tuple.get_item(4)?.extract()?;
            Ok(CompiledForce::SRP {
                area,
                mass,
                cr,
                shadow_bodies,
            })
        }
        "srp_variable_mass" => {
            let area: f64 = tuple.get_item(1)?.extract()?;
            let cr: f64 = tuple.get_item(2)?.extract()?;
            let shadow_bodies: Vec<String> = tuple.get_item(3)?.extract()?;
            Ok(CompiledForce::SRPVariableMass {
                area,
                cr,
                shadow_bodies,
            })
        }
        "ecom_srp" => {
            let dyb_vec: Vec<f64> = tuple.get_item(1)?.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("ecom_srp dyb must be a list of floats")
            })?;
            if dyb_vec.len() != 9 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ecom_srp dyb must have 9 elements, got {}",
                    dyb_vec.len()
                )));
            }
            let mut dyb = [0.0_f64; 9];
            dyb.copy_from_slice(&dyb_vec);
            let shadow_bodies: Vec<String> = tuple.get_item(2)?.extract()?;
            Ok(CompiledForce::EcomSrp { dyb, shadow_bodies })
        }
        "relativistic" => {
            // 元组格式：
            // ("relativistic", central_body, primary_body_or_none,
            //  mu_central, mu_primary_or_none,
            //  enable_schwarzschild, enable_lt, enable_de_sitter,
            //  angular_momentum_vector_or_none, body_radius_override_or_none, gamma)
            let central_body: String = tuple.get_item(1)?.extract()?;
            let primary_obj = tuple.get_item(2)?;
            let primary_body: Option<String> = if primary_obj.is_none() {
                None
            } else {
                Some(primary_obj.extract()?)
            };
            let mu_central: f64 = tuple.get_item(3)?.extract()?;
            let mu_primary_obj = tuple.get_item(4)?;
            let mu_primary: Option<f64> = if mu_primary_obj.is_none() {
                None
            } else {
                Some(mu_primary_obj.extract()?)
            };
            let enable_schwarzschild: bool = tuple.get_item(5)?.extract()?;
            let enable_lense_thirring: bool = tuple.get_item(6)?.extract()?;
            let enable_de_sitter: bool = tuple.get_item(7)?.extract()?;
            // angular_momentum_vector 可选（None 时自动 sxform 算）
            let j_obj = tuple.get_item(8)?;
            let angular_momentum_vector: Option<[f64; 3]> = if j_obj.is_none() {
                None
            } else {
                let v: Vec<f64> = j_obj.extract()?;
                if v.len() == 3 {
                    Some([v[0], v[1], v[2]])
                } else {
                    None
                }
            };
            let radius_obj = tuple.get_item(9)?;
            let body_radius_override: Option<f64> = if radius_obj.is_none() {
                None
            } else {
                Some(radius_obj.extract()?)
            };
            let gamma: f64 = tuple.get_item(10)?.extract()?;
            Ok(CompiledForce::Relativistic {
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
            })
        }
        "low_thrust" => {
            // 元组格式：("low_thrust", mass, thrust, t_start, t_end, direction,
            // direction_frame)。起止时间同时为 None 时表示常开。
            let mass: f64 = tuple.get_item(1)?.extract()?;
            let thrust: f64 = tuple.get_item(2)?.extract()?;
            let start_obj = tuple.get_item(3)?;
            let t_start: Option<f64> = if start_obj.is_none() {
                None
            } else {
                Some(start_obj.extract()?)
            };
            let end_obj = tuple.get_item(4)?;
            let t_end: Option<f64> = if end_obj.is_none() {
                None
            } else {
                Some(end_obj.extract()?)
            };
            let direction: Vec<f64> = tuple.get_item(5)?.extract()?;
            let frame_obj = tuple.get_item(6)?;
            let direction_frame: Option<String> = if frame_obj.is_none() {
                None
            } else {
                Some(frame_obj.extract()?)
            };
            if mass <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust mass must be positive",
                ));
            }
            if thrust < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust thrust must be non-negative",
                ));
            }
            if t_start.is_some() != t_end.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust pulse requires both t_start and t_end",
                ));
            }
            if let (Some(start), Some(end)) = (t_start, t_end) {
                if end < start {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "low_thrust t_end must be greater than or equal to t_start",
                    ));
                }
            }
            if direction.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust direction must have 3 elements",
                ));
            }
            if !matches!(
                direction_frame.as_deref(),
                None | Some("VNB") | Some("LVLH")
            ) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust direction_frame must be None, 'VNB', or 'LVLH'",
                ));
            }
            Ok(CompiledForce::LowThrust {
                mass,
                thrust,
                t_start,
                t_end,
                direction: [direction[0], direction[1], direction[2]],
                direction_frame,
            })
        }
        "uniform_accel" => {
            // 元组格式：("uniform_accel", [aR, aT, aN], direction_frame)。
            let acceleration: Vec<f64> = tuple.get_item(1)?.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "uniform_accel acceleration must be a list of floats",
                )
            })?;
            let direction_frame: String = tuple.get_item(2)?.extract()?;
            if acceleration.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "uniform_accel acceleration must have 3 elements",
                ));
            }
            if direction_frame != "RTN" {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "uniform_accel direction_frame must be 'RTN'",
                ));
            }
            Ok(CompiledForce::UniformAcceleration {
                acceleration_rtn: [acceleration[0], acceleration[1], acceleration[2]],
                direction_frame,
            })
        }
        "drag" => {
            // 元组格式：("drag", area, mass, cd, propagation_frame, f107, ap)
            let area: f64 = tuple.get_item(1)?.extract()?;
            let mass: f64 = tuple.get_item(2)?.extract()?;
            let cd: f64 = tuple.get_item(3)?.extract()?;
            let propagation_frame: String = tuple.get_item(4)?.extract()?;
            let f107: f64 = tuple.get_item(5)?.extract()?;
            let ap: f64 = tuple.get_item(6)?.extract()?;
            Ok(CompiledForce::Drag {
                area,
                mass,
                cd,
                atmosphere: DragAtmosphere::Exponential { f107, ap },
                propagation_frame,
            })
        }
        "drag_nrlmsise00" => {
            // 元组格式：
            // ("drag_nrlmsise00", area, mass, cd, propagation_frame, f107_daily, f107_avg, ap[7])
            let area: f64 = tuple.get_item(1)?.extract()?;
            let mass: f64 = tuple.get_item(2)?.extract()?;
            let cd: f64 = tuple.get_item(3)?.extract()?;
            let propagation_frame: String = tuple.get_item(4)?.extract()?;
            let f107_daily: f64 = tuple.get_item(5)?.extract()?;
            let f107_avg: f64 = tuple.get_item(6)?.extract()?;
            let ap_list: Vec<f64> = tuple.get_item(7)?.extract()?;
            let ap: [f64; 7] = ap_list.as_slice().try_into().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "drag_nrlmsise00 atmosphere ap must have 7 elements, got {}",
                    ap_list.len()
                ))
            })?;
            Ok(CompiledForce::Drag {
                area,
                mass,
                cd,
                atmosphere: DragAtmosphere::Nrlmsise00 {
                    f107_daily,
                    f107_avg,
                    ap,
                },
                propagation_frame,
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown force tag {:?}",
            tag
        ))),
    }
}
