//! 七类 CR3BP 轨道族的统一 Rust 生成模块。
//!
//! 该模块的接口是一次请求、一次返回；种子、修正、延拓、步长、筛选和
//! 结构化终止全部藏在接口后。PyO3 适配器只把已校验参数翻成内部 `Spec`。

mod axial;
mod common;
mod halo;
mod lissajous;
mod nrho;
mod triangular;
mod types;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use types::{Context, Outcome, Spec};

fn required<T>(value: Option<T>, name: &str, family_type: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("{family_type} 缺少必需字段 {name}"))
}

#[allow(clippy::too_many_arguments)]
fn build_spec(
    family_type: &str,
    point: u8,
    n_orbits: usize,
    max_amplitude_km: Option<f64>,
    min_amplitude_km: Option<f64>,
    perilune_height_max_km: Option<f64>,
    north_south: Option<u8>,
    amplitude_in_km: Option<f64>,
    amplitude_out_km: Option<f64>,
    phase_in: Option<f64>,
    phase_out: Option<f64>,
    continuation_direction: Option<&str>,
    match_tolerance_km: Option<f64>,
    n_periods: usize,
) -> Result<Spec, String> {
    match family_type {
        "halo" => Ok(Spec::Halo {
            point,
            max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
            member_limit: n_orbits,
        }),
        "nrho" => Ok(Spec::Nrho {
            point,
            north_south: required(north_south, "north_south", family_type)?,
            perilune_height_max_km: required(
                perilune_height_max_km,
                "perilune_height_max_km",
                family_type,
            )?,
            member_limit: n_orbits,
        }),
        "axial" => Ok(Spec::Axial {
            point,
            max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
            member_limit: n_orbits,
        }),
        "lissajous" => Ok(Spec::Lissajous {
            point,
            amplitude_in_km: required(amplitude_in_km, "amplitude_in_km", family_type)?,
            amplitude_out_km: required(amplitude_out_km, "amplitude_out_km", family_type)?,
            phase_in: required(phase_in, "phase_in", family_type)?,
            phase_out: required(phase_out, "phase_out", family_type)?,
            member_limit: n_orbits,
            n_periods,
        }),
        "spo" | "lpo" | "horseshoe" => {
            let direction = match continuation_direction.unwrap_or("decrease-x0") {
                "increase-x0" => "increase-x0",
                "decrease-x0" => "decrease-x0",
                value => return Err(format!("未知 continuation_direction {value}")),
            };
            let kind = match family_type {
                "spo" => "spo",
                "lpo" => "lpo",
                _ => "horseshoe",
            };
            Ok(Spec::Triangular {
                family_type: kind,
                point,
                min_amplitude_km: required(min_amplitude_km, "min_amplitude_km", family_type)?,
                max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
                member_limit: n_orbits,
                direction,
                match_tolerance_km: required(
                    match_tolerance_km,
                    "match_tolerance_km",
                    family_type,
                )?,
            })
        }
        _ => Err(format!("未知 orbit family {family_type}")),
    }
}

fn generate(context: Context, spec: Spec) -> Result<Outcome, String> {
    match spec {
        Spec::Halo {
            point,
            max_amplitude_km,
            member_limit,
        } => halo::generate(context, point, max_amplitude_km, member_limit),
        Spec::Nrho {
            point,
            north_south,
            perilune_height_max_km,
            member_limit,
        } => nrho::generate(
            context,
            point,
            north_south,
            perilune_height_max_km,
            member_limit,
        ),
        Spec::Axial {
            point,
            max_amplitude_km,
            member_limit,
        } => axial::generate(context, point, max_amplitude_km, member_limit),
        Spec::Lissajous {
            point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            member_limit,
            n_periods,
        } => lissajous::generate(
            context,
            point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            member_limit,
            n_periods,
        ),
        Spec::Triangular {
            family_type,
            point,
            min_amplitude_km,
            max_amplitude_km,
            member_limit,
            direction,
            match_tolerance_km,
        } => triangular::generate(
            context,
            family_type,
            point,
            min_amplitude_km,
            max_amplitude_km,
            member_limit,
            direction,
            match_tolerance_km,
        ),
    }
}

/// 一次调用完成七类 CR3BP 轨道族生成。
#[pyfunction]
#[pyo3(signature = (
    family_type,
    mu,
    characteristic_length_km,
    secondary_radius_km,
    point,
    n_orbits,
    max_amplitude_km=None,
    min_amplitude_km=None,
    perilune_height_max_km=None,
    north_south=None,
    amplitude_in_km=None,
    amplitude_out_km=None,
    phase_in=None,
    phase_out=None,
    continuation_direction=None,
    match_tolerance_km=None,
    n_periods=3,
    rtol=1e-12,
    atol=1e-12,
    max_step=None
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_cr3bp_family_py(
    family_type: &str,
    mu: f64,
    characteristic_length_km: f64,
    secondary_radius_km: f64,
    point: u8,
    n_orbits: usize,
    max_amplitude_km: Option<f64>,
    min_amplitude_km: Option<f64>,
    perilune_height_max_km: Option<f64>,
    north_south: Option<u8>,
    amplitude_in_km: Option<f64>,
    amplitude_out_km: Option<f64>,
    phase_in: Option<f64>,
    phase_out: Option<f64>,
    continuation_direction: Option<&str>,
    match_tolerance_km: Option<f64>,
    n_periods: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if !(0.0..0.5).contains(&mu)
        || characteristic_length_km <= 0.0
        || secondary_radius_km <= 0.0
        || rtol <= 0.0
        || atol <= 0.0
        || max_step.is_some_and(|step| step <= 0.0)
    {
        return Err(PyValueError::new_err("CR3BP 族生成上下文无效"));
    }
    let spec = build_spec(
        family_type,
        point,
        n_orbits,
        max_amplitude_km,
        min_amplitude_km,
        perilune_height_max_km,
        north_south,
        amplitude_in_km,
        amplitude_out_km,
        phase_in,
        phase_out,
        continuation_direction,
        match_tolerance_km,
        n_periods,
    )
    .map_err(PyValueError::new_err)?;
    let context = Context {
        mu,
        characteristic_length_km,
        secondary_radius_km,
        rtol,
        atol,
        max_step,
    };
    let outcome = py
        .allow_threads(|| generate(context, spec))
        .map_err(PyValueError::new_err)?;
    outcome_to_python(py, outcome)
}

fn outcome_to_python(py: Python<'_>, outcome: Outcome) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("family_type", outcome.family_type)?;
    result.set_item("periodicity", outcome.periodicity)?;
    result.set_item("status", outcome.status)?;
    result.set_item("cause", outcome.cause)?;
    result.set_item("message", outcome.message)?;
    result.set_item("requested_members", outcome.requested_members)?;
    result.set_item("generated_members", outcome.members.len())?;
    let members = PyList::empty(py);
    for member in outcome.members {
        let item = PyDict::new(py);
        item.set_item("states", member.states)?;
        item.set_item("times", member.times)?;
        item.set_item("period", member.period)?;
        item.set_item("closure_error", member.closure_error)?;
        item.set_item("amplitude_km", member.amplitude_km)?;
        item.set_item("perilune_height_km", member.perilune_height_km)?;
        item.set_item("sampling_fraction", member.sampling_fraction)?;
        item.set_item("jacobi_drift", member.jacobi_drift)?;
        item.set_item("newton_iterations", member.newton_iterations)?;
        item.set_item("tangent_system_rank", member.tangent_system_rank)?;
        item.set_item("tangent_system_condition", member.tangent_system_condition)?;
        item.set_item("augmented_system_rank", member.augmented_system_rank)?;
        item.set_item(
            "augmented_system_condition",
            member.augmented_system_condition,
        )?;
        item.set_item("step_size", member.step_size)?;
        members.append(item)?;
    }
    result.set_item("members", members)?;
    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Context {
        Context {
            mu: 0.012_150_585_350_562_453,
            characteristic_length_km: 384_400.0,
            secondary_radius_km: 1737.4,
            rtol: 1e-12,
            atol: 1e-12,
            max_step: Some(0.01),
        }
    }

    #[test]
    fn lissajous_single_call_generates_requested_samples() {
        let result = generate(
            context(),
            Spec::Lissajous {
                point: 2,
                amplitude_in_km: 2400.0,
                amplitude_out_km: 7200.0,
                phase_in: 0.01,
                phase_out: 0.55,
                member_limit: 3,
                n_periods: 3,
            },
        )
        .unwrap();
        assert_eq!(result.status, "converged");
        assert_eq!(result.periodicity, "quasi-periodic");
        assert_eq!(result.members.len(), 3);
        assert_eq!(result.members[0].states.len(), 180);
        let x_l = crate::family::collinear_libration_x(context().mu, 2).unwrap();
        for (index, member) in result.members.iter().enumerate() {
            let requested_fraction = (index + 1) as f64 / 3.0;
            let requested_scale_km = requested_fraction * 2400.0_f64.hypot(7200.0);
            let max_distance_km = member
                .states
                .iter()
                .map(|state| {
                    ((state[0] - x_l).powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt()
                        * context().characteristic_length_km
                })
                .fold(0.0, f64::max);
            assert!(max_distance_km < 2.5 * requested_scale_km);
        }
    }

    #[test]
    fn periodic_families_satisfy_unified_result_contract() {
        let specs = [
            Spec::Halo {
                point: 1,
                max_amplitude_km: 3000.0,
                member_limit: 2,
            },
            Spec::Nrho {
                point: 1,
                north_south: 1,
                perilune_height_max_km: 30_000.0,
                member_limit: 2,
            },
            Spec::Nrho {
                point: 2,
                north_south: 2,
                perilune_height_max_km: 30_000.0,
                member_limit: 2,
            },
            Spec::Axial {
                point: 2,
                max_amplitude_km: 1500.0,
                member_limit: 2,
            },
            Spec::Triangular {
                family_type: "spo",
                point: 4,
                min_amplitude_km: 5000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 2,
                direction: "decrease-x0",
                match_tolerance_km: 20.0,
            },
            Spec::Triangular {
                family_type: "lpo",
                point: 5,
                min_amplitude_km: 5000.0,
                max_amplitude_km: 30_000.0,
                member_limit: 2,
                direction: "decrease-x0",
                match_tolerance_km: 20.0,
            },
            Spec::Triangular {
                family_type: "horseshoe",
                point: 4,
                min_amplitude_km: 50_000.0,
                max_amplitude_km: 110_000.0,
                member_limit: 1,
                direction: "decrease-x0",
                match_tolerance_km: 50.0,
            },
        ];

        for spec in specs {
            let result = generate(context(), spec).unwrap();
            assert_eq!(
                result.status, "converged",
                "{}: {}",
                result.cause, result.message
            );
            assert_eq!(result.periodicity, "periodic");
            assert!(!result.members.is_empty());
            for member in result.members {
                assert!(member.period.is_some_and(|period| period > 0.0));
                assert!(member.closure_error.is_some_and(|closure| closure <= 1e-8));
            }
        }
    }
}
