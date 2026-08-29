//! 低推力直接法求解器的 Rust 数值评估内核。
//!
//! Python 保留 SLSQP 编排、初猜和结果解释；本模块负责每轮都会重复执行的
//! 多段受控传播/灵敏度链式组合，以及 Hermite-Simpson 缺陷批量求值。

use e2m2e_forces::forces::augmented_state::{
    augmented_eom_7d, augmented_eom_7d_with_sensitivity, ThrustParams,
};
use e2m2e_forces::forces::compiled::CompiledForce;
use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::parse_force_tuple;

const MAX_STEPS_PER_SEGMENT: usize = 500_000;

fn parse_forces(forces_py: &Bound<'_, PyList>) -> PyResult<Vec<CompiledForce>> {
    let mut forces = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }
    Ok(forces)
}

fn direction(theta1: f64, theta2: f64) -> [f64; 3] {
    [
        theta1.cos() * theta2.cos(),
        theta1.sin() * theta2.cos(),
        theta2.sin(),
    ]
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn propagate_segment(
    method: RkMethod,
    t0: f64,
    tf: f64,
    initial_state: &[f64; 7],
    h_init: f64,
    tol: f64,
    observer: &str,
    forces: &[CompiledForce],
    thrust: ThrustParams,
    angles: Option<(f64, f64)>,
) -> PyResult<([f64; 7], Option<([f64; 36], [f64; 21])>)> {
    let table = method.table();
    let mut t = t0;
    let mut h = h_init;
    let mut n_steps = 0usize;

    if let Some((theta1, theta2)) = angles {
        let mut state = [0.0; 64];
        state[..7].copy_from_slice(initial_state);
        for index in 0..6 {
            state[7 + index * 6 + index] = 1.0;
        }
        while t < tf && n_steps < MAX_STEPS_PER_SEGMENT {
            n_steps += 1;
            if t + h > tf {
                h = tf - t;
            }
            if h > h_init {
                h = h_init;
            }
            let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
                let mut state64 = [0.0; 64];
                state64.copy_from_slice(&yi[..64]);
                augmented_eom_7d_with_sensitivity(
                    forces, observer, ti, &state64, &thrust, theta1, theta2,
                )
                .map(|derivative| derivative.to_vec())
            };
            let (next, error) = explicit_rk_step(table, t, &state, h, callback, None)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            if error <= tol {
                t += h;
                state.copy_from_slice(&next);
                h = suggest_next_step(h, error, tol, method.embedded_order());
            } else {
                h = suggest_next_step(h, error, tol, method.embedded_order());
                if h < 1e-12 * (tf - t0).abs() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "step size collapsed below minimum",
                    ));
                }
            }
        }
        if t < tf {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "propagation reached max_steps ({MAX_STEPS_PER_SEGMENT}) before t_final"
            )));
        }
        let mut final_state = [0.0; 7];
        final_state.copy_from_slice(&state[..7]);
        let mut stm = [0.0; 36];
        stm.copy_from_slice(&state[7..43]);
        let mut sensitivity = [0.0; 21];
        sensitivity.copy_from_slice(&state[43..64]);
        Ok((final_state, Some((stm, sensitivity))))
    } else {
        let mut state = *initial_state;
        while t < tf && n_steps < MAX_STEPS_PER_SEGMENT {
            n_steps += 1;
            if t + h > tf {
                h = tf - t;
            }
            if h > h_init {
                h = h_init;
            }
            let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
                let state7 = [yi[0], yi[1], yi[2], yi[3], yi[4], yi[5], yi[6]];
                augmented_eom_7d(forces, observer, ti, &state7, &thrust)
                    .map(|derivative| derivative.to_vec())
            };
            let (next, error) = explicit_rk_step(table, t, &state, h, callback, None)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            if error <= tol {
                t += h;
                state.copy_from_slice(&next);
                h = suggest_next_step(h, error, tol, method.embedded_order());
            } else {
                h = suggest_next_step(h, error, tol, method.embedded_order());
                if h < 1e-12 * (tf - t0).abs() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "step size collapsed below minimum",
                    ));
                }
            }
        }
        if t < tf {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "propagation reached max_steps ({MAX_STEPS_PER_SEGMENT}) before t_final"
            )));
        }
        Ok((state, None))
    }
}

fn matmul(left: &[f64; 36], right: &[f64; 36]) -> [f64; 36] {
    let mut product = [0.0; 36];
    for row in 0..6 {
        for column in 0..6 {
            for index in 0..6 {
                product[row * 6 + column] += left[row * 6 + index] * right[index * 6 + column];
            }
        }
    }
    product
}

/// 多段低推力打靶评估：接龙传播，并按需组装末端控制灵敏度。
#[pyfunction]
#[pyo3(signature = (controls, t0, tf, initial_state, observer, forces_py, t_max, isp, with_jacobian))]
#[allow(clippy::too_many_arguments)]
pub fn lowthrust_shooting_evaluate_py(
    controls: Vec<Vec<f64>>,
    t0: f64,
    tf: f64,
    initial_state: Vec<f64>,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_max: f64,
    isp: f64,
    with_jacobian: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if controls.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "controls must not be empty",
        ));
    }
    if initial_state.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 7, got {}",
            initial_state.len()
        )));
    }
    if tf <= t0 || t_max <= 0.0 || isp <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tf must exceed t0; t_max and isp must be positive",
        ));
    }
    let forces = parse_forces(forces_py)?;
    let n_segments = controls.len();
    let dt = (tf - t0) / n_segments as f64;
    let mut state = [0.0; 7];
    state.copy_from_slice(&initial_state);
    let mut states = vec![state.to_vec()];
    let mut times = vec![t0];
    let mut stms = Vec::with_capacity(n_segments);
    let mut sensitivities = Vec::with_capacity(n_segments);

    for (index, control) in controls.iter().enumerate() {
        if control.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "control {index} must have length 3, got {}",
                control.len()
            )));
        }
        let throttle = control[0];
        if !(0.0..=1.0).contains(&throttle) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "throttle must be in [0, 1], got {throttle}"
            )));
        }
        let theta1 = control[1];
        let theta2 = control[2];
        let thrust = ThrustParams {
            t_max,
            isp,
            throttle,
            direction: direction(theta1, theta2),
        };
        let segment_t0 = t0 + index as f64 * dt;
        let initial_step = if dt < 1.0 {
            dt
        } else {
            (dt / 10.0).clamp(1.0, dt)
        };
        let (next, derivative_data) = propagate_segment(
            RkMethod::Pd45,
            segment_t0,
            segment_t0 + dt,
            &state,
            initial_step,
            1e-10,
            observer,
            &forces,
            thrust,
            with_jacobian.then_some((theta1, theta2)),
        )?;
        if let Some((stm, sensitivity)) = derivative_data {
            stms.push(stm);
            sensitivities.push(sensitivity);
        }
        state = next;
        times.push(segment_t0 + dt);
        states.push(state.to_vec());
    }

    let output = PyDict::new(py);
    output.set_item("time", times)?;
    output.set_item("states", states)?;
    if with_jacobian {
        let mut jacobian = vec![0.0; 6 * 3 * n_segments];
        let mut composite = [0.0; 36];
        for index in 0..6 {
            composite[index * 6 + index] = 1.0;
        }
        for segment in (0..n_segments).rev() {
            for row in 0..6 {
                for column in 0..3 {
                    for index in 0..6 {
                        jacobian[row * 3 * n_segments + segment * 3 + column] +=
                            composite[row * 6 + index] * sensitivities[segment][index * 3 + column];
                    }
                }
            }
            composite = matmul(&composite, &stms[segment]);
        }
        let rows: Vec<Vec<f64>> = jacobian
            .chunks_exact(3 * n_segments)
            .map(|row| row.to_vec())
            .collect();
        output.set_item("terminal_jacobian", rows)?;
    }
    Ok(output.into())
}

/// 固定离散工况的 Hermite-Simpson 配点缺陷。
///
/// `levels[i]` 是第 i 条区间的推力百分比，只允许 0、60、100；
/// `controls` 仍为每个节点的方向角 `(unused_throttle, theta1, theta2)`，
/// 其中第一列会被忽略，避免把连续油门混入离散弧段模型。
#[allow(clippy::too_many_arguments)]
fn discrete_collocation_defects(
    states: &[Vec<f64>],
    controls: &[Vec<f64>],
    levels: &[i32],
    t_nodes: &[f64],
    observer: &str,
    forces: &[e2m2e_forces::forces::compiled::CompiledForce],
    t_max: f64,
    isp: f64,
) -> Result<Vec<f64>, String> {
    let n_segments = states.len() - 1;
    let mut defects = Vec::with_capacity(7 * n_segments);
    for index in 0..n_segments {
        if states[index].len() != 7 || states[index + 1].len() != 7 {
            return Err("every state must have length 7".into());
        }
        if controls[index].len() != 3 || controls[index + 1].len() != 3 {
            return Err("every control must have length 3".into());
        }
        let xi: [f64; 7] = states[index].clone().try_into().expect("validated length");
        let xip1: [f64; 7] = states[index + 1]
            .clone()
            .try_into()
            .expect("validated length");
        let level = levels[index] as f64 / 100.0;
        let pi = &controls[index];
        let pip1 = &controls[index + 1];
        let dt = t_nodes[index + 1] - t_nodes[index];
        let et = t_nodes[index];
        let thrust_i = ThrustParams {
            t_max,
            isp,
            throttle: level,
            direction: direction(pi[1], pi[2]),
        };
        let thrust_ip1 = ThrustParams {
            t_max,
            isp,
            throttle: level,
            direction: direction(pip1[1], pip1[2]),
        };
        let fi =
            augmented_eom_7d(forces, observer, et, &xi, &thrust_i).map_err(|e| e.to_string())?;
        let fip1 = augmented_eom_7d(forces, observer, et + dt, &xip1, &thrust_ip1)
            .map_err(|e| e.to_string())?;
        let mut midpoint_state = [0.0; 7];
        for component in 0..7 {
            midpoint_state[component] = (xi[component] + xip1[component]) / 2.0
                + dt / 8.0 * (fi[component] - fip1[component]);
        }
        let midpoint_thrust = ThrustParams {
            t_max,
            isp,
            throttle: level,
            direction: direction((pi[1] + pip1[1]) / 2.0, (pi[2] + pip1[2]) / 2.0),
        };
        let midpoint = augmented_eom_7d(
            forces,
            observer,
            et + dt / 2.0,
            &midpoint_state,
            &midpoint_thrust,
        )
        .map_err(|e| e.to_string())?;
        for component in 0..7 {
            defects.push(
                xip1[component]
                    - xi[component]
                    - dt / 6.0 * (fi[component] + 4.0 * midpoint[component] + fip1[component]),
            );
        }
    }
    Ok(defects)
}

#[pyfunction]
#[pyo3(signature = (states, controls, levels, t0, tf, observer, forces_py, t_max, isp))]
#[allow(clippy::too_many_arguments)]
pub fn lowthrust_discrete_collocation_defects_py(
    states: Vec<Vec<f64>>,
    controls: Vec<Vec<f64>>,
    levels: Vec<i32>,
    t0: f64,
    tf: f64,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_max: f64,
    isp: f64,
) -> PyResult<Vec<f64>> {
    if states.len() < 2 || controls.len() != states.len() || levels.len() + 1 != states.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states/controls/levels lengths must be N+1/N+1/N",
        ));
    }
    if tf <= t0 {
        return Err(pyo3::exceptions::PyValueError::new_err("tf must exceed t0"));
    }
    if levels.iter().any(|level| !matches!(level, 0 | 60 | 100)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "discrete thrust levels must be 0, 60, or 100",
        ));
    }
    let forces = parse_forces(forces_py)?;
    let n_segments = states.len() - 1;
    let dt = (tf - t0) / n_segments as f64;
    let t_nodes: Vec<f64> = (0..=n_segments).map(|i| t0 + i as f64 * dt).collect();
    discrete_collocation_defects(
        &states, &controls, &levels, &t_nodes, observer, &forces, t_max, isp,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// 变时长离散工况的 Hermite-Simpson 配点缺陷。
///
/// 与 `lowthrust_discrete_collocation_defects_py` 的唯一区别是 `t_nodes`
/// 显式给出，供把弧起止时刻作为 NLP 决策变量时使用。
#[pyfunction]
#[pyo3(signature = (states, controls, levels, t_nodes, observer, forces_py, t_max, isp))]
#[allow(clippy::too_many_arguments)]
pub fn lowthrust_variable_time_collocation_defects_py(
    states: Vec<Vec<f64>>,
    controls: Vec<Vec<f64>>,
    levels: Vec<i32>,
    t_nodes: Vec<f64>,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_max: f64,
    isp: f64,
) -> PyResult<Vec<f64>> {
    if states.len() < 2
        || controls.len() != states.len()
        || levels.len() + 1 != states.len()
        || t_nodes.len() != states.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states/controls/levels/t_nodes lengths must be N+1/N+1/N/N+1",
        ));
    }
    if levels.iter().any(|level| !matches!(level, 0 | 60 | 100)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "discrete thrust levels must be 0, 60, or 100",
        ));
    }
    if t_nodes.windows(2).any(|w| w[1] <= w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_nodes must be strictly increasing",
        ));
    }
    let forces = parse_forces(forces_py)?;
    discrete_collocation_defects(
        &states, &controls, &levels, &t_nodes, observer, &forces, t_max, isp,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Hermite-Simpson 配点缺陷的批量求值。
#[pyfunction]
#[pyo3(signature = (states, controls, t0, tf, observer, forces_py, t_max, isp))]
#[allow(clippy::too_many_arguments)]
pub fn lowthrust_collocation_defects_py(
    states: Vec<Vec<f64>>,
    controls: Vec<Vec<f64>>,
    t0: f64,
    tf: f64,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_max: f64,
    isp: f64,
) -> PyResult<Vec<f64>> {
    if states.len() < 2 || states.len() != controls.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states and controls must have the same length of at least 2",
        ));
    }
    let forces = parse_forces(forces_py)?;
    let n_segments = states.len() - 1;
    let dt = (tf - t0) / n_segments as f64;
    let mut defects = Vec::with_capacity(7 * n_segments);

    for index in 0..n_segments {
        if states[index].len() != 7 || states[index + 1].len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "every state must have length 7",
            ));
        }
        if controls[index].len() != 3 || controls[index + 1].len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "every control must have length 3",
            ));
        }
        let xi: [f64; 7] = states[index].clone().try_into().expect("validated length");
        let xip1: [f64; 7] = states[index + 1]
            .clone()
            .try_into()
            .expect("validated length");
        let pi = &controls[index];
        let pip1 = &controls[index + 1];
        let thrust_i = ThrustParams {
            t_max,
            isp,
            throttle: pi[0],
            direction: direction(pi[1], pi[2]),
        };
        let thrust_ip1 = ThrustParams {
            t_max,
            isp,
            throttle: pip1[0],
            direction: direction(pip1[1], pip1[2]),
        };
        let et = t0 + index as f64 * dt;
        let fi = augmented_eom_7d(&forces, observer, et, &xi, &thrust_i)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let fip1 = augmented_eom_7d(&forces, observer, et + dt, &xip1, &thrust_ip1)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let mut midpoint_state = [0.0; 7];
        for component in 0..7 {
            midpoint_state[component] = (xi[component] + xip1[component]) / 2.0
                + dt / 8.0 * (fi[component] - fip1[component]);
        }
        let midpoint_thrust = ThrustParams {
            t_max,
            isp,
            throttle: (pi[0] + pip1[0]) / 2.0,
            direction: direction((pi[1] + pip1[1]) / 2.0, (pi[2] + pip1[2]) / 2.0),
        };
        let midpoint = augmented_eom_7d(
            &forces,
            observer,
            et + dt / 2.0,
            &midpoint_state,
            &midpoint_thrust,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        for component in 0..7 {
            defects.push(
                xip1[component]
                    - xi[component]
                    - dt / 6.0 * (fi[component] + 4.0 * midpoint[component] + fip1[component]),
            );
        }
    }
    Ok(defects)
}
