//! Q-law 低推力反馈积分与段中点评估的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::propagate_error_to_pyerr;

/// Q-law 低推力反馈积分（完整热路径在 Rust）。
#[pyfunction]
#[pyo3(signature = (t0, tf, y0, target_oe, mu, t_max, isp, h_init, tol, max_steps))]
#[allow(clippy::too_many_arguments)]
pub fn qlaw_propagate_py(
    t0: f64,
    tf: f64,
    y0: Vec<f64>,
    target_oe: Vec<f64>,
    mu: f64,
    t_max: f64,
    isp: f64,
    h_init: f64,
    tol: f64,
    max_steps: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if y0.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y0 must have length 7, got {}",
            y0.len()
        )));
    }
    if target_oe.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "target_oe must have length 3, got {}",
            target_oe.len()
        )));
    }
    if h_init <= 0.0 || tol <= 0.0 || max_steps == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "h_init, tol and max_steps must be positive",
        ));
    }
    let mut initial_state = [0.0; 7];
    initial_state.copy_from_slice(&y0);
    let mut target = [0.0; 3];
    target.copy_from_slice(&target_oe);

    let result = py.allow_threads(|| {
        e2m2e_dyn::qlaw::propagate(
            t0,
            tf,
            initial_state,
            target,
            mu,
            t_max,
            isp,
            h_init,
            tol,
            max_steps,
        )
    });
    let (times, states) =
        result.map_err(|error| propagate_error_to_pyerr(py, "Q-law propagation failed", error))?;
    let output = PyDict::new(py);
    output.set_item("time", times)?;
    output.set_item("states", states)?;
    Ok(output.into())
}

/// Q-law 段中点评估：Q 值、开普勒根数和惯性系推力方向。
#[pyfunction]
#[pyo3(signature = (state7, target_oe, mu, t_max))]
pub fn qlaw_segment_direction_py(
    state7: Vec<f64>,
    target_oe: Vec<f64>,
    mu: f64,
    t_max: f64,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if state7.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state7 must have length 7, got {}",
            state7.len()
        )));
    }
    if target_oe.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "target_oe must have length 3, got {}",
            target_oe.len()
        )));
    }
    let mut state = [0.0; 7];
    state.copy_from_slice(&state7);
    let mut target = [0.0; 3];
    target.copy_from_slice(&target_oe);
    let result = e2m2e_dyn::qlaw::evaluate_segment(&state, target, mu, t_max);
    let output = PyDict::new(py);
    output.set_item("a", result.a)?;
    output.set_item("e", result.e)?;
    output.set_item("i", result.inclination)?;
    output.set_item("q_value", result.q_value)?;
    output.set_item("u_inertial", result.inertial_direction.to_vec())?;
    Ok(output.into())
}
