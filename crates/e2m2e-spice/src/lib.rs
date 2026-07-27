//! e2m2e-spice: SPICE FFI bindings.
//!
//! 从 e2m2e-integrators 拆分，只包含 SPICE 相关功能。

use pyo3::prelude::*;

#[cfg(feature = "spice")]
pub mod spice_ffi;
#[cfg(feature = "spice")]
pub mod spk_accel;

/// PoC：通过 cspice 查询天体位置。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_poc_body_position(et: f64, target: &str, observer: &str) -> PyResult<Vec<f64>> {
    use cspice::common::AberrationCorrection;
    use cspice::spk::easier_reader;
    use cspice::time::Et;

    let et_tdb = Et::from(et);
    let (state, _lt) = easier_reader(
        target,
        et_tdb,
        "J2000",
        AberrationCorrection::NONE,
        observer,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("cspice spkezr failed: {:?}", e))
    })?;
    Ok(vec![state.position.x, state.position.y, state.position.z])
}

/// PoC：在 Rust cspice 内核池加载一个内核文件。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_poc_furnsh(path: &str) -> PyResult<()> {
    cspice::data::furnish(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("furnsh failed: {:?}", e)))
}

/// 第三体摄动加速度（含直接项 + 间接项）。
#[cfg(feature = "spice")]
#[pyfunction]
fn third_body_acceleration(
    et: f64,
    target: &str,
    observer: &str,
    sc_pos: Vec<f64>,
    mu: f64,
) -> PyResult<Vec<f64>> {
    if sc_pos.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sc_pos must have length 3, got {}",
            sc_pos.len()
        )));
    }
    let a = spk_accel::third_body_acceleration(et, target, observer, &sc_pos, mu, 1e-6).map_err(
        |e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "third_body_acceleration cspice failed: {:?}",
                e
            ))
        },
    )?;
    Ok(vec![a[0], a[1], a[2]])
}

/// 第三体间接项加速度：`a = -μ · r_ob / |r_ob|³`。
#[cfg(feature = "spice")]
#[pyfunction]
fn indirect_term_acceleration(
    et: f64,
    target: &str,
    observer: &str,
    mu: f64,
) -> PyResult<Vec<f64>> {
    let a = spk_accel::indirect_term_acceleration(et, target, observer, mu, 1e-6).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "indirect_term_acceleration cspice failed: {:?}",
            e
        ))
    })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// 占位函数。
#[pyfunction]
fn hello_spice() -> PyResult<String> {
    Ok("hello from e2m2e-spice".to_string())
}

#[pymodule]
fn _spice(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_spice, m)?)?;
    #[cfg(feature = "spice")]
    {
        m.add_function(wrap_pyfunction!(spice_poc_body_position, m)?)?;
        m.add_function(wrap_pyfunction!(spice_poc_furnsh, m)?)?;
        m.add_function(wrap_pyfunction!(third_body_acceleration, m)?)?;
        m.add_function(wrap_pyfunction!(indirect_term_acceleration, m)?)?;
    }
    Ok(())
}
