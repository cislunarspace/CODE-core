//! e2m2e-forces: Force models (N-body, gravity field, SRP, STM).
//!
//! 从 e2m2e-integrators 拆分，包含力模型和 STM 变分方程。

use pyo3::prelude::*;

#[cfg(feature = "spice")]
pub mod forces;
pub mod solid_tide;
pub mod spherical_harmonic;

/// 球谐引力加速度（body-fixed 系）。
#[pyfunction]
fn spherical_harmonic_accel(
    r: Vec<f64>,
    c_flat: Vec<f64>,
    s_flat: Vec<f64>,
    mu: f64,
    radius: f64,
    degree: usize,
    order: usize,
) -> PyResult<Vec<f64>> {
    if r.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "r must have length 3, got {}",
            r.len()
        )));
    }
    let nn = degree + 1;
    if c_flat.len() != nn * nn || s_flat.len() != nn * nn {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "C/S flattened length must be (degree+1)^2 = {}, got C={} S={}",
            nn * nn,
            c_flat.len(),
            s_flat.len()
        )));
    }
    if order > degree {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "order ({}) must be <= degree ({})",
            order, degree
        )));
    }
    Ok(spherical_harmonic::spherical_harmonic_accel(
        &r, &c_flat, &s_flat, mu, radius, degree, order,
    ))
}

/// 固体潮 Step 1。
#[pyfunction]
fn solid_tide_step1(
    perturbers_flat: Vec<f64>,
    k_love_flat: Vec<f64>,
    k_plus_flat: Option<Vec<f64>>,
    mu_central: f64,
    r_central: f64,
) -> PyResult<Vec<f64>> {
    if k_love_flat.len() != 25 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k_love_flat must be length 25, got {}",
            k_love_flat.len()
        )));
    }
    if let Some(kp) = &k_plus_flat {
        if kp.len() != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k_plus_flat must be length 5, got {}",
                kp.len()
            )));
        }
    }
    if !perturbers_flat.len().is_multiple_of(4) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "perturbers_flat length must be multiple of 4, got {}",
            perturbers_flat.len()
        )));
    }
    let k_plus_ref = k_plus_flat.as_deref();
    Ok(solid_tide::solid_tide_step1(
        &perturbers_flat,
        &k_love_flat,
        k_plus_ref,
        mu_central,
        r_central,
    ))
}

/// 固体潮 Step 2。
#[pyfunction]
fn solid_tide_step2(et: f64) -> PyResult<Vec<f64>> {
    Ok(solid_tide::solid_tide_step2(et))
}

/// 极潮。
#[pyfunction]
fn pole_tide(et: f64, xp: f64, yp: f64) -> PyResult<Vec<f64>> {
    Ok(solid_tide::pole_tide(et, xp, yp))
}

/// 占位函数。
#[pyfunction]
fn hello_forces() -> PyResult<String> {
    Ok("hello from e2m2e-forces".to_string())
}

#[pymodule]
fn _forces(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_forces, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_accel, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step1, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step2, m)?)?;
    m.add_function(wrap_pyfunction!(pole_tide, m)?)?;
    Ok(())
}
