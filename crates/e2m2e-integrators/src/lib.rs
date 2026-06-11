use pyo3::prelude::*;

/// A placeholder function to verify the FFI path works end-to-end.
#[pyfunction]
fn hello_integrators() -> PyResult<String> {
    Ok("hello from e2m2e-integrators".to_string())
}

#[pymodule]
fn _integrators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_integrators, m)?)?;
    Ok(())
}
