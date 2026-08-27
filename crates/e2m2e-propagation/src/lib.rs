//! e2m2e-propagation: ODE integrators (RK, ABM, Cowell).
//!
//! 从 e2m2e-integrators 拆分，只包含纯数学积分器，不依赖 SPICE。
//! 物理常数由 build.rs 从包内 constants.toml 生成，通过本模块统一导出。
//!
//! 仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

pub mod abm;
pub mod butcher;
pub mod cowell;
pub mod lambert;
pub mod multistep_methods;
pub mod pd45;
pub mod pd78;
pub mod rk89;
pub mod rk_methods;
pub mod solve_ivp;

/// 自动生成的物理常量模块（单一来源：constants.toml）。
/// 由 build.rs 写入 OUT_DIR，编译期 include。
pub mod constants {
    include!(concat!(env!("OUT_DIR"), "/generated_constants.rs"));
}

/// 返回 Rust 从 constants.toml 生成的常量值（Rust→Python 同源核对入口）。
///
/// 支持 `category.key` 形式：
/// - `universal.speed_of_light_kms`
/// - `universal.au_km`
/// - `body.EARTH.mean_radius_km`
/// - `body.EARTH.gm.DE421`
/// - `datum.DE421.mu`
/// - `datum.WGS84.earth_radius_km`
#[cfg(feature = "pyo3")]
#[pyo3::pyfunction]
fn constant_value_py(key: &str) -> pyo3::PyResult<f64> {
    for (k, v) in constants::CONSTANT_LOOKUP {
        if *k == key {
            return Ok(*v);
        }
    }
    Err(pyo3::exceptions::PyKeyError::new_err(format!(
        "unknown constant key: {key}"
    )))
}

/// 返回一个已初始化的 `_propagation_constants` 子模块（供 `e2m2e-integrators`
/// 在 `_integrators` 中 `add_submodule` 使用）。
#[cfg(feature = "pyo3")]
pub fn _propagation_constants_module_bound(
    py: pyo3::Python<'_>,
) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::types::PyModule>> {
    use pyo3::types::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    let m = pyo3::types::PyModule::new(py, "_propagation_constants")?;
    m.add_function(wrap_pyfunction!(constant_value_py, &m)?)?;
    Ok(m)
}
