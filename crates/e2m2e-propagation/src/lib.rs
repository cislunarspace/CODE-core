//! e2m2e-propagation: ODE integrators (RK, ABM, Cowell).
//!
//! 从 e2m2e-integrators 拆分，只包含纯数学积分器，不依赖 SPICE。
//! 物理常数由 build.rs 从仓库根 constants.toml 生成，通过本模块统一导出。

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
    use crate::constants::*;
    use pyo3::exceptions::PyKeyError;

    match key {
        "universal.speed_of_light_kms" => Ok(SPEED_OF_LIGHT_KMS),
        "universal.gravitational_constant" => Ok(GRAVITATIONAL_CONSTANT),
        "universal.au_km" => Ok(AU_KM),
        "universal.seconds_per_day" => Ok(SECONDS_PER_DAY),
        "universal.days_per_julian_year" => Ok(DAYS_PER_JULIAN_YEAR),
        "universal.days_per_julian_century" => Ok(DAYS_PER_JULIAN_CENTURY),
        "universal.km_to_m" => Ok(KM_TO_M),
        "universal.solar_flux_w_m2" => Ok(SOLAR_FLUX_W_M2),
        "universal.solar_flux_tsi_w_m2" => Ok(SOLAR_FLUX_TSI_W_M2),
        "universal.rad_per_deg" => Ok(RAD_PER_DEG),
        "universal.solar_pressure_1au" => Ok(SOLAR_PRESSURE_1AU),

        "datum.DE421.mu" => Ok(DATUM_DE421_MU),
        "datum.DE421.char_length_km" => Ok(DATUM_DE421_CHAR_LENGTH_KM),
        "datum.DE421.char_time_s" => Ok(DATUM_DE421_CHAR_TIME_S),
        "datum.DE421.earth_gm" => Ok(DATUM_DE421_EARTH_GM),
        "datum.DE421.moon_gm" => Ok(DATUM_DE421_MOON_GM),
        "datum.DE421.sun_gm" => Ok(DATUM_DE421_SUN_GM),
        "datum.DE421.emb_gm" => Ok(DATUM_DE421_EMB_GM),

        "datum.DE440.mu" => Ok(DATUM_DE440_MU),
        "datum.DE440.earth_gm" => Ok(DATUM_DE440_EARTH_GM),
        "datum.DE440.moon_gm" => Ok(DATUM_DE440_MOON_GM),
        "datum.DE440.sun_gm" => Ok(DATUM_DE440_SUN_GM),
        "datum.DE440.emb_gm" => Ok(DATUM_DE440_EMB_GM),

        "datum.WGS84.earth_gm" => Ok(DATUM_WGS84_EARTH_GM),
        "datum.WGS84.earth_radius_km" => Ok(DATUM_WGS84_EARTH_RADIUS_KM),
        "datum.WGS84.earth_flattening" => Ok(DATUM_WGS84_EARTH_FLATTENING),

        "body.SUN.mean_radius_km" => Ok(SUN_MEAN_RADIUS_KM),
        "body.SUN.naif_id" => Ok(SUN_NAIF_ID as f64),
        "body.SUN.gm.DE440" => Ok(SUN_GM_DE440),

        "body.EARTH.mean_radius_km" => Ok(EARTH_MEAN_RADIUS_KM),
        "body.EARTH.gravity_ref_radius_km" => Ok(EARTH_GRAVITY_REF_RADIUS_KM),
        "body.EARTH.flattening" => Ok(EARTH_FLATTENING),
        "body.EARTH.naif_id" => Ok(EARTH_NAIF_ID as f64),
        "body.EARTH.rotation_rate_iers_rad_s" => Ok(EARTH_ROTATION_RATE_IERS_RAD_S),
        "body.EARTH.rotation_rate_gmat_rad_s" => Ok(EARTH_ROTATION_RATE_GMAT_RAD_S),
        "body.EARTH.gm.DE421" => Ok(EARTH_GM_DE421),
        "body.EARTH.gm.DE440" => Ok(EARTH_GM_DE440),
        "body.EARTH.gm.WGS84" => Ok(EARTH_GM_WGS84),

        "body.MOON.mean_radius_km" => Ok(MOON_MEAN_RADIUS_KM),
        "body.MOON.gravity_ref_radius_km" => Ok(MOON_GRAVITY_REF_RADIUS_KM),
        "body.MOON.naif_id" => Ok(MOON_NAIF_ID as f64),
        "body.MOON.gm.DE421" => Ok(MOON_GM_DE421),
        "body.MOON.gm.DE440" => Ok(MOON_GM_DE440),

        "body.EMB.naif_id" => Ok(EMB_NAIF_ID as f64),
        "body.EMB.gm.DE421" => Ok(EMB_GM_DE421),
        "body.EMB.gm.DE440" => Ok(EMB_GM_DE440),

        "body.MERCURY.mean_radius_km" => Ok(MERCURY_MEAN_RADIUS_KM),
        "body.MERCURY.naif_id" => Ok(MERCURY_NAIF_ID as f64),
        "body.MERCURY.gm.DE440" => Ok(MERCURY_GM_DE440),
        "body.VENUS.mean_radius_km" => Ok(VENUS_MEAN_RADIUS_KM),
        "body.VENUS.naif_id" => Ok(VENUS_NAIF_ID as f64),
        "body.VENUS.gm.DE440" => Ok(VENUS_GM_DE440),
        "body.MARS.mean_radius_km" => Ok(MARS_MEAN_RADIUS_KM),
        "body.MARS.naif_id" => Ok(MARS_NAIF_ID as f64),
        "body.MARS.gm.DE440" => Ok(MARS_GM_DE440),
        "body.JUPITER.mean_radius_km" => Ok(JUPITER_MEAN_RADIUS_KM),
        "body.JUPITER.naif_id" => Ok(JUPITER_NAIF_ID as f64),
        "body.JUPITER.gm.DE440" => Ok(JUPITER_GM_DE440),
        "body.SATURN.mean_radius_km" => Ok(SATURN_MEAN_RADIUS_KM),
        "body.SATURN.naif_id" => Ok(SATURN_NAIF_ID as f64),
        "body.SATURN.gm.DE440" => Ok(SATURN_GM_DE440),
        "body.URANUS.mean_radius_km" => Ok(URANUS_MEAN_RADIUS_KM),
        "body.URANUS.naif_id" => Ok(URANUS_NAIF_ID as f64),
        "body.URANUS.gm.DE440" => Ok(URANUS_GM_DE440),
        "body.NEPTUNE.mean_radius_km" => Ok(NEPTUNE_MEAN_RADIUS_KM),
        "body.NEPTUNE.naif_id" => Ok(NEPTUNE_NAIF_ID as f64),
        "body.NEPTUNE.gm.DE440" => Ok(NEPTUNE_GM_DE440),
        "body.PLUTO.mean_radius_km" => Ok(PLUTO_MEAN_RADIUS_KM),
        "body.PLUTO.naif_id" => Ok(PLUTO_NAIF_ID as f64),
        "body.PLUTO.gm.DE440" => Ok(PLUTO_GM_DE440),

        _ => Err(PyKeyError::new_err(format!("unknown constant key: {key}"))),
    }
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
