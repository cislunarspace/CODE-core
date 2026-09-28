//! 力模型（球谐/固体潮/极潮/第三体/间接项/SRP/NRLMSISE-00）的 PyO3 绑定。

use pyo3::prelude::*;

/// 球谐引力加速度（body-fixed 系）。
///
/// Python 侧 `GravityField._compute_acceleration_in_input_frame` 的 Rust 加速版。
/// 输入位置 `r` 与输出加速度均在 body-fixed 系（坐标变换仍由 Python 完成）。
/// `c_flat`/`s_flat` 是 C/S 系数矩阵的行优先扁平化（shape=(degree+1)**2）。
#[pyfunction]
pub fn spherical_harmonic_accel(
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
    Ok(e2m2e_forces::spherical_harmonic::spherical_harmonic_accel(
        &r, &c_flat, &s_flat, mu, radius, degree, order,
    ))
}

/// 固体潮 Step 1（频率无关，天体无关）。
///
/// Python 侧 `earth_tide.solid_tide_step1` 的 Rust 加速版。输入扰动体位置由
/// Python 完成坐标变换后传入（本函数不查 SPICE）。
///
/// **参数**
///
/// - ``perturbers_flat`` ：扁平化扰动体列表，每 4 个一组 ``[px, py, pz, gm]`` （位置 km、
///   gm km³/s²）。长度必须是 4 的倍数。
/// - ``k_love_flat`` ：Love 数表 5×5 行优先扁平化，长度 25。
/// - ``k_plus_flat`` ：弹性 Love 数 5 元素，或 ``None`` （无贡献）。
/// - ``mu_central`` 、``r_central`` ：中心天体 GM 与参考半径。
///
/// **返回**
///
/// 长度 50 的 ``Vec<f64>`` ：``C(25) ++ S(25)`` ，各为 5×5 行优先扁平化。
#[pyfunction]
pub fn solid_tide_step1(
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
    Ok(e2m2e_forces::solid_tide::solid_tide_step1(
        &perturbers_flat,
        &k_love_flat,
        k_plus_ref,
        mu_central,
        r_central,
    ))
}

/// 固体潮 Step 2（频率相关，地球专用）。返回长度 50 的 ``Vec<f64>`` （C25 + S25）。
#[pyfunction]
pub fn solid_tide_step2(et: f64) -> PyResult<Vec<f64>> {
    Ok(e2m2e_forces::solid_tide::solid_tide_step2(et))
}

/// 极潮（固体极潮 + 海洋极潮，IERS TN32）。返回长度 50 的 ``Vec<f64>`` （C25 + S25）。
#[pyfunction]
pub fn pole_tide(et: f64, xp: f64, yp: f64) -> PyResult<Vec<f64>> {
    Ok(e2m2e_forces::solid_tide::pole_tide(et, xp, yp))
}

/// 第三体摄动加速度（含直接项 + 间接项）。
///
/// 移植自 Python `ThirdBodyGravity.compute_acceleration` 。一次调用完成
/// cspice 查扰动体位置 + 加速度公式，消除 Python↔cspice 跨界 + numpy
/// 数组分配开销。
///
/// # 参数
/// - `et` ：SPICE et 秒（past J2000 TDB）
/// - `target` ：摄动天体名（"MOON"/"SUN"/"5"=JUPITER 等）
/// - `observer` ：原点天体名（通常 "EARTH"）
/// - `sc_pos` ：航天器位置 [x, y, z] km（相对 observer），长度 3
/// - `mu` ：摄动天体 GM（km³/s²）
///
/// # 返回
/// 长度 3 的加速度 `Vec<f64>` ，单位 km/s²。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn third_body_acceleration(
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
    let a =
        e2m2e_spice::spk_accel::third_body_acceleration(et, target, observer, &sc_pos, mu, 1e-6)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "third_body_acceleration cspice failed: {:?}",
                    e
                ))
            })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// 第三体间接项加速度：`a = -μ · r_ob / |r_ob|³` 。
///
/// 移植自 Python `IndirectTerm.compute_acceleration` 。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn indirect_term_acceleration(
    et: f64,
    target: &str,
    observer: &str,
    mu: f64,
) -> PyResult<Vec<f64>> {
    let a = e2m2e_spice::spk_accel::indirect_term_acceleration(et, target, observer, mu, 1e-6)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "indirect_term_acceleration cspice failed: {:?}",
                e
            ))
        })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// GravityField 完整加速度（含坐标变换 + 球谐 + 潮汐）。
///
/// 移植自 Python `GravityField.compute_acceleration` 。
///
/// # 参数
/// - `et` ：SPICE et 秒
/// - `r_sc`: 航天器位置 [x, y, z] km（propagation frame 下，通常 J2000 地心）
/// - `c_flat`/`s_flat` ：球谐系数 (degree+1)² 长度
/// - `mu`/`radius`/`degree`/`order` ：球谐参数
/// - `input_frame` ：body-fixed frame 名（"ITRF93"/"MOON_PA"）
/// - `propagation_frame` ：传播 frame 名（通常 "J2000"）
/// - `body` ：中心天体名（"EARTH"/"MOON"）
/// - `tide_mode` ：0=None, 1=Solid, 2=SolidAndPole（Pole 档暂不支持，回退 Python）
/// - `k_love_flat` ：Love 数表 5×5 行优先
/// - `k_plus_flat` ：弹性 Love 数 5 元素或空
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gravity_field_acceleration(
    et: f64,
    r_sc: Vec<f64>,
    c_flat: Vec<f64>,
    s_flat: Vec<f64>,
    mu: f64,
    radius: f64,
    degree: usize,
    order: usize,
    input_frame: &str,
    propagation_frame: &str,
    body: &str,
    propagation_origin: &str,
    tide_mode: usize,
    k_love_flat: Vec<f64>,
    k_plus_flat: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    if r_sc.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "r_sc must have length 3, got {}",
            r_sc.len()
        )));
    }
    let mode = match tide_mode {
        0 => e2m2e_forces::forces::gravity_field::TideMode::None,
        1 => e2m2e_forces::forces::gravity_field::TideMode::Solid,
        2 => e2m2e_forces::forces::gravity_field::TideMode::SolidAndPole,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tide_mode must be 0/1/2, got {}",
                tide_mode
            )))
        }
    };
    let tide = e2m2e_forces::forces::gravity_field::TideConfig {
        mode,
        k_love_flat,
        k_plus_flat,
    };
    let r_arr = [r_sc[0], r_sc[1], r_sc[2]];
    let a = e2m2e_forces::forces::gravity_field::gravity_field_acceleration(
        et,
        &r_arr,
        &c_flat,
        &s_flat,
        mu,
        radius,
        degree,
        order,
        input_frame,
        propagation_frame,
        body,
        propagation_origin,
        &tide,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "gravity_field_acceleration failed: {:?}",
            e
        ))
    })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// SRP 加速度（含阴影）。
///
/// 移植自 Python `SolarRadiationPressure.compute_acceleration` 。
///
/// # 参数
/// - `et` ：SPICE et 秒
/// - `sc_pos` ：航天器位置 [x, y, z] km（observer 系下）
/// - `area`/`mass`/`cr` ：SRP cannonball 参数（area m²、mass kg、cr 无量纲）
/// - `shadow_bodies` ：遮挡体名称列表（如 ["EARTH", "MOON"]），空 = 无阴影
/// - `observer` ：观察者天体（通常 "EARTH"）
#[cfg(feature = "spice")]
#[pyfunction]
pub fn srp_acceleration(
    et: f64,
    sc_pos: Vec<f64>,
    area: f64,
    mass: f64,
    cr: f64,
    shadow_bodies: Vec<String>,
    observer: &str,
) -> PyResult<Vec<f64>> {
    if sc_pos.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sc_pos must have length 3, got {}",
            sc_pos.len()
        )));
    }
    let pos_arr = [sc_pos[0], sc_pos[1], sc_pos[2]];
    let a = e2m2e_forces::forces::srp::srp_acceleration(
        et,
        &pos_arr,
        area,
        mass,
        cr,
        &shadow_bodies,
        observer,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("srp_acceleration failed: {:?}", e))
    })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// NRLMSISE-00 大气密度与温度查询。
///
/// 供 Python `NRLMSISE00Atmosphere.density` 调用（ADR 0030：数值只在 Rust）。
/// `epoch_et` 优先经星历预采样缓存（`ephem_cache::lookup_utc_calendar`）折成年积日
/// 与 UTC 日内秒，缓存未启用时回退 `et2utc`（需装载 leapsecond 内核）——因此在
/// 打靶/分段积分的并行区（`StrictGuard`）内不触发 cspice。
/// `ap` 长度必须为 7：平坦（7 个元素全等）时按静态空间天气处理，否则启用
/// 3 小时分辨率的 Ap 史先验。
///
/// # 参数
/// - `epoch_et` ：SPICE et 秒
/// - `altitude_km`/`geodetic_lat_deg`/`geodetic_lon_deg` ：WGS84 大地坐标
/// - `f107_daily`/`f107_avg` ：前一日 F10.7 与 81 日滑动平均（sfu）
/// - `ap` ：3 小时分辨率地磁 Ap 指数史（7 元）
///
/// # 返回
/// `(密度 kg/m³, 温度 K)`。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn nrlmsise00_density_py(
    epoch_et: f64,
    altitude_km: f64,
    geodetic_lat_deg: f64,
    geodetic_lon_deg: f64,
    f107_daily: f64,
    f107_avg: f64,
    ap: Vec<f64>,
) -> PyResult<(f64, f64)> {
    let ap: [f64; 7] = ap.as_slice().try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "nrlmsise00 ap must have 7 elements, got {}",
            ap.len()
        ))
    })?;
    let (_year, day_of_year, ut_seconds) = e2m2e_forces::nrlmsise00::et_to_utc_doy(epoch_et)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("nrlmsise00 et2utc failed: {e}"))
        })?;
    // Ap 史非平坦才启用暴时先验（与 Rust drag 路径同一实现）。
    let storm_time = e2m2e_forces::nrlmsise00::storm_time_from_ap(&ap);
    let input = e2m2e_forces::nrlmsise00::Nrlmsise00Input {
        day_of_year,
        ut_seconds,
        altitude_km,
        geodetic_lat_deg,
        geodetic_lon_deg,
        f107_daily,
        f107_avg,
        ap,
    };
    let out = e2m2e_forces::nrlmsise00::density(&input, storm_time);
    Ok((out.density_kg_m3, out.temperature_k))
}
