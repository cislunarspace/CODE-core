//! e2m2e 积分器 crate：单步 Runge-Kutta、多步 Adams 与二阶 Cowell 的 Rust 实现，
//! 通过 PyO3 暴露给 Python。
//!
//! 三种积分族共享同一入口风格：给定当前状态与右端项，推进一步并返回
//! 误差估计与步长建议。具体族的区别见 [`crate::butcher`]、[`crate::abm`]、
//! [`crate::cowell`] 的模块文档。

use pyo3::prelude::*;
use pyo3::types::PyList;
#[cfg(feature = "spice")]
use pyo3::types::{PyDict, PyTuple};

pub(crate) mod abm;
pub(crate) mod butcher;
pub(crate) mod cowell;
#[cfg(feature = "spice")]
pub(crate) mod forces;
#[cfg(feature = "spice")]
pub mod multiple_shooting;
pub(crate) mod multistep_methods;
pub(crate) mod pd45;
pub(crate) mod pd78;
pub(crate) mod rk89;
pub mod rk_methods;
#[cfg(feature = "spice")]
pub mod segmented_shooting;
#[cfg(feature = "spice")]
pub mod single_shooting;
pub(crate) mod solid_tide;
pub mod solve_ivp;
pub(crate) mod spherical_harmonic;
#[cfg(feature = "spice")]
pub(crate) mod spice_ffi;
#[cfg(feature = "spice")]
pub(crate) mod spk_accel;

use butcher::{explicit_rk_step, suggest_next_step};
use multistep_methods::MultistepMethod;
use rk_methods::RkMethod;

/// 单步 Runge-Kutta 的结果。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct StepResult {
    pub y_new: Vec<f64>,
    pub error: f64,
    pub h_next: f64,
}

#[pymethods]
impl StepResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let y_new = PyList::new(py, &self.y_new)?;
        Ok(format!(
            "StepResult(y_new={y_new:?}, error={:.3e}, h_next={:.3e})",
            self.error, self.h_next
        ))
    }
}

/// 单步多步的结果。携带滚动后的历史缓冲，供 Python 传播循环传入下一步调用。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct MultistepResult {
    pub y_new: Vec<f64>,
    pub error: f64,
    pub h_next: f64,
    pub history: Vec<Vec<f64>>,
}

#[pymethods]
impl MultistepResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let y_new = PyList::new(py, &self.y_new)?;
        Ok(format!(
            "MultistepResult(y_new={y_new:?}, error={:.3e}, h_next={:.3e})",
            self.error, self.h_next
        ))
    }
}

/// 单步 Cowell (Störmer-Cowell) 的结果。`x_new` 仅含位置；
/// 历史缓冲混合位置与加速度采样（见 `cowell_step`）。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct CowellResult {
    pub x_new: Vec<f64>,
    pub error: f64,
    pub h_next: f64,
    pub history: Vec<Vec<f64>>,
}

#[pymethods]
impl CowellResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let x_new = PyList::new(py, &self.x_new)?;
        Ok(format!(
            "CowellResult(x_new={x_new:?}, error={:.3e}, h_next={:.3e})",
            self.error, self.h_next
        ))
    }
}

/// 调用 Python 右端项回调，校验返回值长度。
fn call_python_rhs(f: &Bound<PyAny>, n: usize, t: f64, y: &[f64]) -> PyResult<Vec<f64>> {
    let py = f.py();
    let yi_list = PyList::new(py, y)?;
    let result = f.call1((t, yi_list))?;
    let vals: Vec<f64> = result.extract()?;

    if vals.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "callback returned {} values but state vector has {} elements",
            vals.len(),
            n
        )));
    }

    Ok(vals)
}

/// 执行一次显式 Runge-Kutta 单步。
///
/// ``state_error_dim``：步长误差控制只统计前 N 维（``None`` 时统计全部）。
/// 用于 STM 增广传播——物理状态占前 6 维，STM 展平占后 36 维，后者不应
/// 主导步长控制。
#[pyfunction]
#[pyo3(signature = (method, t, y, h, tol, f, state_error_dim=None))]
fn rk_step(
    method: RkMethod,
    t: f64,
    y: Vec<f64>,
    h: f64,
    tol: f64,
    f: &Bound<PyAny>,
    state_error_dim: Option<usize>,
) -> PyResult<StepResult> {
    if h <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step size h must be positive",
        ));
    }
    if tol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance tol must be positive",
        ));
    }
    if y.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state vector y must not be empty",
        ));
    }
    if let Some(dim) = state_error_dim {
        if dim == 0 || dim > y.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_error_dim must be in 1..={}, got {}",
                y.len(),
                dim
            )));
        }
    }

    let n = y.len();
    let callback = |ti: f64, yi: &[f64]| -> PyResult<Vec<f64>> { call_python_rhs(f, n, ti, yi) };

    let table = method.table();
    let (y_new, error) = explicit_rk_step(table, t, &y, h, callback, state_error_dim)?;

    if y_new.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step produced non-finite values",
        ));
    }

    let h_next = suggest_next_step(h, error, tol, method.embedded_order());

    Ok(StepResult {
        y_new,
        error,
        h_next,
    })
}

/// 执行一次多步预测-校正单步。
#[pyfunction]
fn multistep_step(
    method: MultistepMethod,
    t: f64,
    y: Vec<f64>,
    h: f64,
    tol: f64,
    f: &Bound<PyAny>,
    history: Vec<Vec<f64>>,
) -> PyResult<MultistepResult> {
    if h <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step size h must be positive",
        ));
    }
    if tol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance tol must be positive",
        ));
    }
    if y.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state vector y must not be empty",
        ));
    }
    let steps = method.steps();
    if history.len() != steps {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "multistep method needs {steps} history samples, got {}",
            history.len()
        )));
    }
    let n = y.len();
    for hist in &history {
        if hist.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "history sample has length {} but state vector has length {n}",
                hist.len()
            )));
        }
    }

    let callback = |ti: f64, yi: &[f64]| -> PyResult<Vec<f64>> { call_python_rhs(f, n, ti, yi) };

    let (y_new, error, new_history) = match method {
        MultistepMethod::Abm => abm::abm_step(t, &y, h, &history, callback)?,
    };

    if y_new.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step produced non-finite values",
        ));
    }

    let h_next = suggest_next_step(h, error, tol, method.embedded_order());

    Ok(MultistepResult {
        y_new,
        error,
        h_next,
        history: new_history,
    })
}

/// 执行一次 Cowell (Störmer-Cowell) 8 阶单步，用于 x'' = a(t, x)。
///
/// `history` = `[x_{n−1}, x_n, a_{n−7}, ..., a_n]`（10 个向量：2 个位置采样
/// + 8 个加速度采样，由旧到新）。`accel` 计算 a(t, x)。
/// 输出仅含位置。固定步长；改变 `h` 需重新初始化历史缓冲。
#[pyfunction]
fn cowell_step(
    t: f64,
    h: f64,
    tol: f64,
    accel: &Bound<PyAny>,
    history: Vec<Vec<f64>>,
) -> PyResult<CowellResult> {
    if h <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step size h must be positive",
        ));
    }
    if tol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance tol must be positive",
        ));
    }
    if history.len() != cowell::COWELL_HISTORY_LEN {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cowell needs {} history samples [x_(n-1), x_n, a_(n-7), ..., a_n] \
             (2 positions + 8 accelerations), got {}",
            cowell::COWELL_HISTORY_LEN,
            history.len()
        )));
    }
    let n = history[0].len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "position dimension must be positive",
        ));
    }
    for hist in &history {
        if hist.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "history sample has length {} but position dimension is {n}",
                hist.len()
            )));
        }
    }

    let callback =
        |ti: f64, xi: &[f64]| -> PyResult<Vec<f64>> { call_python_rhs(accel, n, ti, xi) };

    let (x_new, error, new_history) = cowell::cowell_step(t, h, &history, callback)?;

    if x_new.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step produced non-finite values",
        ));
    }

    let h_next = suggest_next_step(h, error, tol, cowell::COWELL_EMBEDDED_ORDER);

    Ok(CowellResult {
        x_new,
        error,
        h_next,
        history: new_history,
    })
}

/// 占位函数，用于验证 FFI 路径端到端通畅。
#[pyfunction]
fn hello_integrators() -> PyResult<String> {
    Ok("hello from e2m2e-integrators".to_string())
}

/// 球谐引力加速度（body-fixed 系）。
///
/// Python 侧 `GravityField._compute_acceleration_in_input_frame` 的 Rust 加速版。
/// 输入位置 `r` 与输出加速度均在 body-fixed 系（坐标变换仍由 Python 完成）。
/// `c_flat`/`s_flat` 是 C/S 系数矩阵的行优先扁平化（shape=(degree+1)**2）。
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

/// 固体潮 Step 1（频率无关，天体无关）。
///
/// Python 侧 `earth_tide.solid_tide_step1` 的 Rust 加速版。输入扰动体位置由
/// Python 完成坐标变换后传入（本函数不查 SPICE）。
///
/// # 参数
/// - `perturbers_flat`：扁平化扰动体列表，每 4 个一组 `[px, py, pz, gm]`（位置 km、
///   gm km³/s²）。长度必须是 4 的倍数。
/// - `k_love_flat`：Love 数表 5×5 行优先扁平化，长度 25。
/// - `k_plus_flat`：弹性 Love 数 5 元素，或 `None`（无贡献）。
/// - `mu_central`、`r_central`：中心天体 GM 与参考半径。
///
/// # 返回
/// 长度 50 的 `Vec<f64>`：`C(25) ++ S(25)`，各为 5×5 行优先扁平化。
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

/// 固体潮 Step 2（频率相关，地球专用）。返回长度 50 的 `Vec<f64>`（C25 + S25）。
#[pyfunction]
fn solid_tide_step2(et: f64) -> PyResult<Vec<f64>> {
    Ok(solid_tide::solid_tide_step2(et))
}

/// 极潮（固体极潮 + 海洋极潮，IERS TN32）。返回长度 50 的 `Vec<f64>`（C25 + S25）。
#[pyfunction]
fn pole_tide(et: f64, xp: f64, yp: f64) -> PyResult<Vec<f64>> {
    Ok(solid_tide::pole_tide(et, xp, yp))
}

/// PoC：通过 cspice 查询 `target` 相对 `observer` 在 J2000 系下的位置（km）。
///
/// 用于验证：
/// 1. maturin + cspice 链路是否正常
/// 2. Python spiceypy 已 furnsh 的内核池是否对 Rust cspice 可见（共享内核池）
///
/// 仅在 `spice` feature 下编译。返回长度 3 的 `Vec<f64>`。
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
///
/// Rust cspice 与 Python spiceypy 是**独立的 CSPICE 实例**（静态链接，全局状态
/// 不共享）。Python 侧 furnsh 的内核，Rust 看不见；反之亦然。要让 Rust 查询
/// 可用，必须用本函数在 Rust 侧再 furnsh 一次（同一份文件，两边独立加载）。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_poc_furnsh(path: &str) -> PyResult<()> {
    cspice::data::furnish(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("furnsh failed: {:?}", e)))
}

/// 第三体摄动加速度（含直接项 + 间接项）。
///
/// 移植自 Python `ThirdBodyGravity.compute_acceleration`。一次调用完成
/// "cspice 查扰动体位置 + 加速度公式"，消除 Python↔cspice 跨界 + numpy
/// 数组分配开销。
///
/// # 参数
/// - `et`：SPICE et 秒（past J2000 TDB）
/// - `target`：摄动天体名（"MOON"/"SUN"/"5"=JUPITER 等）
/// - `observer`：原点天体名（通常 "EARTH"）
/// - `sc_pos`：航天器位置 [x, y, z] km（相对 observer），长度 3
/// - `mu`：摄动天体 GM（km³/s²）
///
/// # 返回
/// 长度 3 的加速度 `Vec<f64>`，单位 km/s²。
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
///
/// 移植自 Python `IndirectTerm.compute_acceleration`。
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

/// GravityField 完整加速度（含坐标变换 + 球谐 + 潮汐）。
///
/// 移植自 Python `GravityField.compute_acceleration`。
///
/// # 参数
/// - `et`：SPICE et 秒
/// - `r_sc`: 航天器位置 [x, y, z] km（propagation frame 下，通常 J2000 地心）
/// - `c_flat`/`s_flat`：球谐系数 (degree+1)² 长度
/// - `mu`/`radius`/`degree`/`order`：球谐参数
/// - `input_frame`：body-fixed frame 名（"ITRF93"/"MOON_PA"）
/// - `propagation_frame`：传播 frame 名（通常 "J2000"）
/// - `body`：中心天体名（"EARTH"/"MOON"）
/// - `tide_mode`：0=None, 1=Solid, 2=SolidAndPole（Pole 档暂不支持，回退 Python）
/// - `k_love_flat`：Love 数表 5×5 行优先
/// - `k_plus_flat`：弹性 Love 数 5 元素或空
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gravity_field_acceleration(
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
        0 => forces::gravity_field::TideMode::None,
        1 => forces::gravity_field::TideMode::Solid,
        2 => forces::gravity_field::TideMode::SolidAndPole,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tide_mode must be 0/1/2, got {}",
                tide_mode
            )))
        }
    };
    let tide = forces::gravity_field::TideConfig {
        mode,
        k_love_flat,
        k_plus_flat,
    };
    let r_arr = [r_sc[0], r_sc[1], r_sc[2]];
    let a = forces::gravity_field::gravity_field_acceleration(
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
/// 移植自 Python `SolarRadiationPressure.compute_acceleration`。
///
/// # 参数
/// - `et`：SPICE et 秒
/// - `sc_pos`：航天器位置 [x, y, z] km（observer 系下）
/// - `area`/`mass`/`cr`：SRP cannonball 参数（area m²、mass kg、cr 无量纲）
/// - `shadow_bodies`：遮挡体名称列表（如 ["EARTH", "MOON"]），空 = 无阴影
/// - `observer`：观察者天体（通常 "EARTH"）
#[cfg(feature = "spice")]
#[pyfunction]
fn srp_acceleration(
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
    let a = forces::srp::srp_acceleration(et, &pos_arr, area, mass, cr, &shadow_bodies, observer)
        .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("srp_acceleration failed: {:?}", e))
    })?;
    Ok(vec![a[0], a[1], a[2]])
}

/// 解析单个 Python force 元组为 CompiledForce。
///
/// 元组格式（首元素是 type 标签）：
/// - `("gravity", c_flat, s_flat, mu, radius, degree, order, input_frame,
///     propagation_frame, body, propagation_origin, tide_mode, k_love_flat,
///     k_plus_flat_or_none)`
/// - `("third_body", body, mu)`
/// - `("indirect", body, mu)`
/// - `("srp", area, mass, cr, shadow_bodies_list)`
#[cfg(feature = "spice")]
pub(crate) fn parse_force_tuple(
    item: &Bound<'_, PyAny>,
) -> PyResult<forces::compiled::CompiledForce> {
    use forces::compiled::CompiledForce;
    use forces::gravity_field::TideMode;

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
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown force tag {:?}",
            tag
        ))),
    }
}

/// 全 Rust 力模型传播器（消除 Python↔Rust 跨界）。
///
/// Python 侧把所有 force 序列化为元组列表，Rust 在内部循环里直接调
/// `compute_total_acceleration`，每个 RK 子阶段不再跨界回 Python。
///
/// # 参数
/// - `method`: RkMethod
/// - `t0`/`y0`: 初始时刻与状态
/// - `h_init`: 初始步长
/// - `tol`: 容差
/// - `t_eval`: 评估时刻数组
/// - `observer`: 传播系 origin（如 "EARTH"）
/// - `forces_py`: force 元组列表
/// - `max_steps`: 最大步数
///
/// # 返回
/// Python dict：`{"time": [...], "states": [[...]], "n_steps": int, "n_rejected": int}`
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn propagate_compiled(
    method: RkMethod,
    t0: f64,
    y0: Vec<f64>,
    h_init: f64,
    tol: f64,
    t_eval: Vec<f64>,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    max_steps: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use forces::compiled::{compute_total_acceleration, CompiledForce};

    if y0.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y0 must have length 6, got {}",
            y0.len()
        )));
    }
    if tol <= 0.0 || h_init <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tol and h_init must be positive",
        ));
    }

    // 解析 forces
    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    let table = method.table();
    let n = y0.len();
    let mut y = y0;
    let mut t = t0;
    let mut h = h_init;
    let mut times: Vec<f64> = vec![t0];
    let mut states: Vec<Vec<f64>> = vec![y.clone()];
    let mut eval_idx = 1usize; // t_eval[0] == t0 已经记录
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;

    // 用 RefCell 包装 cspice 错误状态（不能直接通过 explicit_rk_step 的 E 传）
    use std::cell::RefCell;
    let last_error: RefCell<Option<String>> = RefCell::new(None);

    while t < t_eval[t_eval.len() - 1] && n_steps < max_steps {
        n_steps += 1;
        // 限制步长不超过下一个评估点（提高 t_eval 命中率）
        if eval_idx < t_eval.len() {
            let t_next_eval = t_eval[eval_idx];
            if t + h > t_next_eval {
                h = t_next_eval - t;
            }
        }

        // RK 单步：用 Rust 闭包调 compute_total_acceleration
        let forces_ref = &forces;
        let observer_ref = observer;
        let err_cell = &last_error;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let state6 = [yi[0], yi[1], yi[2], yi[3], yi[4], yi[5]];
            compute_total_acceleration(forces_ref, ti, &state6, observer_ref)
                .map(|a| vec![yi[3], yi[4], yi[5], a[0], a[1], a[2]])
                .map_err(|e| {
                    *err_cell.borrow_mut() = Some(e.clone());
                    e
                })
        };

        let (y_new, error) = match explicit_rk_step(table, t, &y, h, callback, None) {
            Ok(r) => r,
            Err(msg) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "RK step force error: {}",
                    msg
                )));
            }
        };
        let _ = n;

        if error <= tol {
            t += h;
            y = y_new;
            // 输出落在 t_eval 的点
            while eval_idx < t_eval.len() && t >= t_eval[eval_idx] - 1e-9 {
                times.push(t_eval[eval_idx]);
                states.push(y.clone());
                eval_idx += 1;
            }
            let h_next = suggest_next_step(h, error, tol, method.embedded_order());
            h = h_next;
        } else {
            n_rejected += 1;
            let h_next = suggest_next_step(h, error, tol, method.embedded_order());
            h = h_next;
            if h < 1e-12 * (t_eval[t_eval.len() - 1] - t0).abs() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "step size collapsed below minimum",
                ));
            }
        }
    }

    // 返回 dict
    let dict = PyDict::new(py);
    dict.set_item("time", times)?;
    dict.set_item("states", states)?;
    dict.set_item("n_steps", n_steps)?;
    dict.set_item("n_rejected", n_rejected)?;
    Ok(dict.into())
}

/// Python 接口：42 维增广状态传播（状态 + STM）。
///
/// 纯 N 体模型（EARTH/MOON/SUN 等），用于星历修正的逐段积分。
/// 调用 `e2m2e-forces` 的 `propagate_with_stm`（DOP853 + STM 变分方程）。
///
/// # 参数
/// - `bodies`: 天体名称列表（如 `["EARTH", "MOON", "SUN"]`）
/// - `origin`: 原点天体名称（如 `"EARTH"`）
/// - `gm_values`: 各天体的 GM（km³/s²），与 `bodies` 一一对应
/// - `t_span`: `(t_start, t_end)` 积分区间（SPICE et 秒）
/// - `t_eval`: 输出时间点数组
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`（km, km/s）
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长（秒），`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...]}`
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (bodies, origin, gm_values, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_with_stm_py(
    bodies: Vec<String>,
    origin: String,
    gm_values: Vec<f64>,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::nbody_stm::{propagate_with_stm, NBodyConfig};

    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 6, got {}",
            initial_state.len()
        )));
    }
    if gm_values.len() != bodies.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gm_values length ({}) must match bodies length ({})",
            gm_values.len(),
            bodies.len()
        )));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }

    let config = NBodyConfig {
        bodies,
        origin,
        gm_values,
    };

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = propagate_with_stm(
        &config, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("STM propagation failed: {}", e))
    })?;

    // 转为 Python 对象
    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();
    let stm_list: Vec<Vec<f64>> = result.stms.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("stm", stm_list)?;
    dict.set_item("time", result.times)?;
    Ok(dict.into())
}

/// 编译型力模型 + STM 的 PD45 传播（消除 cspice 隔离）。
///
/// 与 `propagate_with_stm_py`（纯 NBody）不同，本函数支持所有编译型力模型：
/// PointMass、GravityField、ThirdBody、IndirectTerm、SRP、Relativistic。
/// 使用 integrators crate 的 cspice 实例，避免跨 .so 内核池隔离问题。
///
/// # 参数
/// - `observer`: 传播系 origin 天体名（如 "EARTH"）
/// - `forces_py`: force 元组列表（格式同 `propagate_compiled`）
/// - `t_span`: `(t_start, t_end)` 积分区间（SPICE et 秒）
/// - `t_eval`: 输出时间点数组
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`（km, km/s）
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长（秒），`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...],
///                "n_steps": int, "n_rejected": int}`
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces_py, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_compiled_stm_py(
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use forces::compiled::CompiledForce;
    use forces::compiled_stm::propagate_compiled_stm;

    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 6, got {}",
            initial_state.len()
        )));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }

    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = propagate_compiled_stm(
        &forces, observer, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("STM propagation failed: {}", e))
    })?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();
    let stm_list: Vec<Vec<f64>> = result.stms.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("stm", stm_list)?;
    dict.set_item("time", result.times)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    Ok(dict.into())
}

#[pymodule]
fn _integrators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_integrators, m)?)?;
    m.add_function(wrap_pyfunction!(rk_step, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ivp::solve_ivp_py, m)?)?;
    m.add_function(wrap_pyfunction!(multistep_step, m)?)?;
    m.add_function(wrap_pyfunction!(cowell_step, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_accel, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step1, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step2, m)?)?;
    m.add_function(wrap_pyfunction!(pole_tide, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_poc_body_position, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_poc_furnsh, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(third_body_acceleration, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(indirect_term_acceleration, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(gravity_field_acceleration, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(srp_acceleration, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_compiled, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_with_stm_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_compiled_stm_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        multiple_shooting::multiple_shooting_correct_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_class::<multiple_shooting::MultipleShootingRustResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        segmented_shooting::segmented_shooting_correct_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_class::<segmented_shooting::SegmentedShootingResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        segmented_shooting::segmented_shooting_correct_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_class::<segmented_shooting::SegmentedShootingResult>()?;
    m.add_class::<RkMethod>()?;
    m.add_class::<MultistepMethod>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<MultistepResult>()?;
    m.add_class::<CowellResult>()?;
    Ok(())
}
