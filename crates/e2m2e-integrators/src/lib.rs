//! e2m2e 积分器 crate：单步 Runge-Kutta、多步 Adams 与二阶 Cowell 的 Rust 实现，
//! 通过 PyO3 暴露给 Python。
//!
//! 三种积分族共享同一入口风格：给定当前状态与右端项，推进一步并返回
//! 误差估计与步长建议。具体族的区别见 [`crate::butcher`]、[`crate::abm`]、
//! [`crate::cowell`] 的模块文档。

use pyo3::prelude::*;
use pyo3::types::PyList;

pub(crate) mod abm;
pub(crate) mod butcher;
pub(crate) mod cowell;
pub(crate) mod multistep_methods;
pub(crate) mod pd45;
pub(crate) mod pd78;
pub(crate) mod rk89;
pub(crate) mod solid_tide;
pub(crate) mod spherical_harmonic;
pub mod rk_methods;

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

    Ok(StepResult { y_new, error, h_next })
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

    let callback = |ti: f64, xi: &[f64]| -> PyResult<Vec<f64>> { call_python_rhs(accel, n, ti, xi) };

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
    if perturbers_flat.len() % 4 != 0 {
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
    Ok(vec![
        state.position.x,
        state.position.y,
        state.position.z,
    ])
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

/// PyO3 模块初始化函数，将 Rust 函数与类注册到 Python 模块。
#[pymodule]
fn _integrators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_integrators, m)?)?;
    m.add_function(wrap_pyfunction!(rk_step, m)?)?;
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
    m.add_class::<RkMethod>()?;
    m.add_class::<MultistepMethod>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<MultistepResult>()?;
    m.add_class::<CowellResult>()?;
    Ok(())
}
