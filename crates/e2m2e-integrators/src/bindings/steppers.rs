//! 单步/多步/Cowell 步进器与 solve_ivp 家族的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::multistep_methods::MultistepMethod;
use e2m2e_propagation::rk_methods::RkMethod;
use e2m2e_propagation::{abm, cowell};

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
/// 历史缓冲混合位置与加速度采样（见 `cowell_step` ）。
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
/// ``state_error_dim`` ：步长误差控制只统计前 N 维（``None`` 时统计全部）。
/// 用于 STM 增广传播：物理状态占前 6 维，STM 展平占后 36 维，后者不应
/// 主导步长控制。
#[pyfunction]
#[pyo3(signature = (method, t, y, h, tol, f, state_error_dim=None))]
pub fn rk_step(
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
pub fn multistep_step(
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
/// `history` = `[x_{n−1}, x_n, a_{n−7}, ..., a_n]` （10 个向量：2 个位置采样
/// + 8 个加速度采样，由旧到新）。`accel` 计算 a(t, x)。
/// 输出仅含位置。固定步长；改变 `h` 需重新初始化历史缓冲。
#[pyfunction]
pub fn cowell_step(
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

/// Python 接口：完整自适应步长 ODE 积分器（scipy `solve_ivp` 等价物）。
///
/// 使用 DOP853 (Prince-Dormand 8(7)13M) 方法。纯 Rust 积分循环在
/// `e2m2e_propagation::solve_ivp` ，本函数只做 Python 回调适配与结果封装。
///
/// # 参数
/// - `t_span`: `(t_start, t_end)` 积分区间
/// - `y0`: 初始状态向量
/// - `t_eval`: 输出时间点数组
/// - `rtol`: 相对容差
/// - `atol`: 绝对容差
/// - `f`: Python callable `f(t, y) -> dy/dt`
/// - `max_step`: 最大步长（默认 `f64::INFINITY` ）
/// - `max_steps`: 最大步数（默认 `MAX_ADAPTIVE_STEPS` ）
/// - `state_error_dim`: 步长误差控制只统计前 N 维（用于 STM 增广传播）
///
/// # 返回
/// Python dict：`{"states": [[...]], "time": [...], "n_steps": int}`
#[pyfunction]
#[pyo3(signature = (t_span, y0, t_eval, rtol, atol, f, max_step=None, max_steps=None, state_error_dim=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_py(
    t_span: (f64, f64),
    y0: Vec<f64>,
    t_eval: Vec<f64>,
    rtol: f64,
    atol: f64,
    f: &Bound<PyAny>,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    state_error_dim: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::pd78::PD78_TABLE as DOP853;
    use e2m2e_propagation::solve_ivp::{solve_ivp_impl, MAX_ADAPTIVE_STEPS};

    if y0.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y0 must not be empty",
        ));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rtol and atol must be positive",
        ));
    }

    let n = y0.len();
    let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
        call_python_rhs(f, n, ti, yi).map_err(|e| e.to_string())
    };

    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(MAX_ADAPTIVE_STEPS);

    let states = solve_ivp_impl(
        &DOP853,
        callback,
        t_span,
        &y0,
        &t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        state_error_dim,
    );

    let n_steps = states.len(); // 近似：实际步数 ≥ 输出点数

    // 构造输出时间戳
    let out_times: Vec<f64> = t_eval[..states.len()].to_vec();

    let dict = PyDict::new(py);
    dict.set_item("states", states)?;
    dict.set_item("time", out_times)?;
    dict.set_item("n_steps", n_steps)?;
    Ok(dict.into())
}

/// Python 接口：带事件检测的自适应步长 ODE 积分器。
///
/// 事件检测在 Rust 积分内循环完成：每个接受步的端点评估事件函数，
/// 符号变化（经 direction 过滤）时在步内对线性插值态二分求精（无稠密输出）。
///
/// **参数**
///
/// - ``events``: ``[(callable, terminal, direction), ...]`` ，callable 为
///   ``g(t, y) -> float`` ；``terminal=True`` 触发即停；``direction`` > 0 只记
///   上行穿越、< 0 只记下行、0 双向（scipy ``solve_ivp`` 语义）
/// - ``method``: RK 方法（默认 ``Pd78`` ，即 DOP853）
/// - 其余参数同 ``solve_ivp_py``
///
/// **返回**
///
/// Python dict：``{"states", "time", "n_steps", "t_events", "y_events", "terminal_event"}`` ；
/// terminal 截断时 ``time``/``states`` 末点为求精后的事件点，
/// ``terminal_event`` 为触发终止的事件索引（未终止为 None）。
#[pyfunction]
#[pyo3(signature = (t_span, y0, t_eval, rtol, atol, f, events, method=None, max_step=None, max_steps=None, state_error_dim=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_events_py<'py>(
    t_span: (f64, f64),
    y0: Vec<f64>,
    t_eval: Vec<f64>,
    rtol: f64,
    atol: f64,
    f: &Bound<'py, PyAny>,
    events: Vec<(Bound<'py, PyAny>, bool, f64)>,
    method: Option<RkMethod>,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    state_error_dim: Option<usize>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::solve_ivp::{solve_ivp_events_impl, EventSpec, MAX_ADAPTIVE_STEPS};

    if y0.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y0 must not be empty",
        ));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rtol and atol must be positive",
        ));
    }

    let n = y0.len();
    let rhs = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
        call_python_rhs(f, n, ti, yi).map_err(|e| e.to_string())
    };

    type PyEvent<'a> = Box<dyn Fn(f64, &[f64]) -> Result<f64, String> + 'a>;
    let specs: Vec<EventSpec<PyEvent<'py>>> = events
        .into_iter()
        .map(|(g, terminal, direction)| {
            let closure: PyEvent<'py> =
                Box::new(move |ti: f64, yi: &[f64]| -> Result<f64, String> {
                    let yi_list = PyList::new(g.py(), yi).map_err(|e| e.to_string())?;
                    let value = g.call1((ti, yi_list)).map_err(|e| e.to_string())?;
                    value.extract::<f64>().map_err(|e| e.to_string())
                });
            EventSpec::new(closure, terminal, direction)
        })
        .collect();

    let table = method.unwrap_or(RkMethod::Pd78).table();
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(MAX_ADAPTIVE_STEPS);

    let result = solve_ivp_events_impl(
        table,
        rhs,
        t_span,
        &y0,
        &t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        state_error_dim,
        &specs,
    );

    let dict = PyDict::new(py);
    dict.set_item("states", result.states)?;
    dict.set_item("time", result.t)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("t_events", result.t_events)?;
    dict.set_item("y_events", result.y_events)?;
    dict.set_item("terminal_event", result.terminal_event)?;
    Ok(dict.into())
}

/// 事件路径内核 RHS 闭包类型：`(t, y) -> Result<dy/dt, String>`。纯 Rust 闭包
/// 不会失败，`Err` 分支不可达；错误类型取 `String` 与 Python 事件回调闭包在
/// `solve_ivp_events_impl` 的统一 `E` 上对齐。
type KernelRhs = Box<dyn Fn(f64, &[f64]) -> Result<Vec<f64>, String> + Send>;

/// 事件路径内核分派：由动力学标识 + 参数表构造纯 Rust RHS 闭包（issue #594）。
///
/// EOM/STM 复用 `e2m2e-forces` 的 CR3BP/BCR4BP 实现（与 `propagate_cr3bp_stm`
/// / `propagate_bcr4bp_stm` 背后同一内核），不引入第二份 EOM；`A·Φ` 矩阵乘
/// 复用 `e2m2e_forces::cr3bp::stm_derivative`。
///
/// 返回 `(增广状态维数, RHS 闭包)`；维数供调用方校验 `y0` 长度。
pub(crate) fn build_event_kernel_rhs(
    kernel: &str,
    params: &HashMap<String, f64>,
) -> Result<(usize, KernelRhs), String> {
    let required = |key: &str| {
        params
            .get(key)
            .copied()
            .ok_or_else(|| format!("kernel {kernel} missing required param `{key}`"))
    };
    match kernel {
        "cr3bp" => {
            let mu = required("mu")?;
            let rhs = move |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
                let mut state = [0.0_f64; 6];
                state.copy_from_slice(y);
                Ok(e2m2e_forces::cr3bp::cr3bp_eom(mu, &state).to_vec())
            };
            Ok((6, Box::new(rhs)))
        }
        "cr3bp-with-stm" => {
            let mu = required("mu")?;
            let rhs = move |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
                let mut state = [0.0_f64; 6];
                state.copy_from_slice(&y[..6]);
                let mut stm = [0.0_f64; 36];
                stm.copy_from_slice(&y[6..]);
                let a = e2m2e_forces::cr3bp::cr3bp_jacobian_6x6(mu, &state);
                let mut out = Vec::with_capacity(42);
                out.extend_from_slice(&e2m2e_forces::cr3bp::cr3bp_eom(mu, &state));
                out.extend_from_slice(&e2m2e_forces::cr3bp::stm_derivative(&a, &stm));
                Ok(out)
            };
            Ok((42, Box::new(rhs)))
        }
        "bcr4bp" => {
            let mu = required("mu")?;
            let mu_sun = required("mu_sun")?;
            let sun_distance = required("sun_distance")?;
            let sun_angular_rate = required("sun_angular_rate")?;
            let sun_phase0 = required("sun_phase0")?;
            let rhs = move |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
                let mut state = [0.0_f64; 6];
                state.copy_from_slice(y);
                Ok(e2m2e_forces::bcr4bp::bcr4bp_eom(
                    mu,
                    mu_sun,
                    sun_distance,
                    sun_angular_rate,
                    sun_phase0,
                    &state,
                    t,
                )
                .to_vec())
            };
            Ok((6, Box::new(rhs)))
        }
        "bcr4bp-with-stm" => {
            let mu = required("mu")?;
            let mu_sun = required("mu_sun")?;
            let sun_distance = required("sun_distance")?;
            let sun_angular_rate = required("sun_angular_rate")?;
            let sun_phase0 = required("sun_phase0")?;
            let rhs = move |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
                let mut state = [0.0_f64; 6];
                state.copy_from_slice(&y[..6]);
                let mut stm = [0.0_f64; 36];
                stm.copy_from_slice(&y[6..]);
                let a = e2m2e_forces::bcr4bp::bcr4bp_jacobian_6x6(
                    mu,
                    mu_sun,
                    sun_distance,
                    sun_angular_rate,
                    sun_phase0,
                    &state,
                    t,
                );
                let mut out = Vec::with_capacity(42);
                out.extend_from_slice(&e2m2e_forces::bcr4bp::bcr4bp_eom(
                    mu,
                    mu_sun,
                    sun_distance,
                    sun_angular_rate,
                    sun_phase0,
                    &state,
                    t,
                ));
                out.extend_from_slice(&e2m2e_forces::cr3bp::stm_derivative(&a, &stm));
                Ok(out)
            };
            Ok((42, Box::new(rhs)))
        }
        other => Err(format!(
            "unknown EOM kernel `{other}` (supported: cr3bp, cr3bp-with-stm, bcr4bp, bcr4bp-with-stm)"
        )),
    }
}

/// Python 接口：带事件检测的自适应步长积分器，EOM 走 Rust 内置内核。
///
/// 与 [`solve_ivp_events_py`] 唯一的差别在力模型一侧：RHS 不再是 Python
/// 回调，而是由 `kernel`（动力学标识）+ `params`（参数表）分派到
/// `e2m2e-forces` 的 CR3BP/BCR4BP EOM/STM 内核，积分全程每步 RHS 求值
/// 留在 Rust 内（issue #594）。事件函数仍为 Python 回调，语义与返回
/// dict 同 [`solve_ivp_events_py`]。
///
/// **参数**
///
/// - `kernel`: 动力学标识，`"cr3bp"` / `"cr3bp-with-stm"` /
///   `"bcr4bp"` / `"bcr4bp-with-stm"`（with-stm 变体要求 42 维增广初值）
/// - `params`: 该动力学的无量纲参数表；`cr3bp` 需 `mu`，`bcr4bp` 另需
///   `mu_sun` / `sun_distance` / `sun_angular_rate` / `sun_phase0`
/// - 其余参数同 [`solve_ivp_events_py`]
#[pyfunction]
#[pyo3(signature = (t_span, y0, t_eval, rtol, atol, kernel, params, events, method=None, max_step=None, max_steps=None, state_error_dim=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_events_kernel_py<'py>(
    t_span: (f64, f64),
    y0: Vec<f64>,
    t_eval: Vec<f64>,
    rtol: f64,
    atol: f64,
    kernel: &str,
    params: HashMap<String, f64>,
    events: Vec<(Bound<'py, PyAny>, bool, f64)>,
    method: Option<RkMethod>,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    state_error_dim: Option<usize>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::solve_ivp::{solve_ivp_events_impl, EventSpec, MAX_ADAPTIVE_STEPS};

    if y0.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y0 must not be empty",
        ));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rtol and atol must be positive",
        ));
    }

    let (expected_dim, rhs) =
        build_event_kernel_rhs(kernel, &params).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if y0.len() != expected_dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y0 must have length {expected_dim} for kernel `{kernel}`, got {}",
            y0.len()
        )));
    }

    type PyEvent<'a> = Box<dyn Fn(f64, &[f64]) -> Result<f64, String> + 'a>;
    let specs: Vec<EventSpec<PyEvent<'py>>> = events
        .into_iter()
        .map(|(g, terminal, direction)| {
            let closure: PyEvent<'py> =
                Box::new(move |ti: f64, yi: &[f64]| -> Result<f64, String> {
                    let yi_list = PyList::new(g.py(), yi).map_err(|e| e.to_string())?;
                    let value = g.call1((ti, yi_list)).map_err(|e| e.to_string())?;
                    value.extract::<f64>().map_err(|e| e.to_string())
                });
            EventSpec::new(closure, terminal, direction)
        })
        .collect();

    let table = method.unwrap_or(RkMethod::Pd78).table();
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(MAX_ADAPTIVE_STEPS);

    let result = solve_ivp_events_impl(
        table,
        rhs,
        t_span,
        &y0,
        &t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        state_error_dim,
        &specs,
    );

    let dict = PyDict::new(py);
    dict.set_item("states", result.states)?;
    dict.set_item("time", result.t)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("t_events", result.t_events)?;
    dict.set_item("y_events", result.y_events)?;
    dict.set_item("terminal_event", result.terminal_event)?;
    Ok(dict.into())
}

#[cfg(test)]
mod event_kernel_tests {
    //! 事件路径内核分派（`build_event_kernel_rhs`）的单测：内核 RHS 与
    //! `e2m2e-forces` 的 EOM/STM 内核逐位一致、参数与标识校验生效。

    use super::build_event_kernel_rhs;
    use std::collections::HashMap;

    fn params(items: &[(&str, f64)]) -> HashMap<String, f64> {
        items
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect::<HashMap<_, _>>()
    }

    const MU: f64 = 0.012_150_585_609_624_04;
    const Y: [f64; 6] = [0.8, 0.05, 0.01, 0.02, -0.1, 0.03];

    #[test]
    fn cr3bp_kernel_is_exactly_the_forces_eom() {
        let (dim, rhs) = build_event_kernel_rhs("cr3bp", &params(&[("mu", MU)])).unwrap();
        assert_eq!(dim, 6);
        assert_eq!(
            rhs(0.0, &Y).unwrap(),
            e2m2e_forces::cr3bp::cr3bp_eom(MU, &Y).to_vec()
        );
    }

    #[test]
    fn cr3bp_with_stm_kernel_is_eom_with_a_times_phi() {
        let (dim, rhs) = build_event_kernel_rhs("cr3bp-with-stm", &params(&[("mu", MU)])).unwrap();
        assert_eq!(dim, 42);
        let mut stm = [0.0_f64; 36];
        for (i, v) in stm.iter_mut().enumerate() {
            *v = ((i * 7 + 3) % 11) as f64 * 0.1 - 0.5;
        }
        let mut y = Y.to_vec();
        y.extend_from_slice(&stm);
        let out = rhs(0.0, &y).unwrap();
        assert_eq!(out.len(), 42);
        assert_eq!(
            &out[..6],
            &e2m2e_forces::cr3bp::cr3bp_eom(MU, &Y).to_vec()[..]
        );
        let a = e2m2e_forces::cr3bp::cr3bp_jacobian_6x6(MU, &Y);
        assert_eq!(
            &out[6..],
            &e2m2e_forces::cr3bp::stm_derivative(&a, &stm).to_vec()[..]
        );
    }

    #[test]
    fn bcr4bp_kernel_is_exactly_the_forces_eom_including_time_dependence() {
        let p = params(&[
            ("mu", MU),
            ("mu_sun", 328900.5614),
            ("sun_distance", 389.17),
            ("sun_angular_rate", -0.925_195_966_551_2),
            ("sun_phase0", 0.3),
        ]);
        let (dim, rhs) = build_event_kernel_rhs("bcr4bp", &p).unwrap();
        assert_eq!(dim, 6);
        for t in [0.0_f64, 1.5, -2.25] {
            assert_eq!(
                rhs(t, &Y).unwrap(),
                e2m2e_forces::bcr4bp::bcr4bp_eom(
                    MU,
                    328900.5614,
                    389.17,
                    -0.925_195_966_551_2,
                    0.3,
                    &Y,
                    t
                )
                .to_vec()
            );
        }
    }

    #[test]
    fn unknown_kernel_and_missing_params_are_rejected() {
        assert!(build_event_kernel_rhs("ephemeris", &params(&[])).is_err());
        assert!(build_event_kernel_rhs("cr3bp", &params(&[])).is_err());
        assert!(build_event_kernel_rhs("cr3bp-with-stm", &params(&[])).is_err());
        assert!(build_event_kernel_rhs("bcr4bp", &params(&[("mu", MU)])).is_err());
        assert!(build_event_kernel_rhs("bcr4bp-with-stm", &params(&[("mu", MU)])).is_err());
    }
}
