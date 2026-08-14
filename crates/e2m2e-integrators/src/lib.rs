//! e2m2e 积分器 crate：`e2m2e._integrators` 扩展的 PyO3 绑定与编译入口。
//!
//! 纯数学积分器（Butcher 表、RK/ABM/Cowell、solve_ivp）在
//! `e2m2e-propagation` crate，力模型在 `e2m2e-forces`，SPICE FFI 在
//! `e2m2e-spice`；本 crate 只做 PyO3 绑定与 shooting 算法。

use pyo3::prelude::*;
#[cfg(feature = "spice")]
use pyo3::types::PyTuple;
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "spice")]
pub mod frame_convert;
#[cfg(feature = "spice")]
pub mod homotopy;
#[cfg(feature = "spice")]
pub mod multiple_shooting;
pub mod normal_form;
#[cfg(feature = "spice")]
pub mod segmented_shooting;

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
/// `history` = `[x_{n−1}, x_n, a_{n−7}, ..., a_n]` （10 个向量：2 个位置采样
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

/// 解析 abi-version.txt 的纯数字内容为 u32（const fn，编译期求值）。
const fn parse_abi_version(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut result: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b >= b'0' && b <= b'9' {
            result = result * 10 + (b - b'0') as u32;
        }
        i += 1;
    }
    result
}

/// 从 abi-version.txt 读取（单一来源），build.rs 同步生成 Python 侧 _rust_abi.py。
///
/// # 版本沿革（issue #317 第 3.3 项）
///
/// abi-version 只在**新增/改 pyfunction 边界** 时 bump（Rust 内部函数签名
/// 变更不 bump——它们不是 Python 可见的 ABI）。每次 bump 须在本节补行：
///
/// - **v1** （#300，5b616cc）：初始 ABI 版本戳 + 统一网关 ``_check_rust_abi``。
/// - **v2** （3b28353）：新增 ``propagate_with_state_py`` （EphemerisDynamics
///   纯状态 Rust 路径）。
/// - **v3** （ff63403）：新增 ``transfer_grid_search_serial_py`` +
///   ``TransferPointResult`` pyclass（转移网格搜索串行评估）。
/// - **v4** （#334）：新增 ``spice_spkezr`` + ``spice_pxform`` （Rust CSPICE
///   实例诊断查询 API）。
/// - **v5**：多重与分段打靶结果将公开 ``converged`` 替换为
///   ``status`` / ``cause`` / ``message`` 三元组。
///
/// 「1→3 跳号」实为 1→2→3 两次单步 bump，分别在上述两 commit；不存在跳过的
/// 中间版本。ADR 0018 记录的 ∂a/∂v 雅可比接口扩是 Rust 内部签名变更，未 bump。
const RUST_PY_ABI: u32 = parse_abi_version(include_str!("../abi-version.txt"));

/// 返回 Rust↔Python ABI 版本号（编译期常量，反映此 .pyd/.so 真实状态）。
#[pyfunction]
fn _py_abi_version() -> u32 {
    RUST_PY_ABI
}

/// 占位函数，用于验证 FFI 路径端到端通畅。
#[pyfunction]
fn hello_integrators() -> PyResult<String> {
    Ok("hello from e2m2e-integrators".to_string())
}

/// Python 接口：完整自适应步长 ODE 积分器（scipy `solve_ivp` 等价物）。
///
/// 使用 DOP853 (Prince-Dormand 8(7)13M) 方法。纯 Rust 积分循环在
/// `e2m2e_propagation::solve_ivp`，本函数只做 Python 回调适配与结果封装。
///
/// # 参数
/// - `t_span`: `(t_start, t_end)` 积分区间
/// - `y0`: 初始状态向量
/// - `t_eval`: 输出时间点数组
/// - `rtol`: 相对容差
/// - `atol`: 绝对容差
/// - `f`: Python callable `f(t, y) -> dy/dt`
/// - `max_step`: 最大步长（默认 `f64::INFINITY`）
/// - `max_steps`: 最大步数（默认 `MAX_ADAPTIVE_STEPS`）
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
/// - ``events``: ``[(callable, terminal, direction), ...]``，callable 为
///   ``g(t, y) -> float``；``terminal=True`` 触发即停；``direction`` > 0 只记
///   上行穿越、< 0 只记下行、0 双向（scipy ``solve_ivp`` 语义）
/// - ``method``: RK 方法（默认 ``Pd78``，即 DOP853）
/// - 其余参数同 ``solve_ivp_py``
///
/// **返回**
///
/// Python dict：``{"states", "time", "n_steps", "t_events", "y_events", "terminal_event"}``；
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
/// - ``perturbers_flat``：扁平化扰动体列表，每 4 个一组 ``[px, py, pz, gm]`` （位置 km、
///   gm km³/s²）。长度必须是 4 的倍数。
/// - ``k_love_flat``：Love 数表 5×5 行优先扁平化，长度 25。
/// - ``k_plus_flat``：弹性 Love 数 5 元素，或 ``None`` （无贡献）。
/// - ``mu_central``、``r_central``：中心天体 GM 与参考半径。
///
/// **返回**
///
/// 长度 50 的 ``Vec<f64>``：``C(25) ++ S(25)``，各为 5×5 行优先扁平化。
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
fn solid_tide_step2(et: f64) -> PyResult<Vec<f64>> {
    Ok(e2m2e_forces::solid_tide::solid_tide_step2(et))
}

/// 极潮（固体极潮 + 海洋极潮，IERS TN32）。返回长度 50 的 ``Vec<f64>`` （C25 + S25）。
#[pyfunction]
fn pole_tide(et: f64, xp: f64, yp: f64) -> PyResult<Vec<f64>> {
    Ok(e2m2e_forces::solid_tide::pole_tide(et, xp, yp))
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

/// 首次调用时经 Once 触发 Rust CSPICE 实例的行星名别名注册（对称 Python
/// 侧 SPICEManager.load_kernel 的 boddef）。幂等。
#[cfg(feature = "spice")]
pub(crate) fn ensure_bodies_registered() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(e2m2e_spice::spice_ffi::register_bodies);
}

/// 在 Rust cspice 内核池加载一个内核文件。
///
/// Rust cspice 与 Python spiceypy 是**独立的 CSPICE 实例** （静态链接，全局状态
/// 不共享）。Python 侧 furnsh 的内核，Rust 看不见；反之亦然。要让 Rust 查询
/// 可用，必须用本函数在 Rust 侧再 furnsh 一次（同一份文件，两边独立加载）。
///
/// 同时在首次加载时把行星名注册到质心/本体 ID（`register_bodies`），使本
/// 实例对 "MARS"/"JUPITER" 等的解析与 Python spiceypy 实例（那边在
/// manager.load_kernel 里 boddef）以及 DFH 一致——否则 CSPICE 默认表会把
/// "MARS" 解析成不存在的本体 499。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_furnsh(path: &str) -> PyResult<()> {
    ensure_bodies_registered();
    cspice::data::furnish(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("furnsh failed: {:?}", e))
    })?;
    LOADED_KERNELS.lock().unwrap().push(path.to_string());
    Ok(())
}

/// 已通过 [`spice_furnsh`] 加载到 Rust cspice 内核池的内核路径清单。
///
/// CSPICE `unload_c` 对未加载文件会设置错误信号，而 Python 侧
/// `SPICEManager.unload_kernel` 的语义是幂等的（重复卸载不抛，见
/// tests/data/kernels/test_spice_manager.py::test_load_and_unload）。清单保证
/// [`spice_unload`] 只对确实 furnsh 过的文件调 `unload_c`，未加载文件静默跳过。
#[cfg(feature = "spice")]
static LOADED_KERNELS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// 从 Rust cspice 内核池卸载一个内核文件（与 [`spice_furnsh`] 对称）。
///
/// Rust cspice 与 Python spiceypy 独立（见 [`spice_furnsh`] 文档）。
/// `SPICEManager.load_kernel` 双 furnsh，卸载必须对称：否则 Rust 内核池残留
/// 已卸载文件，测试结果依赖同进程执行顺序（issue #387）。只卸载清单中
/// 确已加载的文件，其余静默跳过（保持幂等语义）。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_unload(path: &str) -> PyResult<()> {
    let mut loaded = LOADED_KERNELS.lock().unwrap();
    if !loaded.iter().any(|p| p == path) {
        return Ok(());
    }
    cspice::data::unload(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("unload failed: {:?}", e))
    })?;
    loaded.retain(|p| p != path);
    Ok(())
}

/// 诊断用：在 Rust CSPICE 实例上查 spkezr（与 spiceypy.spkezr 同名函数对齐）。
///
/// 用于对比 Python（spiceypy）与 Rust（cspice-sys）两个独立 CSPICE 实例的
/// 查询结果，排查内核加载 / boddef 同步问题。常规查询仍走
/// ``SPICEManager`` / spiceypy。返回 ``(state[6], lt)``。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_spkezr(
    target: &str,
    et: f64,
    frame: &str,
    abcorr: &str,
    observer: &str,
) -> PyResult<(Vec<f64>, f64)> {
    ensure_bodies_registered();
    let (state, lt) = e2m2e_spice::spice_ffi::spkezr(target, et, frame, abcorr, observer)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok((state.to_vec(), lt))
}

/// 诊断用：在 Rust CSPICE 实例上查 pxform（与 spiceypy.pxform 同名函数对齐）。
///
/// 用于对比 Python（spiceypy）与 Rust（cspice-sys）两个独立 CSPICE 实例的
/// 帧旋转查询，排查内核加载同步问题。常规查询仍走 ``SPICEManager`` /
/// spiceypy。返回 3×3 行优先矩阵。
#[cfg(feature = "spice")]
#[pyfunction]
fn spice_pxform(from: &str, to: &str, et: f64) -> PyResult<Vec<Vec<f64>>> {
    ensure_bodies_registered();
    let m = e2m2e_spice::spice_ffi::pxform(from, to, et)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(m.iter().map(|row| row.to_vec()).collect())
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
) -> PyResult<e2m2e_forces::forces::compiled::CompiledForce> {
    use e2m2e_forces::forces::compiled::CompiledForce;
    use e2m2e_forces::forces::gravity_field::TideMode;

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
        "ecom_srp" => {
            let dyb_vec: Vec<f64> = tuple.get_item(1)?.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("ecom_srp dyb must be a list of floats")
            })?;
            if dyb_vec.len() != 9 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ecom_srp dyb must have 9 elements, got {}",
                    dyb_vec.len()
                )));
            }
            let mut dyb = [0.0_f64; 9];
            dyb.copy_from_slice(&dyb_vec);
            let shadow_bodies: Vec<String> = tuple.get_item(2)?.extract()?;
            Ok(CompiledForce::EcomSrp { dyb, shadow_bodies })
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
        "low_thrust" => {
            // 元组格式：("low_thrust", mass, thrust, t_start, t_end, direction,
            // direction_frame)。起止时间同时为 None 时表示常开。
            let mass: f64 = tuple.get_item(1)?.extract()?;
            let thrust: f64 = tuple.get_item(2)?.extract()?;
            let start_obj = tuple.get_item(3)?;
            let t_start: Option<f64> = if start_obj.is_none() {
                None
            } else {
                Some(start_obj.extract()?)
            };
            let end_obj = tuple.get_item(4)?;
            let t_end: Option<f64> = if end_obj.is_none() {
                None
            } else {
                Some(end_obj.extract()?)
            };
            let direction: Vec<f64> = tuple.get_item(5)?.extract()?;
            let frame_obj = tuple.get_item(6)?;
            let direction_frame: Option<String> = if frame_obj.is_none() {
                None
            } else {
                Some(frame_obj.extract()?)
            };
            if mass <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust mass must be positive",
                ));
            }
            if thrust < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust thrust must be non-negative",
                ));
            }
            if t_start.is_some() != t_end.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust pulse requires both t_start and t_end",
                ));
            }
            if let (Some(start), Some(end)) = (t_start, t_end) {
                if end < start {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "low_thrust t_end must be greater than or equal to t_start",
                    ));
                }
            }
            if direction.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust direction must have 3 elements",
                ));
            }
            if !matches!(
                direction_frame.as_deref(),
                None | Some("VNB") | Some("LVLH")
            ) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low_thrust direction_frame must be None, 'VNB', or 'LVLH'",
                ));
            }
            Ok(CompiledForce::LowThrust {
                mass,
                thrust,
                t_start,
                t_end,
                direction: [direction[0], direction[1], direction[2]],
                direction_frame,
            })
        }
        "drag" => {
            // 元组格式：("drag", area, mass, cd, propagation_frame, f107, ap)
            let area: f64 = tuple.get_item(1)?.extract()?;
            let mass: f64 = tuple.get_item(2)?.extract()?;
            let cd: f64 = tuple.get_item(3)?.extract()?;
            let propagation_frame: String = tuple.get_item(4)?.extract()?;
            let f107: f64 = tuple.get_item(5)?.extract()?;
            let ap: f64 = tuple.get_item(6)?.extract()?;
            Ok(CompiledForce::Drag {
                area,
                mass,
                cd,
                f107,
                ap,
                propagation_frame,
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown force tag {:?}",
            tag
        ))),
    }
}

/// `propagate_compiled` 主循环（释 GIL 段）的输出：time / states / 步数统计。
/// 独立成 type alias 修 clippy `type_complexity` （与 `AccelJacobiResult` 同法）。
///
/// 与唯一使用者 `propagate_compiled` 同步 cfg：无 spice feature 时该函数被
/// 编译期剔除，alias 须一并剔除，否则 `cargo clippy --workspace` （默认无 spice）
/// 报 dead_code（#318 CI）。
#[cfg(feature = "spice")]
type CompiledPropResult = (Vec<f64>, Vec<Vec<f64>>, usize, usize, usize);

/// `propagate_compiled` 与 `propagate_segments` 共用的 6 维编译积分核心。
///
/// 无 Python 对象交互（力模型序列已解析为 [`CompiledForce`]），可安全置于
/// `allow_threads` 区并 rayon 并发。输出语义：``t_eval`` 逐点对应输出
/// （``t_eval[0]≈t0`` 时含初值；不追加 t_span 终点——追加是 Python 侧
/// ``ForceModel._prepare_t_eval`` 的行为）。
#[cfg(feature = "spice")]
#[allow(clippy::too_many_arguments)]
fn propagate_compiled_core(
    method: RkMethod,
    t0: f64,
    y0: &[f64],
    h_init: f64,
    tol: f64,
    t_eval: &[f64],
    observer: &str,
    forces: &[e2m2e_forces::forces::compiled::CompiledForce],
    max_steps: usize,
) -> Result<CompiledPropResult, String> {
    use e2m2e_forces::forces::compiled::{compute_total_acceleration, next_force_discontinuity};
    let table = method.table();
    let mut y = y0.to_vec();
    let mut t = t0;
    let mut h = h_init;
    // 输出起点跟随 t_eval：当 t_eval[0]==t0 时记录初始状态、eval_idx 从 1 起步；
    // 当 t_eval[0]>t0（逐段积分常态：patch point 时刻非整数小时，et_grid 整数
    // 小时点严格大于 t0）时不预设 t0 到输出，eval_idx 从 0 起步由循环匹配。
    // 此前硬编码 vec![t0] + eval_idx=1 假设 t_eval[0]==t0，导致 t_eval[0]>t0
    // 时首个输出点状态错置为初值、与后续点错位。
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    let mut eval_idx = 0usize;
    if !t_eval.is_empty() && (t0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t0);
        states.push(y.clone());
        eval_idx = 1;
    }
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    let mut n_steps_capped = 0usize;

    // 用 RefCell 包装 cspice 错误状态（不能直接通过 explicit_rk_step 的 E 传）
    use std::cell::RefCell;
    let last_error: RefCell<Option<String>> = RefCell::new(None);

    while t < t_eval[t_eval.len() - 1] && n_steps < max_steps {
        n_steps += 1;
        // 限制步长不超过下一个评估点（提高 t_eval 命中率），且不超过
        // h_init（作为最大步长：稀疏 t_eval 下自适应步长失控，实测
        // 2 点 vs 31 点网格的 30 天结果差 22 万 km）
        let t_final = t_eval[t_eval.len() - 1];
        let mut t_next = if eval_idx < t_eval.len() {
            t_eval[eval_idx]
        } else {
            t_final
        };
        if let Some(boundary) = next_force_discontinuity(forces, t, t_final) {
            t_next = t_next.min(boundary);
        }
        if t + h > t_next {
            h = t_next - t;
        }
        if h > h_init {
            n_steps_capped += 1;
            h = h_init;
        }

        // RK 单步：用 Rust 闭包调 compute_total_acceleration
        let forces_ref = &forces;
        let observer_ref = observer;
        let err_cell = &last_error;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let state6 = [yi[0], yi[1], yi[2], yi[3], yi[4], yi[5]];
            compute_total_acceleration(forces_ref, ti, &state6, observer_ref)
                .map(|a| vec![yi[3], yi[4], yi[5], a[0], a[1], a[2]])
                .inspect_err(|e| {
                    *err_cell.borrow_mut() = Some(e.clone());
                })
        };

        let (y_new, error) = match explicit_rk_step(table, t, &y, h, callback, None) {
            Ok(r) => r,
            Err(msg) => return Err(format!("RK step force error: {}", msg)),
        };

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
                return Err("step size collapsed below minimum".to_string());
            }
        }
    }

    Ok((times, states, n_steps, n_rejected, n_steps_capped))
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
    use e2m2e_forces::forces::compiled::CompiledForce;

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
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }

    // 解析 forces
    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    // 主积分循环包进 py.allow_threads 释放 GIL：compiled 力模型为纯 Rust
    // （compute_total_acceleration 不回调 Python，cspice 走 FFI），与
    // multiple_shooting_correct / transfer_grid_search 路径同理（#318）。
    // 释 GIL 段（闭包内）：RK 主循环 + 每步 compute_total_acceleration；
    // 持 GIL 段（闭包外）：上面的 force 元组解析 + 下面的 PyDict 返回构造。
    // 闭包内不构造 PyErr（不借 Python 对象），仅回传 String，闭包外 map_err 转 PyErr。
    let (times, states, n_steps, n_rejected, n_steps_capped) = py
        .allow_threads(move || -> Result<CompiledPropResult, String> {
            propagate_compiled_core(
                method, t0, &y0, h_init, tol, &t_eval, observer, &forces, max_steps,
            )
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    // 返回 dict
    let dict = PyDict::new(py);
    dict.set_item("time", times)?;
    dict.set_item("states", states)?;
    dict.set_item("n_steps", n_steps)?;
    dict.set_item("n_rejected", n_rejected)?;
    dict.set_item("n_steps_capped", n_steps_capped)?;
    Ok(dict.into())
}

/// 多段并发积分（segmented 逐段积分填 et_grid 用，issue #400 性能下沉）。
///
/// 每段从 ``seg_states[i]`` 积分到 ``seg_t1[i]``，输出 ``t_eval_list[i]``
/// 逐点对应的状态序列（不追加段终点，语义同 `propagate_compiled_core`）。
/// 段间独立（只依赖本段输入），rayon 并发；并行前提与多重打靶段积分相同
/// （strict + 预采样星历缓存，零 cspice FFI）。rayon 保序 collect + 各段
/// 积分确定 → 并行与串行位级一致（``E2M2E_MS_PARALLEL=0`` 强制串行）。
///
/// 初值步长上限复刻 Python ``ForceModel._estimate_initial_step``（2πr/v/100），
/// 与 ``fm.propagate`` 路径的步长控制语义一致。
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces, seg_t0, seg_t1, seg_states, t_eval_list, rtol, max_steps=500_000, method=RkMethod::Pd45))]
#[allow(clippy::too_many_arguments)]
fn propagate_segments_py(
    observer: &str,
    forces: Vec<PyObject>,
    seg_t0: Vec<f64>,
    seg_t1: Vec<f64>,
    seg_states: Vec<Vec<f64>>,
    t_eval_list: Vec<Vec<f64>>,
    rtol: f64,
    max_steps: usize,
    method: RkMethod,
    py: Python<'_>,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    use e2m2e_forces::forces::compiled::CompiledForce;

    let n_seg = seg_t0.len();
    if seg_t1.len() != n_seg || seg_states.len() != n_seg || t_eval_list.len() != n_seg {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seg_t0/seg_t1/seg_states/t_eval_list 长度必须一致",
        ));
    }
    if forces.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "forces must not be empty",
        ));
    }

    // 解析 forces: Vec<PyObject> -> Vec<CompiledForce>
    let mut compiled_forces: Vec<CompiledForce> = Vec::with_capacity(forces.len());
    for item in &forces {
        compiled_forces.push(parse_force_tuple(&item.bind(py).as_borrowed())?);
    }
    // 状态转 6 元组
    let states6: Vec<[f64; 6]> = seg_states
        .iter()
        .map(|s| {
            if s.len() != 6 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "state must have 6 elements",
                ));
            }
            Ok([s[0], s[1], s[2], s[3], s[4], s[5]])
        })
        .collect::<PyResult<Vec<_>>>()?;

    let parallel = std::env::var("E2M2E_MS_PARALLEL").map_or(true, |v| v != "0");
    let run = |i: usize| -> Result<Vec<Vec<f64>>, String> {
        let y0 = &states6[i];
        let r = (y0[0] * y0[0] + y0[1] * y0[1] + y0[2] * y0[2]).sqrt();
        let v = (y0[3] * y0[3] + y0[4] * y0[4] + y0[5] * y0[5]).sqrt();
        let h_init = if r == 0.0 || v == 0.0 {
            1e-6 * (seg_t1[i] - seg_t0[i]).abs()
        } else {
            2.0 * std::f64::consts::PI * r / v / 100.0
        };
        let (_, states, ..) = propagate_compiled_core(
            method,
            seg_t0[i],
            y0,
            h_init,
            rtol,
            &t_eval_list[i],
            observer,
            &compiled_forces,
            max_steps,
        )?;
        Ok(states)
    };
    let results: Vec<Result<Vec<Vec<f64>>, String>> =
        py.allow_threads(move || -> Vec<Result<Vec<Vec<f64>>, String>> {
            if parallel {
                use rayon::prelude::*;
                (0..n_seg).into_par_iter().map(run).collect()
            } else {
                (0..n_seg).map(run).collect()
            }
        });
    let mut all = Vec::with_capacity(n_seg);
    for r in results {
        all.push(r.map_err(pyo3::exceptions::PyRuntimeError::new_err)?);
    }
    Ok(all)
}

/// 7D 可变质量低推力传播：状态 `[x, y, z, vx, vy, vz, m]`。
///
/// 受控动力学复用 `e2m2e-forces` 的 `augmented_eom_7d`：重力走
/// `compute_total_acceleration`，推力与质量流走 `ThrustParams`。控制律为
/// 常量 throttle 与常量方向（与 `ThrustParams` 的常量语义对齐）；时变控制
/// 留待求解器期次。
///
/// # 参数
/// - `method`: RK 方法
/// - `t0`: 起始时刻（SPICE et 秒）
/// - `y0`: 初始状态，长度 7
/// - `h_init`: 初始步长
/// - `tol`: 步长误差容差
/// - `t_eval`: 评估时刻数组
/// - `observer`: 传播系 origin（如 "EARTH"）
/// - `forces_py`: 非推力 force 元组列表（格式同 `propagate_compiled`）
/// - `thrust_spec`: `(t_max, isp, throttle, dir_x, dir_y, dir_z)`
/// - `max_steps`: 最大步数
///
/// # 返回
/// Python dict：`{"time": [...], "states": [[7], ...], "n_steps": int, "n_rejected": int}`
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn propagate_compiled_lowthrust(
    method: RkMethod,
    t0: f64,
    y0: Vec<f64>,
    h_init: f64,
    tol: f64,
    t_eval: Vec<f64>,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    thrust_spec: (f64, f64, f64, f64, f64, f64),
    max_steps: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::augmented_state::{augmented_eom_7d, ThrustParams};
    use e2m2e_forces::forces::compiled::CompiledForce;

    if y0.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y0 must have length 7 (low-thrust augmented state), got {}",
            y0.len()
        )));
    }
    if y0[6] <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial mass (y0[6]) must be positive, got {}",
            y0[6]
        )));
    }
    if tol <= 0.0 || h_init <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tol and h_init must be positive",
        ));
    }

    let (t_max, isp, throttle, dir_x, dir_y, dir_z) = thrust_spec;
    if !(0.0..=1.0).contains(&throttle) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "throttle must be in [0, 1], got {throttle}"
        )));
    }
    let dir_norm = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();
    if throttle > 0.0 && dir_norm < 1e-15 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "thrust direction must be non-zero when throttle > 0",
        ));
    }
    // 归一化方向（throttle=0 时方向不影响结果，给个占位单位向量避免 NaN）
    let direction: [f64; 3] = if dir_norm > 1e-15 {
        [dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm]
    } else {
        [1.0, 0.0, 0.0]
    };
    let thrust = ThrustParams {
        t_max,
        isp,
        throttle,
        direction,
    };

    // 解析非推力 force（重力等），格式同 propagate_compiled
    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    let table = method.table();
    let mut y = y0;
    let mut t = t0;
    let mut h = h_init;
    // 输出起点跟随 t_eval：当 t_eval[0]==t0 时记录初值、eval_idx 从 1 起步；
    // 否则（逐段积分常态）不预设 t0 到输出、eval_idx 从 0 起步由循环匹配。
    // 与 propagate_compiled 同源 bug 修复对齐（此前硬编码 vec![t0] + eval_idx=1）。
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    let mut eval_idx = 0usize;
    if !t_eval.is_empty() && (t0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t0);
        states.push(y.clone());
        eval_idx = 1;
    }
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    let mut n_steps_capped = 0usize;

    // cspice 错误状态经 RefCell 透传（不能通过 explicit_rk_step 的 E 传）
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
        // 稀疏 t_eval 下自适应步长失控：限制不超过 h_init（与 propagate_compiled 一致）
        if h > h_init {
            n_steps_capped += 1;
            h = h_init;
        }

        // RK 单步：用 Rust 闭包调 augmented_eom_7d，返回 7D 导数
        let forces_ref = &forces;
        let observer_ref = observer;
        let err_cell = &last_error;
        let thrust_ref = &thrust;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let state7 = [yi[0], yi[1], yi[2], yi[3], yi[4], yi[5], yi[6]];
            augmented_eom_7d(forces_ref, observer_ref, ti, &state7, thrust_ref)
                .map(|d| vec![d[0], d[1], d[2], d[3], d[4], d[5], d[6]])
                .inspect_err(|e| {
                    *err_cell.borrow_mut() = Some(e.clone());
                })
        };

        let (y_new, error) = match explicit_rk_step(table, t, &y, h, callback, None) {
            Ok(r) => r,
            Err(msg) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "RK step force error: {msg}"
                )));
            }
        };

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

    if n_steps >= max_steps && t < t_eval[t_eval.len() - 1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "propagation reached max_steps ({max_steps}) before t_final"
        )));
    }

    // 返回 dict
    let dict = PyDict::new(py);
    dict.set_item("time", times)?;
    dict.set_item("states", states)?;
    dict.set_item("n_steps", n_steps)?;
    dict.set_item("n_rejected", n_rejected)?;
    dict.set_item("n_steps_capped", n_steps_capped)?;
    Ok(dict.into())
}

/// 7D 可变质量低推力 + 灵敏度传播（64D 增广状态）。
///
/// 在 ``propagate_compiled_lowthrust`` （7D 受控）基础上，同时积分：
/// - Φ（6×6 状态对初值 STM，链式接龙用）
/// - S（7×3 状态对控制参数 (throttle, θ₁, θ₂) 的灵敏度）
///
/// 一次传播同时产出末端状态、STM、灵敏度，供低推力求解器组装解析雅可比
/// （替代 SLSQP 数值差分）。详见 ``docs/plans/lowthrust-analytic-jacobian-prd.md``。
///
/// **参数**
///
/// - ``method``: RK 方法
/// - ``t0``: 起始时刻（SPICE et 秒）
/// - ``y0``: 初始状态，长度 7
/// - ``h_init``, ``tol``: 步长控制
/// - ``t_eval``: 评估时刻数组（取首末两点即可）
/// - ``observer``: 传播系 origin
/// - ``forces_py``: 非推力 force 元组列表
/// - ``thrust_spec``: ``(t_max, isp, throttle, θ₁, θ₂)``
/// - ``max_steps``: 最大步数
///
/// **返回**
///
/// Python dict：``{"time": [...], "states": [[7]], "stm": [[36]], "sensitivity": [[21]], "n_steps": int, "n_rejected": int}`` （均为末端时刻的值序列）
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn propagate_compiled_lowthrust_sensitivity(
    method: RkMethod,
    t0: f64,
    y0: Vec<f64>,
    h_init: f64,
    tol: f64,
    t_eval: Vec<f64>,
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    thrust_spec: (f64, f64, f64, f64, f64),
    max_steps: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::augmented_state::{augmented_eom_7d_with_sensitivity, ThrustParams};
    use e2m2e_forces::forces::compiled::CompiledForce;

    if y0.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y0 must have length 7, got {}",
            y0.len()
        )));
    }
    if y0[6] <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial mass (y0[6]) must be positive, got {}",
            y0[6]
        )));
    }
    if tol <= 0.0 || h_init <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tol and h_init must be positive",
        ));
    }

    let (t_max, isp, throttle, theta1, theta2) = thrust_spec;
    if !(0.0..=1.0).contains(&throttle) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "throttle must be in [0, 1], got {throttle}"
        )));
    }
    let thrust = ThrustParams {
        t_max,
        isp,
        throttle,
        // 方向由 (θ₁,θ₂) 在 EOM 内部参数化，这里给占位单位向量
        direction: [1.0, 0.0, 0.0],
    };

    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    // 初始 64D 增广状态：[x₇, Φ=I₆, S=0]
    let mut y = vec![0.0_f64; 64];
    y[..7].copy_from_slice(&y0);
    for i in 0..6 {
        y[7 + i * 6 + i] = 1.0; // Φ(0) = I₆
    }
    // S(0) = 0（已是默认）

    let table = method.table();
    let mut t = t0;
    let mut h = h_init;
    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    let mut n_steps_capped = 0usize;

    use std::cell::RefCell;
    let last_error: RefCell<Option<String>> = RefCell::new(None);

    // 记录落在 t_eval 的点。输出起点跟随 t_eval：当 t_eval[0]==t0 时记录初值
    // （含 Φ=I₆、S=0）、eval_idx 从 1 起步；否则（逐段积分常态）不预设 t0 到
    // 输出、eval_idx 从 0 起步由循环匹配。与 propagate_compiled 同源 bug 修复
    // 对齐（此前硬编码 vec![t0] + eval_idx=1）。
    let mut eval_idx = 0usize;
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    let mut stms: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    let mut sens: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    if !t_eval.is_empty() && (t0 - t_eval[0]).abs() <= 1e-9 {
        times.push(t0);
        states.push(y[..7].to_vec());
        stms.push(y[7..43].to_vec());
        sens.push(y[43..64].to_vec());
        eval_idx = 1;
    }

    while t < t_eval[t_eval.len() - 1] && n_steps < max_steps {
        n_steps += 1;
        if eval_idx < t_eval.len() {
            let t_next_eval = t_eval[eval_idx];
            if t + h > t_next_eval {
                h = t_next_eval - t;
            }
        }
        // 稀疏 t_eval 下自适应步长失控：限制不超过 h_init（与 propagate_compiled 一致）
        if h > h_init {
            n_steps_capped += 1;
            h = h_init;
        }

        let forces_ref = &forces;
        let observer_ref = observer;
        let err_cell = &last_error;
        let thrust_ref = &thrust;
        let theta1_c = theta1;
        let theta2_c = theta2;
        let callback = |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> {
            let mut state64 = [0.0_f64; 64];
            state64.copy_from_slice(&yi[..64]);
            augmented_eom_7d_with_sensitivity(
                forces_ref,
                observer_ref,
                ti,
                &state64,
                thrust_ref,
                theta1_c,
                theta2_c,
            )
            .map(|d| d.to_vec())
            .inspect_err(|e| {
                *err_cell.borrow_mut() = Some(e.clone());
            })
        };

        let (y_new, error) = match explicit_rk_step(table, t, &y, h, callback, None) {
            Ok(r) => r,
            Err(msg) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "RK step force error: {msg}"
                )));
            }
        };

        if error <= tol {
            t += h;
            y = y_new;
            while eval_idx < t_eval.len() && t >= t_eval[eval_idx] - 1e-9 {
                times.push(t_eval[eval_idx]);
                states.push(y[..7].to_vec());
                stms.push(y[7..43].to_vec());
                sens.push(y[43..64].to_vec());
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

    if n_steps >= max_steps && t < t_eval[t_eval.len() - 1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "propagation reached max_steps ({max_steps}) before t_final"
        )));
    }

    let dict = PyDict::new(py);
    dict.set_item("time", times)?;
    dict.set_item("states", states)?;
    dict.set_item("stm", stms)?;
    dict.set_item("sensitivity", sens)?;
    dict.set_item("n_steps", n_steps)?;
    dict.set_item("n_rejected", n_rejected)?;
    dict.set_item("n_steps_capped", n_steps_capped)?;
    Ok(dict.into())
}

/// Python 接口：42 维增广状态传播（状态 + STM）。
///
/// 纯 N 体模型（EARTH/MOON/SUN 等），用于星历修正的逐段积分。
/// 调用 ``e2m2e-forces`` 的 ``propagate_with_stm`` （DOP853 + STM 变分方程）。
///
/// **参数**
///
/// - ``bodies``: 天体名称列表（如 ``["EARTH", "MOON", "SUN"]``）
/// - ``origin``: 原点天体名称（如 ``"EARTH"``）
/// - ``gm_values``: 各天体的 GM（km³/s²），与 ``bodies`` 一一对应
/// - ``t_span``: ``(t_start, t_end)`` 积分区间（SPICE et 秒）
/// - ``t_eval``: 输出时间点数组
/// - ``initial_state``: 初始状态 ``[x, y, z, vx, vy, vz]`` （km, km/s）
/// - ``rtol``, ``atol``: 积分容差
/// - ``max_step``: 最大步长（秒），``None`` 则不限制
/// - ``max_steps``: 最大步数，``None`` 则用默认上限
///
/// **返回**
///
/// Python dict：``{"states": [[6], ...], "stm": [[36], ...], "time": [...]}``
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

/// Python 接口：6 维纯状态传播（不含 STM）。
///
/// 纯 N 体模型，与 `propagate_with_stm_py` 同用 `solve_ivp_capped`，保证
/// 两条路径的 states 前 6 维逐位相等（parity）。供 `EphemerisDynamics`
/// 的纯状态路径（`with_stm=False`）透明走 Rust，省去 42 维 STM 的开销。
///
/// # 参数
/// 同 `propagate_with_stm_py`，但不返回 STM。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "time": [...]}`
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (bodies, origin, gm_values, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_with_state_py(
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
    use e2m2e_forces::forces::nbody_stm::{propagate_with_state, NBodyConfig};

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

    let result = propagate_with_state(
        &config, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("propagation failed: {}", e)))?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("time", result.times)?;
    Ok(dict.into())
}

/// 编译型力模型 + STM 的 PD45 传播（消除 cspice 隔离）。
///
/// 与 ``propagate_with_stm_py`` （纯 NBody）不同，本函数支持所有编译型力模型：
/// PointMass、GravityField、ThirdBody、IndirectTerm、SRP、Relativistic。
/// 使用 integrators crate 的 cspice 实例，避免跨 .so 内核池隔离问题。
///
/// **参数**
///
/// - ``observer``: 传播系 origin 天体名（如 "EARTH"）
/// - ``forces_py``: force 元组列表（格式同 ``propagate_compiled``）
/// - ``t_span``: ``(t_start, t_end)`` 积分区间（SPICE et 秒）
/// - ``t_eval``: 输出时间点数组
/// - ``initial_state``: 初始状态 ``[x, y, z, vx, vy, vz]`` （km, km/s）
/// - ``rtol``, ``atol``: 积分容差
/// - ``max_step``: 最大步长（秒），``None`` 则不限制
/// - ``max_steps``: 最大步数，``None`` 则用默认上限
///
/// **返回**
///
/// Python dict：``{"states": [[6], ...], "stm": [[36], ...], "time": [...], "n_steps": int, "n_rejected": int}``
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces_py, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None, method=RkMethod::Pd78))]
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
    method: RkMethod,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::compiled::CompiledForce;
    use e2m2e_forces::forces::compiled_stm::propagate_compiled_stm;

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
        &forces, observer, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps, method,
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

/// 把 Rust 内部 [`e2m2e_forces::PropagateError`] 翻译成 Python 异常。
///
/// 所有内部 ``PropagateError`` → ``e2m2e.exceptions.PropagationFailure``
/// （``E2M2EError`` 子类）。消息前缀都加 ``prefix``（形如
/// "CR3BP propagation failed: ..."）。Python 侧据此按类型捕获，不再依赖
/// 错误消息字符串前缀匹配——改 Rust 措辞不影响 ``except PropagationFailure``。
fn propagate_error_to_pyerr(py: Python<'_>, prefix: &str, e: impl std::fmt::Display) -> PyErr {
    let msg = format!("{prefix}: {e}");
    match py
        .import("e2m2e.exceptions")
        .and_then(|m| m.getattr("PropagationFailure"))
        .and_then(|cls| cls.call1((msg.clone(),)))
    {
        Ok(instance) => PyErr::from_value(instance),
        Err(_) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    }
}

/// Python 接口：CR3BP 6 维纯状态传播（PD78）。
///
/// 纯数学（无量纲），不依赖 SPICE。供 `CR3BP_Dynamics` 透明走 Rust。
/// 循环结构与 `propagate_compiled_stm` 一致，保证与带 STM 路径的 states 逐位相同。
///
/// # 参数
/// - `mu`: 质量参数 μ = m₂/(m₁+m₂)
/// - `t_span`: `(t_start, t_end)` 积分区间（无量纲时间）
/// - `t_eval`: 输出时间点数组
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长，`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// Python dict：`{"time": [...], "states": [[6], ...], "n_steps": int, "n_rejected": int}`
#[pyfunction]
#[pyo3(signature = (mu, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp_py(
    mu: f64,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::cr3bp::propagate_cr3bp;

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

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    // 积分包进 py.allow_threads 释放 GIL：PD78 纯 Rust（不回调 Python），与
    // propagate_compiled / multiple_shooting_correct 同理（#313）。释 GIL 段 =
    // propagate_cr3bp 主循环；持 GIL 段 = 上面的入参校验 + 下面的 PyDict 构造。
    // 闭包内不构造 PyErr，仅回传 PropagateError，闭包外按类型翻译成 Python 异常。
    let result = py
        .allow_threads(|| {
            propagate_cr3bp(
                mu, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "CR3BP propagation failed", e))?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("time", result.times)?;
    dict.set_item("states", states_list)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    Ok(dict.into())
}

/// Python 接口：CR3BP 42 维增广状态传播（状态 + STM，PD78）。
///
/// 纯数学（无量纲），不依赖 SPICE。初始 STM 设为单位矩阵；步长误差控制只
/// 统计前 6 维，避免 STM 分量主导步长。
///
/// # 参数
/// 同 `propagate_cr3bp_py`。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...],
/// "n_steps": int, "n_rejected": int}`；`stm[k][i*6+j] = ∂state(t_k)[i]/∂state(t0)[j]`。
#[pyfunction]
#[pyo3(signature = (mu, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp_stm_py(
    mu: f64,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::cr3bp::propagate_cr3bp_stm;

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

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    // 积分包进 py.allow_threads 释放 GIL：PD78 纯 Rust（不回调 Python），与
    // propagate_compiled 同理（#313，design_dro STM 修正段冻结主线程的根因）。
    // 释 GIL 段 = propagate_cr3bp_stm 主循环；持 GIL 段 = 入参校验 + PyDict 构造。
    let result = py
        .allow_threads(|| {
            propagate_cr3bp_stm(
                mu, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "CR3BP STM propagation failed", e))?;

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

/// Python 接口：BCR4BP 6 维纯状态传播（PD78）。
///
/// 纯数学（无量纲），不依赖 SPICE。在 CR3BP 之上叠加太阳质点摄动，太阳
/// 位置由解析公式 `r_s(t) = a_s·(cos θ, sin θ, 0)`、`θ = θ0 + ω_s·t` 给出。
/// 循环结构与 `propagate_cr3bp_py` 一致；RK callback 把当前步时间传入 EOM
/// （BCR4BP 显式含时）。供 `BCR4BP_Dynamics` 透明走 Rust。
///
/// # 参数
/// - `mu`: 地月质量参数 μ = m₂/(m₁+m₂)
/// - `mu_sun`: 太阳无量纲质量 m_s = GM_sun / GM_EMB
/// - `sun_distance`: 太阳圆周轨道半径 a_s（无量纲）
/// - `sun_angular_rate`: 太阳会合系角速度 ω_s（无量纲，负值表示逆行）
/// - `sun_phase0`: t = 0 时刻的太阳相位角 θ0（弧度）
/// - `t_span`: `(t_start, t_end)` 积分区间（无量纲时间）
/// - `t_eval`: 输出时间点数组
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长，`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// Python dict：`{"time": [...], "states": [[6], ...], "n_steps": int, "n_rejected": int}`
#[pyfunction]
#[pyo3(signature = (mu, mu_sun, sun_distance, sun_angular_rate, sun_phase0, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcr4bp_py(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::bcr4bp::propagate_bcr4bp;

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

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    // 积分包进 py.allow_threads 释放 GIL：PD78 纯 Rust（不回调 Python），与
    // propagate_compiled 同理（#313）。释 GIL 段 = propagate_bcr4bp 主循环；
    // 持 GIL 段 = 入参校验 + PyDict 构造。
    let result = py
        .allow_threads(|| {
            propagate_bcr4bp(
                mu,
                mu_sun,
                sun_distance,
                sun_angular_rate,
                sun_phase0,
                t_span,
                &t_eval,
                &state0,
                rtol,
                atol,
                max_step,
                max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "BCR4BP propagation failed", e))?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("time", result.times)?;
    dict.set_item("states", states_list)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    Ok(dict.into())
}

/// Python 接口：BCR4BP 42 维增广状态传播（状态 + STM，PD78）。
///
/// 纯数学（无量纲），不依赖 SPICE。初始 STM 设为单位矩阵；步长误差控制只
/// 统计前 6 维，避免 STM 分量主导步长。参数同 `propagate_bcr4bp_py`。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...],
/// "n_steps": int, "n_rejected": int}`；`stm[k][i*6+j] = ∂state(t_k)[i]/∂state(t0)[j]`。
#[pyfunction]
#[pyo3(signature = (mu, mu_sun, sun_distance, sun_angular_rate, sun_phase0, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcr4bp_stm_py(
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase0: f64,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::bcr4bp::propagate_bcr4bp_stm;

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

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    // 积分包进 py.allow_threads 释放 GIL：PD78 纯 Rust（不回调 Python），与
    // propagate_compiled 同理（#313）。释 GIL 段 = propagate_bcr4bp_stm 主循环；
    // 持 GIL 段 = 入参校验 + PyDict 构造。
    let result = py
        .allow_threads(|| {
            propagate_bcr4bp_stm(
                mu,
                mu_sun,
                sun_distance,
                sun_angular_rate,
                sun_phase0,
                t_span,
                &t_eval,
                &state0,
                rtol,
                atol,
                max_step,
                max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "BCR4BP STM propagation failed", e))?;

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

/// 二体 Lambert 求解（Izzo 算法）的 Python 接口。
///
/// # 参数
/// - `r0`/`rf`：出发/到达位置 [x, y, z]（km）
/// - `tof`：飞行时间（s）
/// - `mu`：中心天体 GM（km³/s²）
/// - `long_way`：True 取长程解（转移角 > π）
/// - `revs`：完整圈数（≥ 1 时返回右分支低能解）
///
/// # 返回
/// Python dict：`{"v0": [3], "vf": [3], "n_iter": int}`；无解/不收敛抛 ValueError。
#[pyfunction]
#[pyo3(signature = (r0, rf, tof, mu, long_way, revs))]
fn lambert_izzo_py(
    r0: Vec<f64>,
    rf: Vec<f64>,
    tof: f64,
    mu: f64,
    long_way: bool,
    revs: u32,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::lambert::{lambert_izzo, TransferDirection};

    if r0.len() != 3 || rf.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "r0/rf must have length 3, got {} and {}",
            r0.len(),
            rf.len()
        )));
    }
    let direction = if long_way {
        TransferDirection::LongWay
    } else {
        TransferDirection::ShortWay
    };
    let r0_arr = [r0[0], r0[1], r0[2]];
    let rf_arr = [rf[0], rf[1], rf[2]];
    let (v0, vf, n_iter) = lambert_izzo(&r0_arr, &rf_arr, tof, mu, direction, revs)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("v0", v0.to_vec())?;
    dict.set_item("vf", vf.to_vec())?;
    dict.set_item("n_iter", n_iter)?;
    Ok(dict.into())
}

/// N×M 网格批量 Lambert 求解（porkchop 用）的 Python 接口。
///
/// # 参数
/// - `geometries`：几何列表，每项 `[r0x, r0y, r0z, rfx, rfy, rfz]` （km）
/// - `tofs`：飞行时间列表（s），对每个几何都求解一遍
/// - `mu`/`long_way`/`revs`：同 `lambert_izzo_py`
///
/// # 返回
/// 长度 `len(geometries) * len(tofs)` 的 list（几何在外，tof 在内），
/// 每项为 dict 或 None（该组合无解）。
#[pyfunction]
#[pyo3(signature = (geometries, tofs, mu, long_way, revs))]
fn lambert_batch_py(
    geometries: Vec<Vec<f64>>,
    tofs: Vec<f64>,
    mu: f64,
    long_way: bool,
    revs: u32,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::lambert::{lambert_batch, TransferDirection};

    let mut geoms: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(geometries.len());
    for g in &geometries {
        if g.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "each geometry must have length 6 [r0, rf], got {}",
                g.len()
            )));
        }
        geoms.push(([g[0], g[1], g[2]], [g[3], g[4], g[5]]));
    }
    let direction = if long_way {
        TransferDirection::LongWay
    } else {
        TransferDirection::ShortWay
    };
    let results = lambert_batch(&geoms, &tofs, mu, direction, revs);

    let list = PyList::empty(py);
    for res in results {
        match res {
            Ok((v0, vf, n_iter)) => {
                let dict = PyDict::new(py);
                dict.set_item("v0", v0.to_vec())?;
                dict.set_item("vf", vf.to_vec())?;
                dict.set_item("n_iter", n_iter)?;
                list.append(dict)?;
            }
            Err(_) => list.append(py.None())?,
        }
    }
    Ok(list.into())
}

/// 校验展平 states 数组长度是 6 的倍数，否则返回 ValueError。
///
/// 5 个 transfer_geometry pyfunction 共用：输入是 n×6 行优先展平。
fn require_states6(name: &str, states: &[f64]) -> PyResult<()> {
    if !states.len().is_multiple_of(6) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} 展平长度必须是 6 的倍数，得到 {}",
            states.len()
        )));
    }
    Ok(())
}

/// 转移搜索几何核：轨迹每步到目标轨道采样点集合的最近距离与索引。
///
/// 移植自 `search_geometry.compute_distance_series` （纯数学，非 SPICE 门控）。
/// `traj_states`/`orbit_states` 为 n×6 行优先展平，只用前 3 维位置；n_traj×n_orbit
/// 超过 1e7 时内部分块（与 numpy 同阈值）。argmin 取首个（numpy 约定）。
///
/// # 返回
/// `(d_per_step, orbit_idx_per_step)`：两个长度 n_traj 的 list。
#[pyfunction]
fn compute_distance_series_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<i64>)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_forces::transfer_geometry::compute_distance_series(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：全局最近点（min_dist, step_idx, orbit_idx）。
///
/// 移植自 `search_geometry.compute_min_distance`。`step_idx` 为 d_per_step 的
/// 首个最小值索引，`orbit_idx = orbit_idx_per_step[step_idx]`。
#[pyfunction]
fn compute_min_distance_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(f64, i64, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_forces::transfer_geometry::compute_min_distance(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：相交检测。
///
/// 移植自 `search_geometry.detect_intersection`。全局最近点距离 < `threshold`
/// 时返回该点完整 6 维状态。返回 `(found, point|None, step_idx)`，比较为严格 `<`。
#[pyfunction]
fn detect_intersection_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
    threshold: f64,
) -> PyResult<(bool, Option<Vec<f64>>, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    let (found, point, idx) = e2m2e_forces::transfer_geometry::detect_intersection(
        &traj_states,
        &orbit_states,
        threshold,
    );
    Ok((found, point.map(|p| p.to_vec()), idx))
}

/// 转移搜索几何核：局部极小检测。
///
/// 移植自 `search_geometry.detect_local_minimum`。在每步最近距离序列上找严格
/// 局部极小（两侧严格大于），取所有极小中值最小者（首个并列）。返回
/// `(found, dist, idx)`；无极小 `(false, inf, -1)`。
#[pyfunction]
fn detect_local_minimum_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(bool, f64, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_forces::transfer_geometry::detect_local_minimum(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：碰撞检测。
///
/// 移植自 `search_geometry.check_collision`。earth 中心 `[-mu,0,0]`、moon 中心
/// `[1-mu,0,0]`；earth 优先（首个命中即返回），无 earth 再扫 moon，比较为严格 `<`。
/// 返回 `(collision, body|None, idx)`，body 为 `"earth"`/`"moon"`。
#[pyfunction]
fn check_collision_py(
    traj_states: Vec<f64>,
    mu: f64,
    collision_earth_radius: f64,
    collision_moon_radius: f64,
) -> PyResult<(bool, Option<String>, i64)> {
    require_states6("traj_states", &traj_states)?;
    Ok(e2m2e_forces::transfer_geometry::check_collision(
        &traj_states,
        mu,
        collision_earth_radius,
        collision_moon_radius,
    ))
}

/// 单候选点评估结果（PyO3 绑定）。
///
/// 字段对齐 Python `search_single_departure` 组装的候选解 dict
/// （`search_parallel.py:189-215` 成功 + `:135-150` 失败分支）。`get_all`
/// 让所有字段在 Python 侧只读可访问；wrapper `grid_search_rust_serial`
/// 转为 `list[dict]` 返回，保持与 Python sequential 后端返回类型一致。
///
/// pyclass 在本 crate（e2m2e-forces 无 pyo3 依赖，纯数学结果由
/// [`TransferPointResult::from`] 转换）。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct TransferPointResult {
    pub status: String,
    pub cause: String,
    pub message: String,
    pub departure_state: Vec<f64>,
    pub departure_time: f64,
    pub alpha: f64,
    pub transfer_trajectory: Option<Vec<f64>>,
    pub transfer_times: Option<Vec<f64>>,
    pub transfer_time: Option<f64>,
    pub min_distance: Option<f64>,
    pub min_distance_idx: Option<i64>,
    pub min_distance_orbit_idx: Option<i64>,
    pub dv_departure: f64,
    pub dv_insertion: Option<f64>,
    pub intersection_found: bool,
    pub intersection_point: Option<Vec<f64>>,
    pub intersection_idx: i64,
    pub first_intersection_idx: Option<i64>,
    pub first_intersection_time: Option<f64>,
    pub first_min_distance_idx: Option<i64>,
    pub first_min_distance_time: Option<f64>,
    pub local_minimum_found: bool,
    pub local_minimum_distance: f64,
    pub local_minimum_idx: i64,
    pub collision_found: bool,
    pub collision_body: Option<String>,
    pub collision_idx: i64,
}

impl From<e2m2e_forces::transfer_grid_search::TransferPointResult> for TransferPointResult {
    fn from(r: e2m2e_forces::transfer_grid_search::TransferPointResult) -> Self {
        Self {
            status: r.status,
            cause: r.cause,
            message: r.message,
            departure_state: r.departure_state.to_vec(),
            departure_time: r.departure_time,
            alpha: r.alpha,
            transfer_trajectory: r.transfer_trajectory,
            transfer_times: r.transfer_times,
            transfer_time: r.transfer_time,
            min_distance: r.min_distance,
            min_distance_idx: r.min_distance_idx,
            min_distance_orbit_idx: r.min_distance_orbit_idx,
            dv_departure: r.dv_departure,
            dv_insertion: r.dv_insertion,
            intersection_found: r.intersection_found,
            intersection_point: r.intersection_point,
            intersection_idx: r.intersection_idx,
            first_intersection_idx: r.first_intersection_idx,
            first_intersection_time: r.first_intersection_time,
            first_min_distance_idx: r.first_min_distance_idx,
            first_min_distance_time: r.first_min_distance_time,
            local_minimum_found: r.local_minimum_found,
            local_minimum_distance: r.local_minimum_distance,
            local_minimum_idx: r.local_minimum_idx,
            collision_found: r.collision_found,
            collision_body: r.collision_body,
            collision_idx: r.collision_idx,
        }
    }
}

/// 建进度回调 drainer（#316 item ①）。
///
/// `callback=Some(cb)` 时建 unbounded channel，spawn 独立 OS 线程排空 rx：
/// 每次先 `recv` 阻塞拿一个 delta，再 `try_recv` 聚合已入队但未处理的 delta，
/// 合并后 `Python::with_gil` reacquire GIL 调 `cb(delta)`——聚合减少 GIL 获取
/// 次数。返回 `(Some(tx), Some(handle))`，tx 喂给 e2m2e-forces 网格内核。
///
/// `callback=None` 返回 `(None, None)`，内核 `progress_tx=None` 不发。
///
/// # GIL 协同
///
/// 调用方（`transfer_grid_search_*_py`）把 channel 创建 + compute + drainer
/// join 全包在 `py.allow_threads` 内：主线程释放 GIL 跑 Rust compute，drainer
/// 线程才能 reacquire GIL 实时回调。compute 结束后 `drop(tx)` → rx 迭代终止
/// → drainer 线程干净退出 → join 返回。
fn spawn_progress_drainer(
    callback: Option<PyObject>,
) -> (
    Option<crossbeam_channel::Sender<usize>>,
    Option<std::thread::JoinHandle<()>>,
) {
    match callback {
        Some(cb) => {
            let (tx, rx) = crossbeam_channel::unbounded::<usize>();
            let drainer = std::thread::spawn(move || {
                while let Ok(n) = rx.recv() {
                    let mut delta = n;
                    while let Ok(m) = rx.try_recv() {
                        delta += m;
                    }
                    // call1 返回的 Bound 引用 GIL lifetime，不能逃逸 with_gil
                    // 闭包；闭包内丢弃返回值（回调失败不终止 drainer）。
                    Python::with_gil(|py| {
                        let _ = cb.bind(py).call1((delta,));
                    });
                }
            });
            (Some(tx), Some(drainer))
        }
        None => (None, None),
    }
}

/// Python 接口：转移网格搜索（串行版，阶段 B）。
///
/// 展平 POD 输入，调纯 Rust ``transfer_grid_search_serial``，
/// 返回 ``Vec<TransferPointResult>`` （保序：外层 departure、内层 alpha）。
/// 串行不用 Rayon，但传入 ``progress_callback`` 时仍走 ``py.allow_threads``
/// 释放 GIL——否则 drainer 线程拿不到 GIL，回调退化为 compute 结束后批量触发。
///
/// **参数**
///
/// - ``dep_states``: ``n_dep*6`` 展平（行优先）
/// - ``dep_times``: ``n_dep``
/// - ``alpha_grid``: ``n_alpha``
/// - ``arrival_states``: ``n_arrival*6`` 展平（行优先）
/// - 标量包：``mu`` / ``max_transfer_time`` / ``integration_dt`` / ``intersection_threshold`` /
///   ``min_distance_threshold`` / ``collision_earth_radius`` / ``collision_moon_radius`` /
///   ``rtol`` / ``atol`` / ``max_step``
/// - ``progress_callback`` （关键字）：``cb(delta: int) -> None``，每个 departure 完成
///   调一次（出发粒度）；``None`` 不回调。
///
/// **返回**
///
/// ``list[TransferPointResult]``，长度 ``n_dep * n_alpha``。Python 侧
/// ``grid_search_rust_serial`` 转 ``list[dict]``。
#[pyfunction]
#[pyo3(signature = (dep_states, dep_times, alpha_grid, arrival_states, mu, max_transfer_time, integration_dt, intersection_threshold, min_distance_threshold, collision_earth_radius, collision_moon_radius, rtol, atol, max_step, *, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
fn transfer_grid_search_serial_py(
    dep_states: Vec<f64>,
    dep_times: Vec<f64>,
    alpha_grid: Vec<f64>,
    arrival_states: Vec<f64>,
    mu: f64,
    max_transfer_time: f64,
    integration_dt: f64,
    intersection_threshold: f64,
    min_distance_threshold: f64,
    collision_earth_radius: f64,
    collision_moon_radius: f64,
    rtol: f64,
    atol: f64,
    max_step: f64,
    progress_callback: Option<PyObject>,
    py: Python<'_>,
) -> PyResult<Vec<TransferPointResult>> {
    use e2m2e_forces::transfer_grid_search::{transfer_grid_search_serial, GridSearchParams};

    let params = GridSearchParams {
        mu,
        max_transfer_time,
        integration_dt,
        intersection_threshold,
        min_distance_threshold,
        collision_earth_radius,
        collision_moon_radius,
        rtol,
        atol,
        max_step,
    };
    let forces_results = py.allow_threads(move || {
        let (tx, drainer) = spawn_progress_drainer(progress_callback);
        let results = transfer_grid_search_serial(
            &dep_states,
            &dep_times,
            &alpha_grid,
            &arrival_states,
            &params,
            tx.as_ref(),
        );
        drop(tx);
        if let Some(h) = drainer {
            let _ = h.join();
        }
        results
    });
    Ok(forces_results
        .into_iter()
        .map(TransferPointResult::from)
        .collect())
}

/// Python 接口：转移网格搜索（阶段 C，Rayon 并行 + GIL 释放）。
///
/// 照搬 ``multiple_shooting_correct_py`` 的
/// ``py.allow_threads`` + 环境变量开关范式（``multiple_shooting.rs:660-676``）。
/// 默认走并行 ``transfer_grid_search_parallel``，``parallel=False`` 或
/// ``E2M2E_SEARCH_PARALLEL=0`` 回退串行 ``transfer_grid_search_serial``——
/// 供并行/串行位级一致性对照（两者结果逐位相同：``par_iter``+``collect`` 保序、
/// ``evaluate_point`` 纯函数）。
///
/// **参数**
///
/// 同 ``transfer_grid_search_serial_py``，新增关键字参数：
///
/// - ``parallel``: ``None`` （默认）时由 ``E2M2E_SEARCH_PARALLEL`` 决定（``"0"`` → 串行，
///   其余/未设→并行）；显式 ``True``/``False`` 覆盖环境变量。
/// - ``n_workers``: ``None`` （默认）时用 Rayon 全局线程池（线程数由
///   ``RAYON_NUM_THREADS`` 决定，未设则 cpu 核数）；显式 ``Some(n)`` 时建一次性
///   ``ThreadPoolBuilder`` 限定 ``n.max(1)`` 个线程并 ``install`` 本次 compute——
///   覆盖 ``RAYON_NUM_THREADS``。串行模式忽略此参数（无线程池）。
/// - ``progress_callback``: 同 ``transfer_grid_search_serial_py``。
///
/// **GIL 与并行**
///
/// ``py.allow_threads`` 释放 GIL 是 Rayon 真并行 + drainer 实时回调的前提——
/// 不释放则 GIL 序列化所有 Rayon worker、drainer 拿不到 GIL。channel 创建 +
/// ThreadPoolBuilder + compute + drainer join 全在闭包内，tx 在闭包内 drop，
/// drainer 干净退出。内部直接调纯 Rust ``transfer_grid_search`` 核心，不绕道持 GIL 的
/// ``propagate_cr3bp_py`` （这是最易踩的坑，见 transfer-grid-search-rust.md:109）。
#[pyfunction]
#[pyo3(signature = (dep_states, dep_times, alpha_grid, arrival_states, mu, max_transfer_time, integration_dt, intersection_threshold, min_distance_threshold, collision_earth_radius, collision_moon_radius, rtol, atol, max_step, *, parallel=None, n_workers=None, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
fn transfer_grid_search_py(
    dep_states: Vec<f64>,
    dep_times: Vec<f64>,
    alpha_grid: Vec<f64>,
    arrival_states: Vec<f64>,
    mu: f64,
    max_transfer_time: f64,
    integration_dt: f64,
    intersection_threshold: f64,
    min_distance_threshold: f64,
    collision_earth_radius: f64,
    collision_moon_radius: f64,
    rtol: f64,
    atol: f64,
    max_step: f64,
    parallel: Option<bool>,
    n_workers: Option<usize>,
    progress_callback: Option<PyObject>,
    py: Python<'_>,
) -> PyResult<Vec<TransferPointResult>> {
    use e2m2e_forces::transfer_grid_search::{
        transfer_grid_search_parallel, transfer_grid_search_serial, GridSearchParams,
        TransferPointResult as ForcesTransferPointResult,
    };

    let use_parallel = parallel
        .unwrap_or_else(|| std::env::var("E2M2E_SEARCH_PARALLEL").map_or(true, |v| v != "0"));

    let params = GridSearchParams {
        mu,
        max_transfer_time,
        integration_dt,
        intersection_threshold,
        min_distance_threshold,
        collision_earth_radius,
        collision_moon_radius,
        rtol,
        atol,
        max_step,
    };

    // 仅并行模式 + 显式 n_workers 时建一次性线程池（install 覆盖 RAYON_NUM_THREADS）。
    // 在 allow_threads 之前构建——build 失败走 PyResult 而非 FFI 边界 panic（线程创建
    // OOM/OS 限制极少见，但 panic 会拖垮整个 Python 进程）；串行模式不建池（install
    // 对单线程 work 无意义，省一次线程创建）。
    let pool = match n_workers {
        Some(n) if use_parallel => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        ),
        _ => None,
    };

    // 释放 GIL 让 Rayon 真并行 + drainer 实时回调；核心纯 Rust 不碰 Python 对象。
    let forces_results = py.allow_threads(move || {
        let (tx, drainer) = spawn_progress_drainer(progress_callback);
        let work = || -> Vec<ForcesTransferPointResult> {
            if use_parallel {
                transfer_grid_search_parallel(
                    &dep_states,
                    &dep_times,
                    &alpha_grid,
                    &arrival_states,
                    &params,
                    tx.as_ref(),
                )
            } else {
                transfer_grid_search_serial(
                    &dep_states,
                    &dep_times,
                    &alpha_grid,
                    &arrival_states,
                    &params,
                    tx.as_ref(),
                )
            }
        };
        // Some(pool) → install 到一次性线程池；None → Rayon 全局池（RAYON_NUM_THREADS）。
        let results = if let Some(p) = pool.as_ref() {
            p.install(work)
        } else {
            work()
        };
        drop(tx);
        if let Some(h) = drainer {
            let _ = h.join();
        }
        results
    });
    Ok(forces_results
        .into_iter()
        .map(TransferPointResult::from)
        .collect())
}

/// 激活 Rust 星历预采样缓存。
///
/// 在积分前把要用到的天体状态与帧旋转矩阵在均匀网格上预采样、建三次样条，
/// 装入进程级缓存。此后 Rust 力模型（ThirdBody/IndirectTerm/GravityField/Relativistic）
/// 每步查表，不再调 cspice FFI。需在 SPICE 内核已加载后调用。
///
/// **参数**
///
/// - ``targets``: 要缓存的天体对 ``[(target, observer), ...]``，如
///   ``[("MOON", "EARTH"), ("SUN", "EARTH"), ("EARTH", "SOLAR SYSTEM BARYCENTER")]``
/// - ``frame_pairs``: 要缓存的帧旋转对 ``[(from, to), ...]``，如
///   ``[("ITRF93", "J2000"), ("MOON_PA", "J2000")]``
/// - ``sxform_pairs``: 要缓存的 6×6 状态变换对 ``[(from, to), ...]``，如
///   ``[("ITRF93", "J2000")]`` （Lense-Thirring 用）。关键字参数，默认 ``None``。
/// - ``et_start``, ``et_end``: 积分时间范围（SPICE et 秒）
/// - ``dt``: 网格步长（秒），默认 3600
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (targets, frame_pairs, et_start, et_end, dt=3600.0, *, sxform_pairs=None))]
fn enable_ephem_cache(
    targets: &Bound<'_, PyList>,
    frame_pairs: &Bound<'_, PyList>,
    et_start: f64,
    et_end: f64,
    dt: f64,
    sxform_pairs: Option<&Bound<'_, PyList>>,
) -> PyResult<()> {
    let mut bodies: Vec<(String, String)> = Vec::new();
    for item in targets.iter() {
        let tup: (String, String) = item.extract()?;
        bodies.push(tup);
    }
    let mut frames: Vec<(String, String)> = Vec::new();
    for item in frame_pairs.iter() {
        let tup: (String, String) = item.extract()?;
        frames.push(tup);
    }
    let mut sxforms: Vec<(String, String)> = Vec::new();
    if let Some(pairs) = sxform_pairs {
        for item in pairs.iter() {
            let tup: (String, String) = item.extract()?;
            sxforms.push(tup);
        }
    }
    let cache = e2m2e_spice::ephem_cache::EphemCache::build(
        &bodies, &frames, &sxforms, et_start, et_end, dt,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ephem cache build: {e:?}")))?;
    e2m2e_spice::ephem_cache::enable(cache);
    Ok(())
}

/// 关闭 Rust 星历缓存（回到逐次 cspice 查询）。
#[cfg(feature = "spice")]
#[pyfunction]
fn disable_ephem_cache() {
    e2m2e_spice::ephem_cache::disable();
}

/// 返回 cspice FFI 调用计数（验证"零 cspice"用）。
#[cfg(feature = "spice")]
#[pyfunction]
fn ephem_ffi_call_count() -> u64 {
    e2m2e_spice::spice_ffi::ffi_call_count()
}

/// 清零 cspice FFI 调用计数。
#[cfg(feature = "spice")]
#[pyfunction]
fn reset_ephem_ffi_call_count() {
    e2m2e_spice::spice_ffi::reset_ffi_call_count();
}

/// 7D 受控动力学单点求值（配点法用）。
///
/// 包装 `augmented_eom_7d`：给定状态 `[r,v,m]` 与控制参数
/// `(t_max, isp, throttle, θ₁, θ₂)`，返回 7D 导数。方向由角度参数化还原。
///
/// # 参数
/// - `forces_py`: 非推力 force 元组列表（格式同 `propagate_compiled`）
/// - `observer`: 传播系 origin
/// - `et`: 历元时刻（SPICE et 秒）
/// - `state7`: 状态 `[x,y,z,vx,vy,vz,m]`
/// - `thrust_spec`: `(t_max, isp, throttle, θ₁, θ₂)`
///
/// # 返回
/// 7D 导数 `[vx,vy,vz, ax,ay,az, ṁ]`
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn augmented_eom_7d_py(
    forces_py: &Bound<'_, PyList>,
    observer: &str,
    et: f64,
    state7: Vec<f64>,
    thrust_spec: (f64, f64, f64, f64, f64),
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::augmented_state::{augmented_eom_7d, ThrustParams};
    use e2m2e_forces::forces::compiled::CompiledForce;

    if state7.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state7 must have length 7, got {}",
            state7.len()
        )));
    }
    let (t_max, isp, throttle, theta1, theta2) = thrust_spec;
    // 角度参数化方向（与 sensitivity 出口一致）
    let direction = [
        theta1.cos() * theta2.cos(),
        theta1.sin() * theta2.cos(),
        theta2.sin(),
    ];
    let thrust = ThrustParams {
        t_max,
        isp,
        throttle,
        direction,
    };

    let mut forces: Vec<CompiledForce> = Vec::with_capacity(forces_py.len());
    for item in forces_py.iter() {
        forces.push(parse_force_tuple(&item)?);
    }

    let mut s = [0.0_f64; 7];
    s.copy_from_slice(&state7);
    let d = augmented_eom_7d(&forces, observer, et, &s, &thrust)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(PyList::new(py, d.iter())?.into())
}

#[pymodule]
fn _integrators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_py_abi_version, m)?)?;
    m.add_function(wrap_pyfunction!(hello_integrators, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::project_hamiltonian_qf_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        normal_form::build_cr3bp_hamiltonian_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rk_step, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ivp_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ivp_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(multistep_step, m)?)?;
    m.add_function(wrap_pyfunction!(cowell_step, m)?)?;
    m.add_function(wrap_pyfunction!(lambert_izzo_py, m)?)?;
    m.add_function(wrap_pyfunction!(lambert_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_accel, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step1, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step2, m)?)?;
    m.add_function(wrap_pyfunction!(pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_cr3bp_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_segments_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_cr3bp_stm_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_bcr4bp_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_bcr4bp_stm_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_distance_series_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_intersection_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_local_minimum_py, m)?)?;
    m.add_function(wrap_pyfunction!(check_collision_py, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_grid_search_serial_py, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_grid_search_py, m)?)?;
    m.add_class::<TransferPointResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_poc_body_position, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_furnsh, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_unload, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_spkezr, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(spice_pxform, m)?)?;
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
    m.add_function(wrap_pyfunction!(propagate_compiled_lowthrust, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        propagate_compiled_lowthrust_sensitivity,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(enable_ephem_cache, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(disable_ephem_cache, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(ephem_ffi_call_count, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(reset_ephem_ffi_call_count, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(augmented_eom_7d_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_with_stm_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_with_state_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(propagate_compiled_stm_py, m)?)?;
    #[cfg(feature = "spice")]
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
        frame_convert::batch_synodic_to_j2000_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        frame_convert::batch_j2000_to_synodic_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(frame_convert::batch_body_states_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(frame_convert::batch_et_to_utc_py, m)?)?;
    m.add_class::<RkMethod>()?;
    m.add_class::<MultistepMethod>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<MultistepResult>()?;
    m.add_class::<CowellResult>()?;

    // Rust 物理常量同源核对入口（#377 阶段2）：把 e2m2e-propagation 从
    // constants.toml 生成的常量以 `_propagation_constants` 子模块挂出，
    // 供 Python 侧逐位对拍。
    m.add_submodule(&e2m2e_propagation::_propagation_constants_module_bound(
        m.py(),
    )?)?;

    Ok(())
}
