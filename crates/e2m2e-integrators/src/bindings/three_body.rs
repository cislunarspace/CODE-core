//! CR3BP/BCR4BP 纯状态与 STM 传播、MEGNO 及地心命运图的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::propagate_error_to_pyerr;

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
pub fn propagate_cr3bp_py(
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
    // propagate_compiled / multiple_shooting_correct 同理。释 GIL 段 =
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
/// 同 `propagate_cr3bp_py` 。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...],
/// "n_steps": int, "n_rejected": int}`；`stm[k][i*6+j] = ∂state(t_k)[i]/∂state(t0)[j]` 。
#[pyfunction]
#[pyo3(signature = (mu, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp_stm_py(
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
    // propagate_compiled 同理；积分期间持 GIL 会冻结主线程。
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
/// 位置由解析公式 `r_s(t) = a_s·(cos θ, sin θ, 0)` 、`θ = θ0 + ω_s·t` 给出。
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
pub fn propagate_bcr4bp_py(
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
    // propagate_compiled 同理。释 GIL 段 = propagate_bcr4bp 主循环；
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
/// 统计前 6 维，避免 STM 分量主导步长。参数同 `propagate_bcr4bp_py` 。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "stm": [[36], ...], "time": [...],
/// "n_steps": int, "n_rejected": int}`；`stm[k][i*6+j] = ∂state(t_k)[i]/∂state(t0)[j]` 。
#[pyfunction]
#[pyo3(signature = (mu, mu_sun, sun_distance, sun_angular_rate, sun_phase0, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_bcr4bp_stm_py(
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
    // propagate_compiled 同理。释 GIL 段 = propagate_bcr4bp_stm 主循环；
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

/// Python 接口：CR3BP MEGNO 传播（14 维：状态 + 切变分 + 两累加器）。
///
/// 纯数学（无量纲），不依赖 SPICE。返回式 142 的 Y(t) 与 Ȳ(t)；正则
/// 轨迹 Ȳ → 2，混沌轨迹线性增长（斜率 ∝ 最大 Lyapunov 指数）。
///
/// # 参数
/// 同 `propagate_cr3bp_stm_py` ，另加 `initial_delta`（切向量初值，
/// None = (1,0,0,0,0,0)）。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "y": [...], "ybar": [...],
/// "time": [...], "n_steps": int, "n_rejected": int}`。
#[pyfunction]
#[pyo3(signature = (mu, t_span, t_eval, initial_state, rtol, atol, initial_delta=None, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp_megno_py(
    mu: f64,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    rtol: f64,
    atol: f64,
    initial_delta: Option<Vec<f64>>,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_dyn::megno::propagate_cr3bp_megno;

    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 6, got {}",
            initial_state.len()
        )));
    }
    let delta = parse_initial_delta(initial_delta)?;
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }
    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    // 释 GIL 段 = propagate_cr3bp_megno 主循环；持 GIL 段 = 入参校验 + 构造。
    let result = py
        .allow_threads(|| {
            propagate_cr3bp_megno(
                mu, t_span, &t_eval, &state0, delta, rtol, atol, max_step, max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "CR3BP MEGNO propagation failed", e))?;

    megno_result_to_dict(py, result)
}

/// Python 接口：BCR4BP MEGNO 传播（太阳参数语义同 `propagate_bcr4bp_py`）。
#[pyfunction]
#[pyo3(signature = (mu, mu_sun, sun_distance, sun_angular_rate, sun_phase0, t_span, t_eval, initial_state, rtol, atol, initial_delta=None, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_bcr4bp_megno_py(
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
    initial_delta: Option<Vec<f64>>,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_dyn::megno::propagate_bcr4bp_megno;

    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 6, got {}",
            initial_state.len()
        )));
    }
    let delta = parse_initial_delta(initial_delta)?;
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t_eval must not be empty",
        ));
    }
    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = py
        .allow_threads(|| {
            propagate_bcr4bp_megno(
                mu,
                mu_sun,
                sun_distance,
                sun_angular_rate,
                sun_phase0,
                t_span,
                &t_eval,
                &state0,
                delta,
                rtol,
                atol,
                max_step,
                max_steps,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "BCR4BP MEGNO propagation failed", e))?;

    megno_result_to_dict(py, result)
}

/// 切向量初值解析（6 维；None = 单位 x 向量）。
fn parse_initial_delta(initial_delta: Option<Vec<f64>>) -> PyResult<Option<[f64; 6]>> {
    match initial_delta {
        None => Ok(None),
        Some(v) if v.len() == 6 => {
            let mut d = [0.0_f64; 6];
            d.copy_from_slice(&v);
            Ok(Some(d))
        }
        Some(v) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_delta must have length 6, got {}",
            v.len()
        ))),
    }
}

/// Python 接口：地心 EM/EMS 六域制图单格传播（命运 + MEGNO，式 142）。
///
/// 模型：地心点质量 + 摄动天体固定开普勒椭圆（月，EMS 另加日）；
/// 命运事件在积分循环内检测（终端早停可关）。
///
/// # 参数
/// - `earth_gm` / `moon_gm` / `sun_gm`：GM，km³/s²；`sun_gm = 0` 表示 EM 模型。
/// - `moon_elements` / `sun_elements`：`[a_km, ecc, inc_rad, raan_rad, argp_rad, m0_rad]`；
///   EM 模型下 `sun_elements` 忽略（可传 None）。
/// - `earth_radius_km` / `moon_radius_km`：几何半径（再入/撞月判据）。
/// - `hill_earth_km` / `hill_moon_km`：Hill 界（逃逸/进入计数）。
/// - `span_s`：积分窗，s；`stop_on_terminal`：终端事件早停；`n_out`：输出状态数。
/// - `initial_state`：6 维地心情性系初态（km / km/s）。
/// - `rtol` / `max_step` / `max_steps`：积分器配置（max_step 缺省 6 小时）。
///
/// # 返回
/// Python dict：`{states, times, y, ybar, reentry, t_reentry_s, impact,
/// t_impact_s, escaped, t_escape_s, min_r_geo_km, min_r_sel_km,
/// moon_hill_entries, terminal, n_steps, n_rejected}`；未触发的时刻字段为 None。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn propagate_geocentric_fate_map_py(
    earth_gm: f64,
    moon_gm: f64,
    moon_elements: Vec<f64>,
    sun_gm: f64,
    sun_elements: Option<Vec<f64>>,
    earth_radius_km: f64,
    moon_radius_km: f64,
    hill_earth_km: f64,
    hill_moon_km: f64,
    span_s: f64,
    stop_on_terminal: bool,
    n_out: usize,
    initial_state: Vec<f64>,
    rtol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_dyn::cartography::{propagate_geocentric_fate_map, FateMapConfig, KeplerBody};

    if moon_elements.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "moon_elements must have length 6 [a, e, i, raan, argp, m0]",
        ));
    }
    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state must have length 6, got {}",
            initial_state.len()
        )));
    }
    let moon = KeplerBody {
        gm: moon_gm,
        a_km: moon_elements[0],
        ecc: moon_elements[1],
        inc_rad: moon_elements[2],
        raan_rad: moon_elements[3],
        argp_rad: moon_elements[4],
        m0_rad: moon_elements[5],
    };
    let sun = if sun_gm > 0.0 {
        let els =
            sun_elements.unwrap_or_else(|| vec![1.495978707e8, 0.0167086342, 0.0, 0.0, 0.0, 0.0]);
        if els.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sun_elements must have length 6",
            ));
        }
        Some(KeplerBody {
            gm: sun_gm,
            a_km: els[0],
            ecc: els[1],
            inc_rad: els[2],
            raan_rad: els[3],
            argp_rad: els[4],
            m0_rad: els[5],
        })
    } else {
        None
    };
    let config = FateMapConfig {
        earth_gm,
        moon,
        sun,
        earth_radius_km,
        moon_radius_km,
        hill_earth_km,
        hill_moon_km,
        span_s,
        stop_on_terminal,
        n_out,
    };
    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = py
        .allow_threads(|| {
            propagate_geocentric_fate_map(&config, &state0, rtol, max_step, max_steps)
        })
        .map_err(|e| propagate_error_to_pyerr(py, "geocentric fate map propagation failed", e))?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();
    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("times", result.times)?;
    dict.set_item("y", result.y)?;
    dict.set_item("ybar", result.ybar)?;
    dict.set_item("reentry", result.reentry)?;
    dict.set_item("t_reentry_s", result.t_reentry_s)?;
    dict.set_item("impact", result.impact)?;
    dict.set_item("t_impact_s", result.t_impact_s)?;
    dict.set_item("escaped", result.escaped)?;
    dict.set_item("t_escape_s", result.t_escape_s)?;
    dict.set_item("min_r_geo_km", result.min_r_geo_km)?;
    dict.set_item("min_r_sel_km", result.min_r_sel_km)?;
    dict.set_item("moon_hill_entries", result.moon_hill_entries)?;
    dict.set_item("terminal", result.terminal)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    Ok(dict.into())
}

/// MEGNO 结果 → Python dict。
fn megno_result_to_dict(
    py: Python<'_>,
    result: e2m2e_dyn::megno::MegnoResult,
) -> PyResult<PyObject> {
    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();
    let deltas_list: Vec<Vec<f64>> = result.deltas.iter().map(|s| s.to_vec()).collect();
    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("deltas", deltas_list)?;
    dict.set_item("y", result.y)?;
    dict.set_item("ybar", result.ybar)?;
    dict.set_item("time", result.times)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    Ok(dict.into())
}
