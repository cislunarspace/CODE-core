//! 编译型力模型传播（RK/STM/IAS15/低推力/分段）的 PyO3 绑定（整模块 spice 门控）。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::propagate_error_to_pyerr;
use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// `propagate_compiled` 主循环（释 GIL 段）的输出：time / states / 步数统计。
/// 独立成 type alias 修 clippy `type_complexity` （与 `AccelJacobiResult` 同法）。
///
/// 与唯一使用者 `propagate_compiled` 同步 cfg：无 spice feature 时该函数被
/// 编译期剔除，alias 须一并剔除，否则 `cargo clippy --workspace` （默认无 spice）
/// 报 dead_code。
#[cfg(feature = "spice")]
type CompiledPropResult = (Vec<f64>, Vec<Vec<f64>>, usize, usize, usize);

/// `propagate_compiled` 与 `propagate_segments` 共用的 6 维编译积分核心。
///
/// 无 Python 对象交互（力模型序列已解析为 [`CompiledForce`]），可安全置于
/// `allow_threads` 区并 rayon 并发。输出语义：``t_eval`` 逐点对应输出
/// （``t_eval[0]≈t0`` 时含初值；不追加 t_span 终点，追加是 Python 侧
/// ``ForceModel._prepare_t_eval`` 的行为）。传播方向由 ``t_eval`` 单调
/// 方向决定（末点早于 ``t0`` 即反向积分，``t_eval`` 单调递减）。
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
    use e2m2e_forces::forces::compiled::{
        compute_total_acceleration, next_force_discontinuity, prev_force_discontinuity,
    };
    // 传播方向由 t_eval 单调方向决定（末点 < t0 即反向积分）。
    let dir = (t_eval[t_eval.len() - 1] - t0).signum();
    let table = method.table();
    let mut y = y0.to_vec();
    let mut t = t0;
    let mut h = dir * h_init;
    // 输出起点跟随 t_eval：当 t_eval[0]==t0 时记录初始状态、eval_idx 从 1 起步；
    // 当 t_eval[0]>t0（逐段积分常态：patch point 时刻非整数小时，et_grid 整数
    // 小时点严格大于 t0）时不预设 t0 到输出，eval_idx 从 0 起步由循环匹配。
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

    while dir * (t_eval[t_eval.len() - 1] - t) > 0.0 && n_steps < max_steps {
        n_steps += 1;
        // 限制步长不超过下一个评估点（提高 t_eval 命中率），且不超过
        // h_init（作为最大步长：稀疏 t_eval 下自适应步长失控）
        let t_final = t_eval[t_eval.len() - 1];
        let mut t_next = if eval_idx < t_eval.len() {
            t_eval[eval_idx]
        } else {
            t_final
        };
        // 推力开关边界作为步长终点：正向取 (t, t_final] 内最早者，
        // 反向取 [t_final, t) 内最晚者，保证 RK 步不跨越不连续面。
        if dir > 0.0 {
            if let Some(boundary) = next_force_discontinuity(forces, t, t_final) {
                t_next = t_next.min(boundary);
            }
        } else if let Some(boundary) = prev_force_discontinuity(forces, t, t_final) {
            t_next = t_next.max(boundary);
        }
        if dir * (t + h - t_next) > 0.0 {
            h = t_next - t;
        }
        if h.abs() > h_init {
            n_steps_capped += 1;
            h = dir * h_init;
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
            while eval_idx < t_eval.len() && dir * (t - t_eval[eval_idx]) >= -1e-9 {
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
            if h.abs() < 1e-12 * (t_eval[t_eval.len() - 1] - t0).abs() {
                return Err("step size collapsed below minimum".to_string());
            }
        }
    }

    Ok((times, states, n_steps, n_rejected, n_steps_capped))
}

/// 全 Rust 力模型传播器（消除 Python↔Rust 跨界）。
///
/// Python 侧把所有 force 序列化为元组列表，Rust 在内部循环里直接调
/// `compute_total_acceleration` ，每个 RK 子阶段不再跨界回 Python。
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
pub fn propagate_compiled(
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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
    }

    // 主积分循环包进 py.allow_threads 释放 GIL：compiled 力模型为纯 Rust
    // （compute_total_acceleration 不回调 Python，cspice 走 FFI），与
    // multiple_shooting_correct / transfer_grid_search 路径同理。
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

/// 多段并发积分（segmented 逐段积分填 et_grid 用）。
///
/// 每段从 ``seg_states[i]`` 积分到 ``seg_t1[i]`` ，输出 ``t_eval_list[i]``
/// 逐点对应的状态序列（不追加段终点，语义同 `propagate_compiled_core` ）。
/// 段间独立（只依赖本段输入），rayon 并发；并行前提与多重打靶段积分相同
/// （strict + 预采样星历缓存，零 cspice FFI）。rayon 保序 collect + 各段
/// 积分确定 → 并行与串行位级一致（``E2M2E_MS_PARALLEL=0`` 强制串行）。
///
/// 初值步长上限复刻 Python ``ForceModel._estimate_initial_step`` （2πr/v/100），
/// 与 ``fm.propagate`` 路径的步长控制语义一致。
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces, seg_t0, seg_t1, seg_states, t_eval_list, rtol, max_steps=500_000, method=RkMethod::Pd45))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_segments_py(
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
        compiled_forces.push(crate::force_parse::parse_force_tuple(
            &item.bind(py).as_borrowed(),
        )?);
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

/// 7D 可变质量低推力传播：状态 `[x, y, z, vx, vy, vz, m]` 。
///
/// 受控动力学复用 `e2m2e-forces` 的 `augmented_eom_7d` ：重力走
/// `compute_total_acceleration` ，推力与质量流走 `ThrustParams` 。控制律为
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
/// - `forces_py`: 非推力 force 元组列表（格式同 `propagate_compiled` ）
/// - `thrust_spec`: `(t_max, isp, throttle, dir_x, dir_y, dir_z)`
/// - `max_steps`: 最大步数
///
/// # 返回
/// Python dict：`{"time": [...], "states": [[7], ...], "n_steps": int, "n_rejected": int}`
#[cfg(feature = "spice")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_lowthrust(
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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
    }

    let table = method.table();
    let mut y = y0;
    let mut t = t0;
    let mut h = h_init;
    // 输出起点跟随 t_eval：当 t_eval[0]==t0 时记录初值、eval_idx 从 1 起步；
    // 否则（逐段积分常态）不预设 t0 到输出、eval_idx 从 0 起步由循环匹配。
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
/// （替代 SLSQP 数值差分）。详见 ``docs/plans/lowthrust-analytic-jacobian-prd.md`` 。
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
pub fn propagate_compiled_lowthrust_sensitivity(
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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
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
    // 输出、eval_idx 从 0 起步由循环匹配。
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
/// - ``bodies``: 天体名称列表（如 ``["EARTH", "MOON", "SUN"]`` ）
/// - ``origin``: 原点天体名称（如 ``"EARTH"`` ）
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
pub fn propagate_with_stm_py(
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
    .map_err(|e| propagate_error_to_pyerr(py, "NBody STM propagation failed", e))?;

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
/// 纯 N 体模型，与 `propagate_with_stm_py` 同用 `solve_ivp_capped` ，保证
/// 两条路径的 states 前 6 维逐位相等（parity）。供 `EphemerisDynamics`
/// 的纯状态路径（`with_stm=False` ）透明走 Rust，省去 42 维 STM 的开销。
///
/// # 参数
/// 同 `propagate_with_stm_py` ，但不返回 STM。
///
/// # 返回
/// Python dict：`{"states": [[6], ...], "time": [...]}`
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (bodies, origin, gm_values, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_with_state_py(
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
    .map_err(|e| propagate_error_to_pyerr(py, "NBody propagation failed", e))?;

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
/// - ``forces_py``: force 元组列表（格式同 ``propagate_compiled`` ）
/// - ``t_span``: ``(t_start, t_end)`` 积分区间（SPICE et 秒）
/// - ``t_eval``: 输出时间点数组
/// - ``initial_state``: 初始状态 ``[x, y, z, vx, vy, vz]`` （km, km/s）
/// - ``rtol``, ``atol``: 积分容差
/// - ``max_step``: 最大步长（秒），``None`` 则不限制
/// - ``max_steps``: 最大步数，``None`` 则用默认上限
/// - ``sens_params``: 可选，``[(force_index, "cr"|"cd"), ...]`` 参数敏感列。
///   ``force_index`` 是 ``forces_py`` 中的下标；每条参数追加
///   ``∂[r,v]/∂p`` 敏感列（ASSIST 式一阶变分方程）。
///
/// **返回**
///
/// Python dict：``{"states": [[6], ...], "stm": [[36], ...], "time": [...],
/// "n_steps": int, "n_rejected": int}``；带 ``sens_params`` 时额外含
/// ``"sensitivity": [[6·n_params], ...]``（列序同 ``sens_params``）。
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces_py, t_span, t_eval, initial_state, rtol, atol, max_step=None, max_steps=None, method=RkMethod::Pd78, sens_params=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_stm_py(
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
    sens_params: Option<Vec<(usize, String)>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::compiled::CompiledForce;
    use e2m2e_forces::forces::compiled_stm::propagate_compiled_stm_sens;

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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
    }
    let sens = parse_sens_params(sens_params, forces.len())?;

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = propagate_compiled_stm_sens(
        &forces, observer, t_span, &t_eval, &state0, rtol, atol, max_step, max_steps, method, &sens,
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
    if !sens.is_empty() {
        dict.set_item("sensitivity", result.sensitivities)?;
    }
    Ok(dict.into())
}

/// 解析 ``sens_params`` 为 ``(force_index, SensParam)`` 列表。
#[cfg(feature = "spice")]
fn parse_sens_params(
    sens_params: Option<Vec<(usize, String)>>,
    n_forces: usize,
) -> PyResult<Vec<(usize, e2m2e_forces::forces::compiled::SensParam)>> {
    use e2m2e_forces::forces::compiled::SensParam;
    let mut out = Vec::new();
    for (force_idx, kind) in sens_params.unwrap_or_default() {
        if force_idx >= n_forces {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sens_params force index {force_idx} out of range ({n_forces} forces)"
            )));
        }
        let param = SensParam::parse(&kind).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown sensitivity parameter {kind:?} (valid: \"cr\", \"cd\")"
            ))
        })?;
        out.push((force_idx, param));
    }
    Ok(out)
}

/// 编译型力模型的 IAS15 传播（15 阶 Gauss-Radau，补偿求和）。
///
/// 与 RK 路径的差异：变阶预测-校正、容差 ``tol`` 为相对加速度采样量级
/// （单参数，无 rtol/atol 之分）、长弧段误差按 Brouwer 律 n^(1/2) 增长。
/// 适合高精度长弧段外推与近距交会（步长自动收缩）。
///
/// **参数**
///
/// - ``observer``, ``forces_py``, ``t_span``, ``t_eval``, ``initial_state``:
///   同 ``propagate_compiled_stm_py``
/// - ``tol``: 相对容差（建议 1e-12 ~ 1e-14）
/// - ``max_step``, ``max_steps``: 同 ``propagate_compiled_stm_py``
/// - ``with_stm``: 是否同时积分 6×6 STM（初值单位阵）
/// - ``sens_params``: 可选参数敏感列，格式同 ``propagate_compiled_stm_py``
///
/// **返回**
///
/// Python dict：``{"states": [[6], ...], "time": [...], "n_steps": int,
/// "n_rejected": int}``；``with_stm=True`` 时含 ``"stm"``，带
/// ``sens_params`` 时含 ``"sensitivity"``。
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (observer, forces_py, t_span, t_eval, initial_state, tol, max_step=None, max_steps=None, with_stm=false, sens_params=None))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_compiled_ias15_py(
    observer: &str,
    forces_py: &Bound<'_, PyList>,
    t_span: (f64, f64),
    t_eval: Vec<f64>,
    initial_state: Vec<f64>,
    tol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    with_stm: bool,
    sens_params: Option<Vec<(usize, String)>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_forces::forces::compiled::CompiledForce;
    use e2m2e_forces::forces::compiled_ias15::propagate_compiled_ias15;

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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
    }
    let sens = parse_sens_params(sens_params, forces.len())?;

    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = py
        .allow_threads(|| {
            propagate_compiled_ias15(
                &forces, observer, t_span, &t_eval, &state0, tol, max_step, max_steps, with_stm,
                &sens,
            )
        })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("IAS15 propagation failed: {}", e))
        })?;

    let states_list: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();

    let dict = PyDict::new(py);
    dict.set_item("states", states_list)?;
    dict.set_item("time", result.times)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("n_rejected", result.n_rejected)?;
    if with_stm {
        let stm_list: Vec<Vec<f64>> = result.stms.iter().map(|s| s.to_vec()).collect();
        dict.set_item("stm", stm_list)?;
    }
    if !sens.is_empty() {
        dict.set_item("sensitivity", result.sensitivities)?;
    }
    Ok(dict.into())
}

/// 7D 受控动力学单点求值（配点法用）。
///
/// 包装 `augmented_eom_7d` ：给定状态 `[r,v,m]` 与控制参数
/// `(t_max, isp, throttle, θ₁, θ₂)` ，返回 7D 导数。方向由角度参数化还原。
///
/// # 参数
/// - `forces_py`: 非推力 force 元组列表（格式同 `propagate_compiled` ）
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
pub fn augmented_eom_7d_py(
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
        forces.push(crate::force_parse::parse_force_tuple(&item)?);
    }

    let mut s = [0.0_f64; 7];
    s.copy_from_slice(&state7);
    let d = augmented_eom_7d(&forces, observer, et, &s, &thrust)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(PyList::new(py, d.iter())?.into())
}
