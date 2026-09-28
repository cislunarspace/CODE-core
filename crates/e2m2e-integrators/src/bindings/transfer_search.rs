//! 转移搜索（porkchop/WSB/低能配对/网格搜索）的 PyO3 绑定。

use pyo3::prelude::*;

use crate::propagate_error_to_pyerr;

/// 解析 porkchop 串/并开关：显式参数优先，否则读取环境变量。
fn porkchop_parallel_enabled(parallel: Option<bool>) -> bool {
    parallel.unwrap_or_else(|| std::env::var("E2M2E_PORKCHOP_PARALLEL").map_or(true, |v| v != "0"))
}

/// porkchop 网格扫描 Rust 后端（规格路径）：终端传播 + Lambert + ΔV 组装。
///
/// 照搬 ``transfer_grid_search_py`` 的 ``py.allow_threads`` + Rayon + 环境变量
/// 开关范式（对称 ``E2M2E_SEARCH_PARALLEL`` ）：默认并行，``parallel=False`` 或
/// ``E2M2E_PORKCHOP_PARALLEL=0`` 强制串行，两者逐位一致。
///
/// **参数**
///
/// - ``t_dep`` / ``tof`` ：出发时刻与飞行时间网格。
/// - ``dep_kind`` / ``arr_kind`` ：``"orbit"`` （周期轨道终端：``*_state`` 为首点
///   状态、``*_t0`` 时间原点、``*_period`` 周期）或 ``"state"`` （固定状态终端，
///   仅 ``*_state`` 有意义）。
/// - ``mu_cr3bp`` / ``rtol`` / ``atol`` / ``max_step`` ：CR3BP 质量参数与终端
///   传播积分器配置；两端均为 ``"state"`` 时均传 ``None`` （无需传播）。
/// - ``mu_central`` / ``long_way`` / ``revs`` ：Lambert 求解配置。
///
/// **返回**
///
/// ``(dv1, dv2)`` 展平列表，长度 ``len(t_dep) * len(tof)`` ，行优先（t_dep 主序）；
/// 无解组合为 NaN。
#[pyfunction]
#[pyo3(signature = (t_dep, tof, dep_kind, dep_state, dep_t0, dep_period, arr_kind, arr_state, arr_t0, arr_period, mu_cr3bp, rtol, atol, max_step, mu_central, long_way, revs, *, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn porkchop_grid_py(
    t_dep: Vec<f64>,
    tof: Vec<f64>,
    dep_kind: &str,
    dep_state: Vec<f64>,
    dep_t0: f64,
    dep_period: f64,
    arr_kind: &str,
    arr_state: Vec<f64>,
    arr_t0: f64,
    arr_period: f64,
    mu_cr3bp: Option<f64>,
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    mu_central: f64,
    long_way: bool,
    revs: u32,
    parallel: Option<bool>,
    py: Python<'_>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use e2m2e_dyn::porkchop::{
        porkchop_grid_parallel, porkchop_grid_serial, LambertParams, PropagationParams,
        TerminalSpec,
    };

    let parse_terminal = |kind: &str, state: &[f64], t0: f64, period: f64| {
        if state.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "终端状态长度必须为 6，得到 {}",
                state.len()
            )));
        }
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(state);
        match kind {
            "orbit" => Ok(TerminalSpec::Orbit {
                state0: s,
                t0,
                period,
            }),
            "state" => Ok(TerminalSpec::State { state: s }),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "终端类型必须是 'orbit' 或 'state'，得到 {other:?}"
            ))),
        }
    };
    let dep = parse_terminal(dep_kind, &dep_state, dep_t0, dep_period)?;
    let arr = parse_terminal(arr_kind, &arr_state, arr_t0, arr_period)?;

    let needs_propagation =
        matches!(dep, TerminalSpec::Orbit { .. }) || matches!(arr, TerminalSpec::Orbit { .. });
    let propagation = match (needs_propagation, mu_cr3bp, rtol, atol, max_step) {
        (true, Some(mu), Some(rtol), Some(atol), Some(max_step)) => Some(PropagationParams {
            mu,
            rtol,
            atol,
            max_step,
        }),
        (true, _, _, _, _) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "含 orbit 终端时 mu_cr3bp、rtol、atol、max_step 均必填",
            ));
        }
        (false, _, _, _, _) => None,
    };
    let lambert = LambertParams {
        mu_central,
        long_way,
        revs,
    };

    let use_parallel = porkchop_parallel_enabled(parallel);

    // 释放 GIL：终端传播与 Lambert 均为纯 Rust（不回调 Python），Rayon 真并行。
    py.allow_threads(move || {
        if use_parallel {
            porkchop_grid_parallel(&t_dep, &tof, &dep, &arr, propagation.as_ref(), &lambert)
        } else {
            porkchop_grid_serial(&t_dep, &tof, &dep, &arr, propagation.as_ref(), &lambert)
        }
    })
    .map_err(|e| propagate_error_to_pyerr(py, "CR3BP 轨道状态传播失败", e))
}

/// porkchop 网格扫描 Rust 后端（状态网格路径）：终端状态已由 Python
/// 按 `get_arrival_state` 协议预提取，本入口只做 Lambert + ΔV 组装。
///
/// **参数**
///
/// - ``dep_states`` ：展平 ``n*6`` ，``dep_states[i*6..]`` 为 ``t_dep[i]`` 时刻出发状态。
/// - ``arr_states`` ：展平 ``n*m*6`` ，行优先（t_dep 主序），``arr_states[(i*m+j)*6..]``
///   为 ``t_dep[i] + tof[j]`` 时刻到达状态。
/// - ``tof`` / ``mu_central`` / ``long_way`` / ``revs`` / ``parallel`` ：同
///   ``porkchop_grid_py`` 。
///
/// **返回** 同 ``porkchop_grid_py`` 。
#[pyfunction]
#[pyo3(signature = (dep_states, arr_states, tof, mu_central, long_way, revs, *, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn porkchop_grid_states_py(
    dep_states: Vec<f64>,
    arr_states: Vec<f64>,
    tof: Vec<f64>,
    mu_central: f64,
    long_way: bool,
    revs: u32,
    parallel: Option<bool>,
    py: Python<'_>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use e2m2e_dyn::porkchop::{
        porkchop_grid_states_parallel, porkchop_grid_states_serial, LambertParams,
    };

    if !dep_states.len().is_multiple_of(6) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dep_states 长度必须为 6 的整数倍，得到 {}",
            dep_states.len()
        )));
    }
    let n = dep_states.len() / 6;
    let m = tof.len();
    if arr_states.len() != n * m * 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "arr_states 长度必须为 n*m*6 = {}，得到 {}",
            n * m * 6,
            arr_states.len()
        )));
    }
    let dep_grid: Vec<[f64; 6]> = dep_states
        .as_chunks::<6>()
        .0
        .iter()
        .map(|c| {
            let mut s = [0.0_f64; 6];
            s.copy_from_slice(c);
            s
        })
        .collect();
    let arr_grid: Vec<[f64; 6]> = arr_states
        .as_chunks::<6>()
        .0
        .iter()
        .map(|c| {
            let mut s = [0.0_f64; 6];
            s.copy_from_slice(c);
            s
        })
        .collect();
    let lambert = LambertParams {
        mu_central,
        long_way,
        revs,
    };

    let use_parallel = porkchop_parallel_enabled(parallel);

    Ok(py.allow_threads(move || {
        if use_parallel {
            porkchop_grid_states_parallel(&dep_grid, &arr_grid, m, &tof, &lambert)
        } else {
            porkchop_grid_states_serial(&dep_grid, &arr_grid, m, &tof, &lambert)
        }
    }))
}

/// WSB 候选结果（PyO3 绑定）。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct WsbCandidate {
    pub sun_phase0: f64,
    pub departure_phase: f64,
    pub tof_sec: f64,
    pub departure_state: Vec<f64>,
    pub perilune_state: Vec<f64>,
    pub perilune_alt_km: f64,
    pub perilune_time_dim: f64,
    pub arrival_state: Vec<f64>,
    pub h2_kepler: f64,
    pub dv_departure: f64,
    pub dv_arrival: f64,
    pub total_dv: f64,
    pub arrival_time_dim: f64,
}

impl From<e2m2e_dyn::wsb::WsbCandidate> for WsbCandidate {
    fn from(candidate: e2m2e_dyn::wsb::WsbCandidate) -> Self {
        Self {
            sun_phase0: candidate.sun_phase0,
            departure_phase: candidate.departure_phase,
            tof_sec: candidate.tof_sec,
            departure_state: candidate.departure_state.to_vec(),
            perilune_state: candidate.perilune_state.to_vec(),
            perilune_alt_km: candidate.perilune_alt_km,
            perilune_time_dim: candidate.perilune_time_dim,
            arrival_state: candidate.arrival_state.to_vec(),
            h2_kepler: candidate.h2_kepler,
            dv_departure: candidate.dv_departure,
            dv_arrival: candidate.dv_arrival,
            total_dv: candidate.total_dv,
            arrival_time_dim: candidate.arrival_time_dim,
        }
    }
}

/// 低能转移流形截面态配对结果（PyO3 绑定）。
#[pyclass(frozen, get_all)]
#[derive(Clone, Debug)]
pub struct LowEnergyPatchCandidate {
    pub i_a: usize,
    pub i_b: usize,
    pub state_a: Vec<f64>,
    pub state_b: Vec<f64>,
    pub delta_r: f64,
    pub delta_v: f64,
    pub cost: f64,
}

impl From<e2m2e_dyn::low_energy_patch::LowEnergyPatchCandidate> for LowEnergyPatchCandidate {
    fn from(candidate: e2m2e_dyn::low_energy_patch::LowEnergyPatchCandidate) -> Self {
        Self {
            i_a: candidate.i_a,
            i_b: candidate.i_b,
            state_a: candidate.state_a.to_vec(),
            state_b: candidate.state_b.to_vec(),
            delta_r: candidate.delta_r,
            delta_v: candidate.delta_v,
            cost: candidate.cost,
        }
    }
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

impl From<e2m2e_dyn::transfer_grid_search::TransferPointResult> for TransferPointResult {
    fn from(r: e2m2e_dyn::transfer_grid_search::TransferPointResult) -> Self {
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

/// 建进度回调 drainer。
///
/// `callback=Some(cb)` 时建 unbounded channel，spawn 独立 OS 线程排空 rx：
/// 每次先 `recv` 阻塞拿一个 delta，再 `try_recv` 聚合已入队但未处理的 delta，
/// 合并后 `Python::with_gil` reacquire GIL 调 `cb(delta)`，聚合减少 GIL 获取
/// 次数。返回 `(Some(tx), Some(handle))` ，tx 喂给 e2m2e-forces 网格内核。
///
/// `callback=None` 返回 `(None, None)` ，内核 `progress_tx=None` 不发。
///
/// # GIL 协同
///
/// 调用方（`transfer_grid_search_*_py` ）把 channel 创建 + compute + drainer
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

/// Python 接口：WSB 三维网格搜索。
///
/// BCR4BP 传播、近月点检测、H₂、到达态插值与候选筛选全程在 Rust 执行；
/// ``parallel``/``n_workers`` 只控制 Rust 内核，Rust worker 不回调 Python。
#[pyfunction]
#[pyo3(signature = (departure_state, target_state, mu, mu_sun, sun_distance, sun_angular_rate, sun_phase_min, sun_phase_max, n_sun_phase, departure_phase_min, departure_phase_max, n_departure_phase, tof_min_sec, tof_max_sec, n_tof, perilune_alt_min, perilune_alt_max, max_total_dv, h2_energy_threshold, tli_speed_factor, n_propagation_samples, rtol, atol, max_step, max_steps, secondary_radius_km, characteristic_length_km, characteristic_time_sec, *, parallel=None, n_workers=None, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn wsb_search_py(
    departure_state: Vec<f64>,
    target_state: Vec<f64>,
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    sun_phase_min: f64,
    sun_phase_max: f64,
    n_sun_phase: usize,
    departure_phase_min: f64,
    departure_phase_max: f64,
    n_departure_phase: usize,
    tof_min_sec: f64,
    tof_max_sec: f64,
    n_tof: usize,
    perilune_alt_min: f64,
    perilune_alt_max: f64,
    max_total_dv: f64,
    h2_energy_threshold: f64,
    tli_speed_factor: f64,
    n_propagation_samples: usize,
    rtol: f64,
    atol: f64,
    max_step: f64,
    max_steps: usize,
    secondary_radius_km: f64,
    characteristic_length_km: f64,
    characteristic_time_sec: f64,
    parallel: Option<bool>,
    n_workers: Option<usize>,
    progress_callback: Option<PyObject>,
    py: Python<'_>,
) -> PyResult<(Vec<WsbCandidate>, usize, usize)> {
    if departure_state.len() != 6 || target_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "departure_state 与 target_state 必须都是长度 6 的状态",
        ));
    }
    if n_sun_phase == 0 || n_departure_phase == 0 || n_tof == 0 || n_propagation_samples == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "WSB 网格计数必须全部大于 0",
        ));
    }

    let mut departure = [0.0_f64; 6];
    let mut target = [0.0_f64; 6];
    departure.copy_from_slice(&departure_state);
    target.copy_from_slice(&target_state);
    let params = e2m2e_dyn::wsb::WsbSearchParams {
        sun_phase_min,
        sun_phase_max,
        n_sun_phase,
        departure_phase_min,
        departure_phase_max,
        n_departure_phase,
        tof_min_sec,
        tof_max_sec,
        n_tof,
        perilune_alt_min,
        perilune_alt_max,
        max_total_dv,
        h2_energy_threshold,
        tli_speed_factor,
        n_propagation_samples,
        rtol,
        atol,
        max_step,
        max_steps,
        secondary_radius_km,
        characteristic_length_km,
        characteristic_time_sec,
    };
    let use_parallel =
        parallel.unwrap_or_else(|| std::env::var("E2M2E_WSB_PARALLEL").map_or(true, |v| v != "0"));
    let pool = match n_workers {
        Some(n) if use_parallel => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        ),
        _ => None,
    };

    let result = py.allow_threads(move || {
        let (tx, drainer) = spawn_progress_drainer(progress_callback);
        let work = || {
            if use_parallel {
                e2m2e_dyn::wsb::wsb_search_parallel(
                    &departure,
                    &target,
                    mu,
                    mu_sun,
                    sun_distance,
                    sun_angular_rate,
                    &params,
                    tx.as_ref(),
                )
            } else {
                e2m2e_dyn::wsb::wsb_search_serial(
                    &departure,
                    &target,
                    mu,
                    mu_sun,
                    sun_distance,
                    sun_angular_rate,
                    &params,
                    tx.as_ref(),
                )
            }
        };
        let result = if let Some(pool) = pool.as_ref() {
            pool.install(work)
        } else {
            work()
        };
        drop(tx);
        if let Some(drainer) = drainer {
            let _ = drainer.join();
        }
        result
    });

    let result = result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok((
        result
            .candidates
            .into_iter()
            .map(WsbCandidate::from)
            .collect(),
        result.n_propagation_failures,
        result.n_perilune_in_window,
    ))
}

/// Python 接口：低能转移流形截面态配对。
///
/// Python 侧先解析庞加莱截面并收集穿越态，本函数只接收展平 POD 状态，完成
/// 笛卡尔积、位置/速度范数、加权代价与稳定排序。计算全程在
/// ``py.allow_threads`` 内，不会让 Rayon worker 回调 Python。
#[pyfunction]
#[pyo3(signature = (states_a, states_b, weight_r, weight_v, *, parallel=None, n_workers=None, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn low_energy_patch_py(
    states_a: Vec<f64>,
    states_b: Vec<f64>,
    weight_r: f64,
    weight_v: f64,
    parallel: Option<bool>,
    n_workers: Option<usize>,
    progress_callback: Option<PyObject>,
    py: Python<'_>,
) -> PyResult<Vec<LowEnergyPatchCandidate>> {
    if !states_a.len().is_multiple_of(6) || !states_b.len().is_multiple_of(6) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "states_a 与 states_b 展平长度必须是 6 的倍数",
        ));
    }

    let use_parallel = parallel
        .unwrap_or_else(|| std::env::var("E2M2E_LOW_ENERGY_PARALLEL").map_or(true, |v| v != "0"));
    let pool = match n_workers {
        Some(n) if use_parallel => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        ),
        _ => None,
    };

    let candidates = py.allow_threads(move || {
        let (tx, drainer) = spawn_progress_drainer(progress_callback);
        let work = || {
            if use_parallel {
                e2m2e_dyn::low_energy_patch::low_energy_patch_parallel(
                    &states_a,
                    &states_b,
                    weight_r,
                    weight_v,
                    tx.as_ref(),
                )
            } else {
                e2m2e_dyn::low_energy_patch::low_energy_patch_serial(
                    &states_a,
                    &states_b,
                    weight_r,
                    weight_v,
                    tx.as_ref(),
                )
            }
        };
        let candidates = if let Some(pool) = pool.as_ref() {
            pool.install(work)
        } else {
            work()
        };
        drop(tx);
        if let Some(drainer) = drainer {
            let _ = drainer.join();
        }
        candidates
    });

    Ok(candidates
        .into_iter()
        .map(LowEnergyPatchCandidate::from)
        .collect())
}

/// Python 接口：转移网格搜索（串行版，阶段 B）。
///
/// 展平 POD 输入，调纯 Rust ``transfer_grid_search_serial`` ，
/// 返回 ``Vec<TransferPointResult>`` （保序：外层 departure、内层 alpha）。
/// 串行不用 Rayon，但传入 ``progress_callback`` 时仍走 ``py.allow_threads``
/// 释放 GIL，否则 drainer 线程拿不到 GIL，回调退化为 compute 结束后批量触发。
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
/// - ``progress_callback`` （关键字）：``cb(delta: int) -> None`` ，每个 departure 完成
///   调一次（出发粒度）；``None`` 不回调。
///
/// **返回**
///
/// ``list[TransferPointResult]`` ，长度 ``n_dep * n_alpha`` 。Python 侧
/// ``grid_search_rust_serial`` 转 ``list[dict]`` 。
#[pyfunction]
#[pyo3(signature = (dep_states, dep_times, alpha_grid, arrival_states, mu, max_transfer_time, integration_dt, intersection_threshold, min_distance_threshold, collision_earth_radius, collision_moon_radius, rtol, atol, max_step, *, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn transfer_grid_search_serial_py(
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
    use e2m2e_dyn::transfer_grid_search::{transfer_grid_search_serial, GridSearchParams};

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
/// ``py.allow_threads`` + 环境变量开关范式（``multiple_shooting.rs:660-676`` ）。
/// 默认走并行 ``transfer_grid_search_parallel`` ，``parallel=False`` 或
/// ``E2M2E_SEARCH_PARALLEL=0`` 回退串行 ``transfer_grid_search_serial``，
/// 供并行/串行位级一致性对照（两者结果逐位相同：``par_iter``+``collect`` 保序、
/// ``evaluate_point`` 纯函数）。
///
/// **参数**
///
/// 同 ``transfer_grid_search_serial_py`` ，新增关键字参数：
///
/// - ``parallel``: ``None`` （默认）时由 ``E2M2E_SEARCH_PARALLEL`` 决定（``"0"`` → 串行，
///   其余/未设→并行）；显式 ``True``/``False`` 覆盖环境变量。
/// - ``n_workers``: ``None`` （默认）时用 Rayon 全局线程池（线程数由
///   ``RAYON_NUM_THREADS`` 决定，未设则 cpu 核数）；显式 ``Some(n)`` 时建一次性
///   ``ThreadPoolBuilder`` 限定 ``n.max(1)`` 个线程并 ``install`` 本次 compute，
///   覆盖 ``RAYON_NUM_THREADS`` 。串行模式忽略此参数（无线程池）。
/// - ``progress_callback``: 同 ``transfer_grid_search_serial_py`` 。
///
/// **GIL 与并行**
///
/// ``py.allow_threads`` 释放 GIL 是 Rayon 真并行 + drainer 实时回调的前提，
/// 不释放则 GIL 序列化所有 Rayon worker、drainer 拿不到 GIL。channel 创建 +
/// ThreadPoolBuilder + compute + drainer join 全在闭包内，tx 在闭包内 drop，
/// drainer 干净退出。内部直接调纯 Rust ``transfer_grid_search`` 核心，不绕道持 GIL 的
/// ``propagate_cr3bp_py`` （这是最易踩的坑，见 transfer-grid-search-rust.md:109）。
#[pyfunction]
#[pyo3(signature = (dep_states, dep_times, alpha_grid, arrival_states, mu, max_transfer_time, integration_dt, intersection_threshold, min_distance_threshold, collision_earth_radius, collision_moon_radius, rtol, atol, max_step, *, parallel=None, n_workers=None, progress_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn transfer_grid_search_py(
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
    use e2m2e_dyn::transfer_grid_search::{
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
    // 在 allow_threads 之前构建，build 失败走 PyResult 而非 FFI 边界 panic（线程创建
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
