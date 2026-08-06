"""Public Python shim for the Rust integrator extension."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from e2m2e._integrators import (
        CowellResult,
        MultistepMethod,
        MultistepResult,
        RkMethod,
        TransferPointResult,
        build_cr3bp_hamiltonian_py,
        check_collision_py,
        compute_distance_series_py,
        compute_min_distance_py,
        detect_intersection_py,
        detect_local_minimum_py,
        hello_integrators,
        lambert_batch_py,
        lambert_izzo_py,
        pole_tide,
        project_hamiltonian_qf_py,
        propagate_bcr4bp_py,
        propagate_bcr4bp_stm_py,
        propagate_cr3bp_py,
        propagate_cr3bp_stm_py,
        solid_tide_step1,
        solid_tide_step2,
        solve_ivp_events_py,
        spherical_harmonic_accel,
        transfer_grid_search_serial_py,
    )
    from e2m2e._integrators import cowell_step as _cowell_step
    from e2m2e._integrators import multistep_step as _multistep_step
    from e2m2e._integrators import rk_step as _rk_step

    # Spice-gated symbols: absent when extension built without --features spice.
    try:
        from e2m2e._integrators import (
            augmented_eom_7d_py,
            disable_ephem_cache,
            enable_ephem_cache,
            ephem_ffi_call_count,
            indirect_term_acceleration,
            multiple_shooting_correct_py,
            propagate_compiled,
            propagate_compiled_lowthrust,
            propagate_compiled_lowthrust_sensitivity,
            propagate_compiled_stm_py,
            propagate_with_state_py,
            propagate_with_stm_py,
            reset_ephem_ffi_call_count,
            segmented_shooting_correct_py,
            spice_poc_furnsh,
            third_body_acceleration,
        )
    except ImportError:
        augmented_eom_7d_py = None  # type: ignore[misc,assignment]
        disable_ephem_cache = None  # type: ignore[misc,assignment]
        enable_ephem_cache = None  # type: ignore[misc,assignment]
        ephem_ffi_call_count = None  # type: ignore[misc,assignment]
        indirect_term_acceleration = None  # type: ignore[misc,assignment]
        multiple_shooting_correct_py = None  # type: ignore[misc,assignment]
        propagate_compiled = None  # type: ignore[misc,assignment]
        propagate_compiled_lowthrust = None  # type: ignore[misc,assignment]
        propagate_compiled_lowthrust_sensitivity = None  # type: ignore[misc,assignment]
        propagate_compiled_stm_py = None  # type: ignore[misc,assignment]
        propagate_with_state_py = None  # type: ignore[misc,assignment]
        propagate_with_stm_py = None  # type: ignore[misc,assignment]
        reset_ephem_ffi_call_count = None  # type: ignore[misc,assignment]
        segmented_shooting_correct_py = None  # type: ignore[misc,assignment]
        spice_poc_furnsh = None  # type: ignore[misc,assignment]
        third_body_acceleration = None  # type: ignore[misc,assignment]
except ModuleNotFoundError:
    # _integrators is a compiled Rust extension; allow import for doc builds
    # where the extension is not built.
    CowellResult = None  # type: ignore[misc,assignment]
    MultistepMethod = None  # type: ignore[misc,assignment]
    MultistepResult = None  # type: ignore[misc,assignment]
    RkMethod = None  # type: ignore[misc,assignment]
    TransferPointResult = None  # type: ignore[misc,assignment]
    build_cr3bp_hamiltonian_py = None  # type: ignore[misc,assignment]
    check_collision_py = None  # type: ignore[misc,assignment]
    compute_distance_series_py = None  # type: ignore[misc,assignment]
    compute_min_distance_py = None  # type: ignore[misc,assignment]
    detect_intersection_py = None  # type: ignore[misc,assignment]
    detect_local_minimum_py = None  # type: ignore[misc,assignment]
    hello_integrators = None  # type: ignore[misc,assignment]
    lambert_batch_py = None  # type: ignore[misc,assignment]
    lambert_izzo_py = None  # type: ignore[misc,assignment]
    pole_tide = None  # type: ignore[misc,assignment]
    project_hamiltonian_qf_py = None  # type: ignore[misc,assignment]
    propagate_bcr4bp_py = None  # type: ignore[misc,assignment]
    propagate_bcr4bp_stm_py = None  # type: ignore[misc,assignment]
    propagate_cr3bp_py = None  # type: ignore[misc,assignment]
    propagate_cr3bp_stm_py = None  # type: ignore[misc,assignment]
    solid_tide_step1 = None  # type: ignore[misc,assignment]
    solid_tide_step2 = None  # type: ignore[misc,assignment]
    solve_ivp_events_py = None  # type: ignore[misc,assignment]
    spherical_harmonic_accel = None  # type: ignore[misc,assignment]
    transfer_grid_search_serial_py = None  # type: ignore[misc,assignment]
    _cowell_step = None  # type: ignore[misc,assignment]
    _multistep_step = None  # type: ignore[misc,assignment]
    _rk_step = None  # type: ignore[misc,assignment]
    augmented_eom_7d_py = None  # type: ignore[misc,assignment]
    disable_ephem_cache = None  # type: ignore[misc,assignment]
    enable_ephem_cache = None  # type: ignore[misc,assignment]
    ephem_ffi_call_count = None  # type: ignore[misc,assignment]
    indirect_term_acceleration = None  # type: ignore[misc,assignment]
    multiple_shooting_correct_py = None  # type: ignore[misc,assignment]
    propagate_compiled = None  # type: ignore[misc,assignment]
    propagate_compiled_lowthrust = None  # type: ignore[misc,assignment]
    propagate_compiled_lowthrust_sensitivity = None  # type: ignore[misc,assignment]
    propagate_compiled_stm_py = None  # type: ignore[misc,assignment]
    propagate_with_state_py = None  # type: ignore[misc,assignment]
    propagate_with_stm_py = None  # type: ignore[misc,assignment]
    reset_ephem_ffi_call_count = None  # type: ignore[misc,assignment]
    segmented_shooting_correct_py = None  # type: ignore[misc,assignment]
    spice_poc_furnsh = None  # type: ignore[misc,assignment]
    third_body_acceleration = None  # type: ignore[misc,assignment]

# ---- Python↔Rust ABI 版本校验 ----
# 单一来源：crates/e2m2e-integrators/abi-version.txt
# build.rs 在 maturin develop 时生成 e2m2e/_rust_abi.py；未构建时回落到硬编码默认值。
try:
    from e2m2e._rust_abi import _ABI_VERSION as _MIN_REQUIRED_RUST_ABI
except ImportError:
    _MIN_REQUIRED_RUST_ABI: int = 1  # type: ignore[no-redef]  # 构建前/无扩展时的安全默认值

_abi_ok: bool = False  # 进程级一次性缓存


def _check_rust_abi() -> None:
    """校验 Rust 扩展 ABI 版本；过期即报，缺失则静默跳过。

    在首次使用 Rust 扩展符号时调用（惰性），结果缓存。扩展不存在时（
    ``ImportError``）保留原有静默降级路径（doc build / 无 spice 构建合法）。
    """
    global _abi_ok
    if _abi_ok:
        return
    try:
        from e2m2e._integrators import _py_abi_version
    except ImportError:
        _abi_ok = True  # 无扩展 → 保留降级，不报
        return
    actual = _py_abi_version()
    if actual < _MIN_REQUIRED_RUST_ABI:
        raise RuntimeError(
            f"e2m2e._integrators 编译产物过期（ABI v{actual} < 所需 v{_MIN_REQUIRED_RUST_ABI}）。"
            "请重建 Rust 扩展：uv run maturin develop --features spice"
        )
    _abi_ok = True


__all__ = [
    "augmented_eom_7d_py",
    "build_cr3bp_hamiltonian_py",
    "check_collision_py",
    "CowellResult",
    "cowell_step",
    "compute_distance_series_py",
    "compute_min_distance_py",
    "detect_intersection_py",
    "detect_local_minimum_py",
    "disable_ephem_cache",
    "enable_ephem_cache",
    "ephem_ffi_call_count",
    "grid_search_rust_serial",
    "hello_integrators",
    "indirect_term_acceleration",
    "initialize_abm_history",
    "initialize_cowell_history",
    "lambert_batch_py",
    "lambert_izzo_py",
    "MultistepMethod",
    "MultistepResult",
    "multistep_step",
    "multiple_shooting_correct_py",
    "pole_tide",
    "project_hamiltonian_qf_py",
    "propagate_compiled",
    "propagate_compiled_lowthrust",
    "propagate_compiled_lowthrust_sensitivity",
    "propagate_compiled_stm_py",
    "propagate_bcr4bp_py",
    "propagate_bcr4bp_stm_py",
    "propagate_cr3bp_py",
    "propagate_cr3bp_stm_py",
    "propagate_with_state_py",
    "propagate_with_stm_py",
    "reset_ephem_ffi_call_count",
    "rk_step",
    "RkMethod",
    "segmented_shooting_correct_py",
    "solid_tide_step1",
    "solid_tide_step2",
    "solve_ivp_events",
    "solve_ivp_events_py",
    "spice_poc_furnsh",
    "spherical_harmonic_accel",
    "third_body_acceleration",
    "TransferPointResult",
    "transfer_grid_search_serial_py",
]


def rk_step(
    method: RkMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    state_error_dim: int | None = None,
):
    """Take a single Runge-Kutta step using the Rust integrator core.

    The callback ``f`` receives a NumPy ndarray and must return one of the same
    length. Returns a ``StepResult`` with ``y_new``, ``error``, ``h_next``.

    ``state_error_dim``：步长误差控制只统计前 N 维（``None`` 时统计全部）。
    STM 增广传播时传 6，让状态转移矩阵的 36 个分量不主导步长控制。
    """
    if CowellResult is not None:
        _check_rust_abi()
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    return _rk_step(method, t, y.tolist(), h, tol, _adapt, state_error_dim)


def multistep_step(
    method: MultistepMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    history: list[npt.ArrayLike],
):
    """Take a single multistep predictor-corrector step.

    ``history`` must hold ``method.steps()`` derivative samples (oldest first),
    each the same length as ``y``, at equal spacing ``h``. The callback ``f``
    has the same signature as for :func:`rk_step`. Returns a
    ``MultistepResult`` whose ``history`` is the rolled buffer for the next step.

    The step size is assumed fixed; changing ``h`` requires re-initialising the
    history (see :func:`initialize_abm_history`).
    """
    if CowellResult is not None:
        _check_rust_abi()
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    hist_lists = [np.asarray(hi, dtype=float).tolist() for hi in history]
    return _multistep_step(method, t, y.tolist(), h, tol, _adapt, hist_lists)


def initialize_abm_history(
    t0: float,
    y0: npt.ArrayLike,
    h: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    n_stages: int = 3,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, list[list[float]]]:
    """Bootstrap the ABM history by running ``n_stages`` RK89 steps.

    The ABM method consumes 4 derivative samples; with the default
    ``n_stages=3`` this returns ``(t0 + 3h, y(3h), [f_0, f_1, f_2, f_3])``.
    The returned history is ready to feed into :func:`multistep_step`.
    """
    if CowellResult is not None:
        _check_rust_abi()
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    history: list[list[float]] = [np.asarray(f(t, y), dtype=float).tolist()]
    for _ in range(n_stages):
        result = rk_step(RkMethod.RK89, t, y, h, tol, f)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        history.append(np.asarray(f(t, y), dtype=float).tolist())
    return t, y, history


def cowell_step(
    t: float,
    h: float,
    tol: float,
    accel: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    history: list[npt.ArrayLike],
):
    """Take a single Cowell (Störmer-Cowell) 8th-order step for ``x'' = a(t, x)``.

    ``history`` = ``[x_{n-1}, x_n, a_{n-7}, ..., a_n]`` (10 vectors: 2 position
    samples + 8 acceleration samples, oldest first). ``accel(t, x)`` returns the
    acceleration depending on position only (gravity, J2). Output is position
    only. Fixed step.

    Returns a ``CowellResult`` with ``x_new``, ``error``, ``h_next``, ``history``.
    """
    if CowellResult is not None:
        _check_rust_abi()
    hist_lists = [np.asarray(hi, dtype=float).tolist() for hi in history]

    def _adapt(t_i: float, x_i: list[float]) -> list[float]:
        x_arr = np.asarray(x_i, dtype=float)
        result = accel(t_i, x_arr)
        return np.asarray(result, dtype=float).tolist()

    return _cowell_step(t, h, tol, _adapt, hist_lists)


def initialize_cowell_history(
    t0: float,
    x0: npt.ArrayLike,
    v0: npt.ArrayLike,
    h: float,
    accel: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    n_startup: int = 7,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, list[list[float]]]:
    """Bootstrap the 8th-order Cowell history via ``n_startup`` RK89 steps.

    Returns ``(t, x, v, history)`` with
    ``history = [x_{n-1}, x_n, a_{n-7}, ..., a_n]`` (2 positions + 8
    accelerations, ready for :func:`cowell_step`). ``n_startup`` must be ≥ 7 so
    the 8 most recent acceleration samples are available; with the default
    ``n_startup=7`` the state advances to ``t0 + 7h``.
    """
    if CowellResult is not None:
        _check_rust_abi()
    if n_startup < 7:
        raise ValueError(
            f"8th-order Cowell needs n_startup >= 7 (8 acceleration samples), got {n_startup}"
        )
    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    t = float(t0)
    d = len(x)
    xs = [x.copy()]
    accels = [np.asarray(accel(t, x), dtype=float)]

    def _first_order(t_i: float, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[d:], np.asarray(accel(t_i, state[:d]), dtype=float)])

    for _ in range(n_startup):
        result = rk_step(RkMethod.RK89, t, np.concatenate([x, v]), h, tol, _first_order)
        y = np.asarray(result.y_new, dtype=float)
        x, v = y[:d], y[d:]
        t += h
        xs.append(x.copy())
        accels.append(np.asarray(accel(t, x), dtype=float))

    history = [xs[-2].tolist(), xs[-1].tolist()] + [a.tolist() for a in accels[-8:]]
    return t, x, v, history


def solve_ivp_events(
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    rtol: float,
    atol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    events: list[tuple[Callable[[float, npt.NDArray[np.floating]], float], bool, float]],
    method: RkMethod | None = None,
    max_step: float | None = None,
    max_steps: int | None = None,
    state_error_dim: int | None = None,
) -> dict[str, Any]:
    """带事件检测的 Rust solve_ivp 封装（scipy 事件语义）。

    事件检测在 Rust 积分内循环完成：每个接受步的端点评估事件函数，
    符号变化（经 direction 过滤）时在步内对线性插值态二分求精（无稠密输出）。

    Args:
        t_span: 积分区间 ``(t0, tf)``。
        y0: 初始状态向量。
        t_eval: 输出时间点数组。
        rtol: 相对容差。
        atol: 绝对容差。
        f: ODE 右端函数 ``f(t, y) -> dy/dt``。
        events: ``[(g, terminal, direction), ...]``，``g(t, y) -> float``，
            零点即事件面；``terminal=True`` 触发即停；``direction`` > 0 只记
            上行穿越（g 由负到正）、< 0 只记下行、0 双向。
        method: RK 方法，默认 PD78（DOP853）。
        max_step: 最大步长。求精精度受步内线性插值误差（``~h²/8·|ÿ|``）限制，
            需要更紧的事件时刻时请设小 max_step。
        max_steps: 最大积分步数。
        state_error_dim: 步长误差控制只统计前 N 维（用于 STM 增广传播）。

    Returns:
        dict：``states``/``time``（t_eval 前缀，terminal 截断时末点为求精后的
        事件点）、``t_events``/``y_events``（逐事件的触发时刻与状态列表）、
        ``terminal_event``（触发终止的事件索引或 None）、``n_steps``。
    """
    if CowellResult is not None:
        _check_rust_abi()
    y0_arr = np.asarray(y0, dtype=float)

    def _adapt_rhs(t_i: float, y_i: list[float]) -> list[float]:
        return np.asarray(f(t_i, np.asarray(y_i, dtype=float)), dtype=float).tolist()

    def _adapt_event(
        g: Callable[[float, npt.NDArray[np.floating]], float],
    ) -> Callable[[float, list[float]], float]:
        def _g(t_i: float, y_i: list[float]) -> float:
            return float(g(t_i, np.asarray(y_i, dtype=float)))

        return _g

    event_specs = [
        (_adapt_event(g), terminal, float(direction)) for g, terminal, direction in events
    ]
    return solve_ivp_events_py(
        (float(t_span[0]), float(t_span[1])),
        y0_arr.tolist(),
        [float(t) for t in np.asarray(t_eval, dtype=float).flat],
        float(rtol),
        float(atol),
        _adapt_rhs,
        event_specs,
        method,
        max_step,
        max_steps,
        state_error_dim,
    )


def grid_search_rust_serial(
    dep_states: npt.ArrayLike,
    dep_times: npt.ArrayLike,
    alpha_grid: npt.ArrayLike,
    arrival_states: npt.ArrayLike,
    *,
    mu: float,
    max_transfer_time: float,
    integration_dt: float,
    intersection_threshold: float,
    min_distance_threshold: float,
    collision_earth_radius: float,
    collision_moon_radius: float,
    rtol: float,
    atol: float,
    max_step: float,
) -> list[dict[str, Any]]:
    """转移网格搜索 Rust 串行后端（阶段 B）。

    展平 POD 输入 → 调 ``transfer_grid_search_serial_py`` → 转 ``list[dict]``。
    返回字段对齐 ``search_parallel.grid_search_sequential``，便于逐候选等价
    对照（整数索引精确相等、浮点 ``allclose``）。

    本 wrapper 只做数组→dict 转换，不依赖 transfer 算法层（分层：算法层调
    数值层合法，数值层不反向依赖）。出发轨道采样（``sample_departure_points``）
    与 Orbit 展平由调用方完成；阶段 D 的 ``grid_search_rust`` 编排器会在此之上
    接入 ``TransferSearch``。

    Args:
        dep_states: ``(n_dep, 6)`` 或展平 ``n_dep*6`` 出发状态。
        dep_times: ``(n_dep,)`` 出发时刻。
        alpha_grid: ``(n_alpha,)`` 切向速度比 α 网格。
        arrival_states: ``(n_arrival, 6)`` 或展平目标轨道状态。
        mu / max_transfer_time / integration_dt / intersection_threshold /
            min_distance_threshold / collision_earth_radius /
            collision_moon_radius: CR3BP 与搜索标量配置。
        rtol / atol / max_step: 积分器容差与最大步长。

    Returns:
        ``list[dict]``，长度 ``n_dep * n_alpha``，顺序为外层 departure、
        内层 alpha（与 ``grid_search_sequential`` 一致）。
    """
    if transfer_grid_search_serial_py is None:
        raise RuntimeError("Rust 扩展未构建（transfer_grid_search_serial_py 不可用）")
    if CowellResult is not None:
        _check_rust_abi()

    dep_states_arr = np.asarray(dep_states, dtype=float).reshape(-1)
    dep_times_arr = np.asarray(dep_times, dtype=float).reshape(-1)
    alpha_arr = np.asarray(alpha_grid, dtype=float).reshape(-1)
    arrival_arr = np.asarray(arrival_states, dtype=float).reshape(-1)

    results = transfer_grid_search_serial_py(
        dep_states_arr.tolist(),
        dep_times_arr.tolist(),
        alpha_arr.tolist(),
        arrival_arr.tolist(),
        float(mu),
        float(max_transfer_time),
        float(integration_dt),
        float(intersection_threshold),
        float(min_distance_threshold),
        float(collision_earth_radius),
        float(collision_moon_radius),
        float(rtol),
        float(atol),
        float(max_step),
    )
    return [_transfer_point_result_to_dict(r) for r in results]


def _transfer_point_result_to_dict(r: TransferPointResult) -> dict[str, Any]:
    """``TransferPointResult`` pyclass → dict，字段对齐 ``search_single_departure``。

    数组字段（``transfer_trajectory``/``intersection_point``/``transfer_times``）
    还原为 numpy 数组，与 Python sequential 后端返回类型一致。
    """
    traj = r.transfer_trajectory
    traj_arr = np.asarray(traj, dtype=float).reshape(-1, 6) if traj is not None else None
    times = r.transfer_times
    times_arr = np.asarray(times, dtype=float) if times is not None else None
    int_pt = r.intersection_point
    int_pt_arr = np.asarray(int_pt, dtype=float) if int_pt is not None else None
    return {
        "success": r.success,
        "departure_state": np.asarray(r.departure_state, dtype=float),
        "departure_time": r.departure_time,
        "alpha": r.alpha,
        "transfer_trajectory": traj_arr,
        "transfer_times": times_arr,
        "transfer_time": r.transfer_time,
        "min_distance": r.min_distance,
        "min_distance_idx": r.min_distance_idx,
        "min_distance_orbit_idx": r.min_distance_orbit_idx,
        "dv_departure": r.dv_departure,
        "dv_insertion": r.dv_insertion,
        "intersection_found": r.intersection_found,
        "intersection_point": int_pt_arr,
        "intersection_idx": r.intersection_idx,
        "first_intersection_idx": r.first_intersection_idx,
        "first_intersection_time": r.first_intersection_time,
        "first_min_distance_idx": r.first_min_distance_idx,
        "first_min_distance_time": r.first_min_distance_time,
        "local_minimum_found": r.local_minimum_found,
        "local_minimum_distance": r.local_minimum_distance,
        "local_minimum_idx": r.local_minimum_idx,
        "collision_found": r.collision_found,
        "collision_body": r.collision_body,
        "collision_idx": r.collision_idx,
        "status": r.status,
    }
