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
        lambert_batch_py,
        lambert_izzo_py,
        solve_ivp_events_py,
    )
    from e2m2e._integrators import cowell_step as _cowell_step
    from e2m2e._integrators import multistep_step as _multistep_step
    from e2m2e._integrators import rk_step as _rk_step
except ModuleNotFoundError:
    # _integrators is a compiled Rust extension; allow import for doc builds
    # where the extension is not built.
    CowellResult = None  # type: ignore[misc,assignment]
    MultistepMethod = None  # type: ignore[misc,assignment]
    MultistepResult = None  # type: ignore[misc,assignment]
    RkMethod = None  # type: ignore[misc,assignment]
    lambert_batch_py = None  # type: ignore[misc,assignment]
    lambert_izzo_py = None  # type: ignore[misc,assignment]
    solve_ivp_events_py = None  # type: ignore[misc,assignment]
    _cowell_step = None  # type: ignore[misc,assignment]
    _multistep_step = None  # type: ignore[misc,assignment]
    _rk_step = None  # type: ignore[misc,assignment]

__all__ = [
    "rk_step",
    "RkMethod",
    "multistep_step",
    "MultistepMethod",
    "MultistepResult",
    "initialize_abm_history",
    "cowell_step",
    "CowellResult",
    "initialize_cowell_history",
    "lambert_izzo_py",
    "lambert_batch_py",
    "solve_ivp_events",
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
