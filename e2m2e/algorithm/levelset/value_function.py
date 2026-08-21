"""值函数网格的任意点值/梯度查询（#499，ADR 0032 决策 4）。

值函数落地后是 ``(时间快照, *状态网格)`` 形状的 numpy 数组。本模块提供
离网点查询：空间维用张量积三次样条（逐轴 not-a-knot 求解组装
``NdBSpline``，网格维节点不足 4 时自动降阶），梯度为**样条的解析导数**
——不是"网格中心差分再插值"；时间维做线性插值，必选。

精度与连续性口径：三次样条值为 O(h⁴)、解析梯度为 O(h³)，相对中心差分
的 O(h²) 高一阶以上；样条 C² 连续，梯度跨网格单元无跳变（闭环控制的
方向与开关函数直接消费梯度，局部滑动模板在单元边界的梯度跳变不可接受）。
对不超过三次的多项式精确（机器精度量级）。

维度无关：接口不假设状态维数、物理含义与单位，2D 与 4D/5D 同一路径。

代价：每次调用为所触及的每个时间快照重建一次张量样条（逐轴带状求解，
41⁴ 量级网格约 0.5 s/快照）。当前在线策略为控制周期级查询，可接受；
若未来密集闭环仿真把查询打成瓶颈，再引入带系数缓存的插值器对象——
那是独立决策，不改变本函数契约。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline

from e2m2e.exceptions import E2M2EError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["ValueFunctionQueryError", "value_function_gradient"]

#: 空间样条最高次数（三次）；网格维节点不足时自动降为 n-1 次
_MAX_DEGREE = 3


class ValueFunctionQueryError(E2M2EError, ValueError):
    """值函数查询越界（状态出网格、时间出时间轴）时抛出。

    同时继承 :class:`ValueError`，调用方可按值错误惯例捕获。
    """


def value_function_gradient(
    axes: tuple[npt.ArrayLike, ...],
    values: npt.ArrayLike,
    times: npt.ArrayLike,
    state: npt.ArrayLike,
    time: float,
) -> tuple[float, npt.NDArray[np.floating]]:
    """查询值函数在任意状态与时刻的值与状态梯度。

    Args:
        axes: 各状态维网格坐标，长度 n 的元组；每个轴一维、严格递增、
            至少 2 个节点。
        values: 值函数数组，形状 ``(len(times), *grid_shape)``。
        times: 时间快照轴，一维严格递增。仅单快照时视为定常场——
            任意 ``time`` 均返回该快照结果，不做越界检查。
        state: 查询状态，形状 ``(n,)``，须落在各轴范围内。
        time: 查询时刻，须落在 ``times`` 范围内（单快照除外）。

    Returns:
        ``(value, gradient)``：标量值与形状 ``(n,)`` 的状态梯度。
        多快照时两相邻快照各自做空间样条求值再按时间线性混合。

    Raises:
        ValueFunctionQueryError: 状态出网格或时间出时间轴。
        ValueError: 输入形状或单调性不合法。
    """
    axes_np, values_np, times_np, state_np = _validate(axes, values, times, state)

    for dim, (axis, coordinate) in enumerate(zip(axes_np, state_np, strict=True)):
        if not (axis[0] <= coordinate <= axis[-1]):
            raise ValueFunctionQueryError(
                f"状态第 {dim} 维坐标 {coordinate!r} 超出网格 [{axis[0]!r}, {axis[-1]!r}]"
            )

    if times_np.size == 1:
        return _spatial_query(axes_np, values_np[0], state_np)

    if not (times_np[0] <= time <= times_np[-1]):
        raise ValueFunctionQueryError(
            f"时间 {time!r} 超出时间轴 [{times_np[0]!r}, {times_np[-1]!r}]"
        )
    hi = int(np.searchsorted(times_np, time, side="right"))
    lo = max(0, min(hi - 1, times_np.size - 2))
    hi = lo + 1
    alpha = float((time - times_np[lo]) / (times_np[hi] - times_np[lo]))

    value_lo, gradient_lo = _spatial_query(axes_np, values_np[lo], state_np)
    if alpha == 0.0:
        return value_lo, gradient_lo
    value_hi, gradient_hi = _spatial_query(axes_np, values_np[hi], state_np)
    return (
        (1.0 - alpha) * value_lo + alpha * value_hi,
        (1.0 - alpha) * gradient_lo + alpha * gradient_hi,
    )


def _validate(
    axes: tuple[npt.ArrayLike, ...],
    values: npt.ArrayLike,
    times: npt.ArrayLike,
    state: npt.ArrayLike,
) -> tuple[
    tuple[npt.NDArray[np.floating], ...],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    if not axes:
        raise ValueError("axes 不能为空")
    axes_np = tuple(np.asarray(axis, dtype=float) for axis in axes)
    for dim, axis in enumerate(axes_np):
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"axes[{dim}] 须为一维且至少 2 个节点")
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"axes[{dim}] 须严格递增")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"axes[{dim}] 含非有限值")

    values_np = np.asarray(values, dtype=float)
    grid_shape = tuple(axis.size for axis in axes_np)
    if values_np.ndim != len(axes_np) + 1 or values_np.shape[1:] != grid_shape:
        raise ValueError(f"values 形状须为 (len(times), {grid_shape})，当前 {values_np.shape}")

    times_np = np.asarray(times, dtype=float)
    if times_np.ndim != 1 or times_np.size < 1:
        raise ValueError("times 须为一维非空")
    if values_np.shape[0] != times_np.size:
        raise ValueError(f"values 时间维 {values_np.shape[0]} 与 times 长度 {times_np.size} 不一致")
    if times_np.size > 1 and not np.all(np.diff(times_np) > 0):
        raise ValueError("times 须严格递增")

    state_np = np.asarray(state, dtype=float).reshape(-1)
    if state_np.size != len(axes_np):
        raise ValueError(f"state 维数须为 {len(axes_np)}，当前 {state_np.size}")
    if not np.all(np.isfinite(state_np)):
        raise ValueError("state 含非有限值")
    return axes_np, values_np, times_np, state_np


def _build_spline(
    axes: tuple[npt.NDArray[np.floating], ...], snapshot: npt.NDArray[np.floating]
) -> NdBSpline:
    """对单个时间快照构造张量积插值样条（逐轴 not-a-knot，边界维自动降阶）。"""
    coefficients = np.ascontiguousarray(snapshot, dtype=float)
    knots: list[npt.NDArray[np.floating]] = []
    degrees: list[int] = []
    for dim, axis in enumerate(axes):
        degree = min(_MAX_DEGREE, axis.size - 1)
        rolled = np.moveaxis(coefficients, dim, 0)
        spline_1d = make_interp_spline(axis, rolled, k=degree, axis=0)
        coefficients = np.moveaxis(spline_1d.c, 0, dim)
        knots.append(spline_1d.t)
        degrees.append(degree)
    return NdBSpline(tuple(knots), np.ascontiguousarray(coefficients), tuple(degrees))


def _spatial_query(
    axes: tuple[npt.NDArray[np.floating], ...],
    snapshot: npt.NDArray[np.floating],
    state: npt.NDArray[np.floating],
) -> tuple[float, npt.NDArray[np.floating]]:
    """单时间快照的空间样条求值：返回值与该点解析梯度（``nu`` 逐维求导）。"""
    spline = _build_spline(axes, snapshot)
    point = state[None, :]
    n_dim = len(axes)
    value = float(spline(point, nu=tuple([0] * n_dim))[0])
    gradient = np.empty(n_dim)
    for dim in range(n_dim):
        nu = tuple(1 if d == dim else 0 for d in range(n_dim))
        gradient[dim] = spline(point, nu=nu)[0]
    return value, gradient
