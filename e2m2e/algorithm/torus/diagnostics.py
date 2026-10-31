"""不变环面的定义级诊断量。

三个度量互相独立，输入只依赖曲线采样、旋转数、频闪周期与动力学：
不变性残差（频闪像落回旋转曲线的偏离）、共轭性残差（各采样点推进角
与旋转数的偏离）、长时到环面距离（多周期传播后到曲线的最大距离）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from e2m2e.algorithm.torus.construction import (
    _band_limited,
    _strobe_images,
    _wrap_angle,
    evaluate_torus_curve,
)

if TYPE_CHECKING:
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics

__all__ = [
    "conjugacy_residual",
    "invariance_residual",
    "torus_long_time_distance",
]

#: 最近点相位搜索的粗扫描倍数（每格点再加密的相位数）。
_COARSE_OVERSAMPLE = 8

#: 最近点相位的牛顿加细次数。
_REFINE_ITERATIONS = 3


def _curve_and_derivative(curve: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """带限曲线值与对相位的导数（复用 construction 的单一份实现）。"""
    return _band_limited(curve, phases)


def _closest_phase(curve: np.ndarray, point: np.ndarray) -> float:
    """曲线上距 point 最近的相位：粗扫描定初猜，牛顿加细。"""
    n = curve.shape[0]
    coarse_count = _COARSE_OVERSAMPLE * n
    coarse = np.arange(coarse_count) * (2.0 * np.pi / coarse_count)
    values = evaluate_torus_curve(curve, coarse)
    phase = float(coarse[int(np.argmin(np.linalg.norm(values - point[None, :], axis=1)))])
    for _ in range(_REFINE_ITERATIONS):
        step_values, derivative = _curve_and_derivative(curve, np.array([phase]))
        difference = step_values[0] - point
        speed = float(derivative[0] @ derivative[0])
        if speed <= 1e-300:
            break
        phase -= float(difference @ derivative[0]) / speed
    return phase % (2.0 * np.pi)


def invariance_residual(
    curve_states: npt.ArrayLike,
    rotation_number: float,
    strobe_period: float,
    dynamics: CR3BP_Dynamics,
) -> float:
    """不变性残差：频闪像与旋转后曲线的逐采样偏离（格点最大范数）。

    Args:
        curve_states: 闭曲线网格采样，形状 (n_points, 6)。
        rotation_number: 旋转数 ρ（rad）。
        strobe_period: 频闪周期 τ（TU）。
        dynamics: CR3BP 动力学（积分容差随 dynamics 实例）。

    Returns:
        max_j ‖P_τ(C_j) − C(φ_j+ρ)‖_∞。
    """
    curve = np.asarray(curve_states, dtype=float)
    n = curve.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    images, _ = _strobe_images(dynamics, curve, float(strobe_period), with_stm=False)
    shifted = evaluate_torus_curve(curve, phi + float(rotation_number))
    return float(np.max(np.abs(images - shifted)))


def conjugacy_residual(
    curve_states: npt.ArrayLike,
    rotation_number: float,
    strobe_period: float,
    dynamics: CR3BP_Dynamics,
) -> float:
    """共轭性残差：各采样点频闪像的落点相位推进与 ρ 的偏离（rad）。

    落点相位取曲线上距频闪像最近的相位（粗扫描加牛顿加细），残差为偏离
    的最大绝对值（相位差取主值）。

    Args:
        curve_states: 闭曲线网格采样，形状 (n_points, 6)。
        rotation_number: 旋转数 ρ（rad）。
        strobe_period: 频闪周期 τ（TU）。
        dynamics: CR3BP 动力学。

    Returns:
        max_j |wrap(φ'(P_τ(C_j)) − φ_j − ρ)|。
    """
    curve = np.asarray(curve_states, dtype=float)
    n = curve.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    images, _ = _strobe_images(dynamics, curve, float(strobe_period), with_stm=False)
    worst = 0.0
    for j in range(n):
        target = float(phi[j] + rotation_number)
        landed = _closest_phase(curve, images[j])
        worst = max(worst, abs(_wrap_angle(landed - target)))
    return worst


def torus_long_time_distance(
    curve_states: npt.ArrayLike,
    strobe_period: float,
    dynamics: CR3BP_Dynamics,
    *,
    n_periods: int = 40,
    sample_index: int = 0,
) -> float:
    """长时到环面距离：从曲线出发多周期传播后各落点到曲线的最大距离。

    距离取六维状态空间的 2 范数（无量纲混合单位）。读数要区分两种贡
    献：曲线的不变性误差随周期数缓慢积累；种子轨道的双曲乘子按周期数
    指数放大传播舍入误差（如 L2 Halo 单周期放大上千倍），强双曲种子
    的长弧读数由后者主导，不代表曲线失真。

    Args:
        curve_states: 闭曲线网格采样，形状 (n_points, 6)。
        strobe_period: 频闪周期 τ（TU）。
        dynamics: CR3BP 动力学。
        n_periods: 传播的频闪周期数。
        sample_index: 出发采样点下标。

    Returns:
        max_k ‖C(φ'_k) − x(kτ)‖₂，x(0) 为出发采样点。
    """
    curve = np.asarray(curve_states, dtype=float)
    tau = float(strobe_period)
    start = curve[int(sample_index)]
    outputs = dynamics.propagate(
        np.asarray(start, dtype=float),
        (0.0, n_periods * tau),
        t_eval=[(k + 1) * tau for k in range(n_periods)],
    )
    states = np.asarray(outputs["states"], dtype=float)
    worst = 0.0
    for state in states:
        nearest = evaluate_torus_curve(curve, np.array([_closest_phase(curve, state)]))[0]
        worst = max(worst, float(np.linalg.norm(nearest - state)))
    return worst
