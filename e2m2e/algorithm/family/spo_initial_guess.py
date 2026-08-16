"""SPO（Short-Period Orbit）初猜模块。

从 L4/L5 线性化动力学中提取短周期模态（频率 ω_s），构造仅含
短周期分量的平面初猜。不含长周期和垂直模态——它们会导致拟周期
运动，阻碍全周期闭合修正的收敛。

References:
    Gómez et al. (2001). Dynamics and mission design near libration
    points, Vol. II. ESA Contract Report.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.templates.enums import LibrationPoint
from ..dynamics import CR3BP_System
from .triangular_initial_guess import _triangular_modes

#: L4/L5 编号 → LibrationPoint 枚举
_TRIANGULAR = {4: LibrationPoint.L4, 5: LibrationPoint.L5}


def compute_spo_initial_guess(
    system: CR3BP_System,
    point: int,
    amplitude_km: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """构造 L4/L5 SPO 初猜状态（仅短周期模态）。

    从 L4/L5 线性化矩阵提取短周期模态 v_s（高频），施加面内
    振幅扰动，不含长周期和垂直模态。

    初猜形状为近似椭圆（主导谐波 n=1），适合全周期闭合修正。

    Args:
        system: CR3BP 系统（含已计算平动点）。
        point: 4（L4）或 5（L5）。
        amplitude_km: 面内振幅（km），扰动幅度。

    Returns:
        (state0, nominal_period)：t=0 的 6 维状态（无量纲 synodic）与
        短周期名义周期 2π/ω_s。
    """
    if point not in _TRIANGULAR:
        raise ValueError(f"point 必须为 4（L4）或 5（L5），当前 {point}")

    omega_s, v_s, _omega_l, v_l, _omega_v, v_z, x_L = _triangular_modes(system, point)
    l_c = system.characteristic_length
    assert l_c is not None

    # 仅使用短周期模态，振幅归一化
    alpha_s = float((amplitude_km / l_c) / np.linalg.norm(v_s[:3]))

    phi = 0.0  # 初始相位
    mode_contrib = alpha_s * (np.real(v_s) * np.cos(phi) - np.imag(v_s) * np.sin(phi))

    state0 = np.zeros(6)
    state0[:3] = x_L
    state0 += mode_contrib

    nominal_period = 2.0 * np.pi / omega_s
    return state0, nominal_period
