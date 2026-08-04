"""LPO（Long-Period Orbit）修正策略。

L4/L5 长周期族是 xy 平面内围绕三角平动点的周期轨道，
不具有 x 轴或 xz 平面对称性。使用通用平面周期修正（全周期闭合）。
最大振幅成员呈马蹄形（Horseshoe），跨越 L4-L1-L5。

References:
    Gómez et al. (2001). Dynamics and mission design near libration
    points, Vol. II. ESA Contract Report.
    Marchal, C. (1990). The Three-Body Problem. Brown 猜想 C.2.
"""

from __future__ import annotations

from .base import CorrectionConfig


def lpo_fixed_x0(x0: float, libration_point: int = 5) -> CorrectionConfig:
    """固定 x₀ 的 LPO 通用平面周期修正。

    LPO 无 x 轴对称性（y₀≠0），与 SPO 使用相同的修正框架。
    直接求解全周期闭合条件：state(T) - state(0) = 0。

    自由变量: [y₀, ẋ₀, ẏ₀, T]（4 个，x₀ 固定，z₀=ż₀=0 平面约束）
    约束: [Δy=0, Δẋ=0, Δẏ=0]（3 个，全周期闭合）
    → 4 自由 vs 3 约束，欠定系统用最小二乘求解（最小范数修正）。

    Args:
        x0: 固定的初始 x 坐标（族参数）。
        libration_point: 平动点编号（4=L4, 5=L5），默认 5。
    """
    return CorrectionConfig(
        setup_type="lpo_fixed_x0",
        symmetry_condition="none",
        fixed_parameters={"x0": x0, "libration_point": libration_point},
        free_variables=["y0", "vx0", "vy0", "T_full"],
        free_variable_indices=[1, 3, 4, 6],
        target_conditions={"dy": 0.0, "dvx": 0.0, "dvy": 0.0},
        constraint_indices=[1, 3, 4],
        constraint_weights={"dy": 1.0, "dvx": 1.0, "dvy": 1.0},
        constraint_types={
            "dy": "closure",
            "dvx": "closure",
            "dvy": "closure",
        },
    )
