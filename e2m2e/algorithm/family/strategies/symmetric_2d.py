"""二维对称修正策略（平面 CR3BP）。"""

from __future__ import annotations

from .base import CorrectionConfig


def symmetric_2d_fixed_x0(x0: float = 0.0) -> CorrectionConfig:
    """固定 x0 的二维对称修正：自由变量为 y_dot0 和 T_half。

    用于在起点和半周期处均垂直穿越 x 轴的平面对称周期轨道。

    Args:
        x0: 固定的初始 x 坐标。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_x_fixed_x0",
        symmetry_condition="x_axis",
        fixed_parameters={"x0": x0},
        free_variables=["y_dot0", "T_half"],
        free_variable_indices=[4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 3],
        constraint_weights={"y": 1.0, "x_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality"},
    )


def symmetric_2d_fixed_t(t_half: float) -> CorrectionConfig:
    """固定半周期的二维对称修正：自由变量为 x0 和 y_dot0。

    Args:
        t_half: 固定的半周期值。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_x_fixed_t",
        symmetry_condition="x_axis",
        fixed_parameters={"T_half": t_half},
        free_variables=["x0", "y_dot0"],
        free_variable_indices=[0, 4],
        target_conditions={"y": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 3],
        constraint_weights={"y": 1.0, "x_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality"},
    )


def symmetric_2d_fixed_y0(y0: float = 0.0) -> CorrectionConfig:
    """固定 y0 的 y 轴对称修正：自由变量为 x_dot0 和 T_half。

    注意：y 轴半周期条件（x=0 且 x_dot=0 于 T_half）本身不构成闭合
    条件——CR3BP 对 μ≠1/2 没有 y 轴镜面定理，收敛解一般不在 2·T_half
    闭合（#627 实测）。RO 设计走 x 轴固定半周期策略（
    ``symmetric_2d_fixed_t``，见 ``cr3bp_orbits.design_ro``），本策略
    仅保留给研究用。

    Args:
        y0: 固定的初始 y 坐标。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_y_fixed_y0",
        symmetry_condition="y_axis",
        fixed_parameters={"y0": y0},
        free_variables=["x_dot0", "T_half"],
        free_variable_indices=[3, 6],
        target_conditions={"x": 0.0, "x_dot": 0.0},
        constraint_indices=[0, 3],
        constraint_weights={"x": 1.0, "x_dot": 1.0},
        constraint_types={"x": "equality", "x_dot": "equality"},
    )
