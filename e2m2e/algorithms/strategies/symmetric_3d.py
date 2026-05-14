"""三维对称修正策略（空间 CR3BP）。"""

from __future__ import annotations

from .base import CorrectionConfig


def symmetric_3d_fixed_x0(x0: float) -> CorrectionConfig:
    """固定 x0 的三维对称修正：自由变量为 z0、y_dot0 和 T_half。

    用于具有 x 轴对称性的空间周期轨道（如 Halo 轨道）。

    Args:
        x0: 固定的初始 x 坐标。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_x_fixed_x0",
        symmetry_condition="x_axis",
        fixed_parameters={"x0": x0},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
        constraint_weights={"y": 1.0, "x_dot": 1.0, "z_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality", "z_dot": "equality"},
    )


def symmetric_xz_fixed_x0(x0: float) -> CorrectionConfig:
    """固定 x0 的 XZ 平面对称修正：自由变量为 z0、y_dot0 和 T_half。

    Args:
        x0: 固定的初始 x 坐标。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_xz_fixed_x0",
        symmetry_condition="xz_plane",
        fixed_parameters={"x0": x0},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )


def symmetric_xz_fixed_z0(z0: float) -> CorrectionConfig:
    """固定 z0 的 XZ 平面对称修正：自由变量为 x0、y_dot0 和 T_half。

    Args:
        z0: 固定的初始 z 坐标。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_xz_fixed_z0",
        symmetry_condition="xz_plane",
        fixed_parameters={"z0": z0},
        free_variables=["x0", "y_dot0", "T_half"],
        free_variable_indices=[0, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )
