"""Halo 轨道微分修正策略。"""

from __future__ import annotations

from .base import CorrectionConfig


def halo_fixed_z0(z0: float, libration_point: int = 1) -> CorrectionConfig:
    """固定 z0 的 Halo 轨道修正（XZ 平面对称）。

    自由变量为 x0、y_dot0 和 T_half。平动点编号作为固定参数记录，
    供延拓器查询使用。

    Args:
        z0: 固定的初始 z 坐标。
        libration_point: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="halo_orbit_fixed_z0",
        symmetry_condition="xz_plane",
        fixed_parameters={"z0": z0, "libration_point": libration_point},
        free_variables=["x0", "y_dot0", "T_half"],
        free_variable_indices=[0, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )


def halo_fixed_x0(x0: float, libration_point: int = 1) -> CorrectionConfig:
    """固定 x0 的 Halo 轨道修正（XZ 平面对称）。

    自由变量为 z0、y_dot0 和 T_half。

    Args:
        x0: 固定的初始 x 坐标。
        libration_point: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="halo_orbit_fixed_x0",
        symmetry_condition="xz_plane",
        fixed_parameters={"x0": x0, "libration_point": libration_point},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )
