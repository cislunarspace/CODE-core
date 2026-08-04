"""Axial 轨道微分修正策略（Gómez Type B 分岔族）。"""

from __future__ import annotations

from .base import CorrectionConfig


def axial_fixed_vz0(vz0: float, libration_point: int = 1) -> CorrectionConfig:
    """固定 vz0 的 Axial 轨道修正（x 轴对称，Type B）。

    Axial 轨道关于 x 轴对称（旋转 π + 时间反演），初始状态在 x 轴上：
    (x0, 0, 0, 0, y_dot0, vz0)，半周期处回到 x 轴（y=0, z=0, x_dot=0）。

    与 Halo（Type A, xz 平面对称, z0≠0, vz0=0）的区别：
    Axial 的 z0=0, vz0≠0——轨道从 xy 平面出发，获得面外速度后
    在半周期返回 xy 平面（z 翻转：z(t) = -z(T-t)）。

    自由变量为 x0、y_dot0 和 T_half（3 自由变量 vs 3 约束，方阵）。

    Args:
        vz0: 固定的初始 z 方向速度（无量纲 DU/TU），带符号区分上/下族。
        libration_point: 拉格朗日点编号（1=L1, 2=L2）。

    Returns:
        包含对应修正参数的 CorrectionConfig。
    """
    return CorrectionConfig(
        setup_type="axial_orbit_fixed_vz0",
        symmetry_condition="x_axis",
        fixed_parameters={"z0": 0.0, "vz0": vz0, "libration_point": libration_point},
        free_variables=["x0", "y_dot0", "T_half"],
        free_variable_indices=[0, 4, 6],
        target_conditions={"y": 0.0, "z": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 2, 3],
    )
