"""
LEO (Low Earth Orbit) 工具函数

在 CR3BP 归一化坐标系中定义 LEO 参数和辅助函数。
LEO 建模为固定半径球面（以地心为圆心），非 CR3BP 周期轨道。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from e2m2e.core import CR3BP_System

_em = CR3BP_System.from_known_system("earth_moon")
MU: float = _em.mu
DU: float = _em.DU

R_EARTH_KM: float = 6371.0

LEO_ALT_400KM: float = 400.0
R_LEO_400KM: float = (R_EARTH_KM + LEO_ALT_400KM) / DU
V_CIRCULAR_LEO_400KM: float = float(np.sqrt((1.0 - MU) / R_LEO_400KM))
T_LEO_400KM: float = float(2.0 * np.pi * R_LEO_400KM / V_CIRCULAR_LEO_400KM)

LEO_ALT_200KM: float = 200.0
R_LEO_200KM: float = (R_EARTH_KM + LEO_ALT_200KM) / DU
V_CIRCULAR_LEO_200KM: float = float(np.sqrt((1.0 - MU) / R_LEO_200KM))
T_LEO_200KM: float = float(2.0 * np.pi * R_LEO_200KM / V_CIRCULAR_LEO_200KM)

R_LEO: float = R_LEO_400KM
V_CIRCULAR_LEO: float = V_CIRCULAR_LEO_400KM
T_LEO: float = T_LEO_400KM
LEO_ALT_KM: float = LEO_ALT_400KM

EARTH_CENTER: npt.NDArray[np.floating] = np.array([-MU, 0.0, 0.0])


def leo_circular_velocity_rotating(
    position: npt.NDArray[np.floating], r_leo: float = R_LEO
) -> npt.NDArray[np.floating]:
    """计算旋转系下 LEO 圆轨道速度。"""
    r_rel = position - EARTH_CENTER
    r_rel_xy = np.sqrt(r_rel[0] ** 2 + r_rel[1] ** 2)
    v_circ = np.sqrt((1.0 - MU) / r_leo)

    if r_rel_xy < 1e-12:
        return np.array([0.0, v_circ, 0.0])

    tangential = np.array([-r_rel[1], r_rel[0], 0.0]) / r_rel_xy
    v_inertial = v_circ * tangential

    omega_cross_r = np.array([-position[1], position[0], 0.0])
    return v_inertial - omega_cross_r


def generate_leo_orbit_states(
    n_points: int = 500, r_leo: float = R_LEO
) -> npt.NDArray[np.floating]:
    """生成 LEO 近似圆轨道状态数组。"""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    states = np.zeros((n_points, 6))

    for i, th in enumerate(theta):
        x = EARTH_CENTER[0] + r_leo * np.cos(th)
        y = r_leo * np.sin(th)
        z = 0.0

        pos = np.array([x, y, z])
        vel = leo_circular_velocity_rotating(pos, r_leo)

        states[i] = [x, y, z, vel[0], vel[1], vel[2]]

    return states
