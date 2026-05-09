"""
GEO (Geostationary Earth Orbit) 工具函数

在 CR3BP 归一化坐标系中定义 GEO 参数和辅助函数。
GEO 建模为固定半径球面（以地心为圆心），非 CR3BP 周期轨道。

归一化参数:
    r_GEO = 42164 / 384405 ≈ 0.10968 DU
    v_circular = sqrt((1-μ)/r_GEO) ≈ 3.001 VU ≈ 3071 m/s
    T_GEO ≈ 0.2296 TU ≈ 1 day
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from e2m2e.core import CR3BP_System

_em = CR3BP_System.from_known_system("earth_moon")
MU: float = _em.mu
DU: float = _em.DU
TU: float = _em.TU
VU: float = _em.VU

R_GEO_KM: float = 42164.0
R_GEO: float = R_GEO_KM / DU
V_CIRCULAR_GEO: float = float(np.sqrt((1.0 - MU) / R_GEO))
T_GEO: float = float(2.0 * np.pi * R_GEO / V_CIRCULAR_GEO)
EARTH_CENTER: npt.NDArray[np.floating] = np.array([-MU, 0.0, 0.0])


def geo_circular_velocity_rotating(position: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """计算旋转系下 GEO 圆轨道速度。"""
    r_rel = position - EARTH_CENTER
    r_rel_xy = np.sqrt(r_rel[0] ** 2 + r_rel[1] ** 2)
    if r_rel_xy < 1e-12:
        return np.array([0.0, V_CIRCULAR_GEO, 0.0])

    tangential = np.array([-r_rel[1], r_rel[0], 0.0]) / r_rel_xy
    v_inertial = V_CIRCULAR_GEO * tangential

    omega_cross_r = np.array([-r_rel[1], r_rel[0], 0.0])
    return v_inertial - omega_cross_r


def detect_geo_sphere_crossing(
    trajectory_states: npt.NDArray[np.floating],
    r_geo: float = R_GEO,
    earth_center: npt.NDArray[np.floating] = EARTH_CENTER,
) -> tuple[bool, int, float]:
    """检测轨迹是否穿越 GEO 球面。"""
    positions = trajectory_states[:, :3]
    dists = np.linalg.norm(positions - earth_center, axis=1)
    diff = dists - r_geo
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        idx = int(sign_changes[0])
        return True, idx, float(dists[idx])

    return False, -1, 0.0


def find_closest_approach_to_geo(
    trajectory_states: npt.NDArray[np.floating],
    r_geo: float = R_GEO,
    earth_center: npt.NDArray[np.floating] = EARTH_CENTER,
) -> tuple[float, int]:
    """找到轨迹最接近 GEO 球面的点。"""
    positions = trajectory_states[:, :3]
    dists = np.linalg.norm(positions - earth_center, axis=1)
    sphere_dists = np.abs(dists - r_geo)
    idx = int(np.argmin(sphere_dists))
    return float(sphere_dists[idx]), idx


def compute_geo_dv2(trajectory_state: npt.NDArray[np.floating]) -> float:
    """计算 GEO 插入 delta-v。"""
    v_geo = geo_circular_velocity_rotating(trajectory_state[:3])
    return float(np.linalg.norm(trajectory_state[3:] - v_geo))


def compute_departure_velocity(
    state: npt.NDArray[np.floating], alpha: float
) -> npt.NDArray[np.floating]:
    """切向速度缩放。"""
    pos = state[:3]
    vel = state[3:]
    r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
    if r_xy < 1e-10:
        return vel.copy()
    tangential = np.array([-pos[1], pos[0], 0.0]) / r_xy
    radial = pos / np.linalg.norm(pos)
    v_rad = np.dot(vel, radial)
    v_tan = np.dot(vel, tangential)
    return v_rad * radial + alpha * v_tan * tangential


def check_collision(
    states: npt.NDArray[np.floating],
    mu: float,
    earth_radius: float,
    moon_radius: float,
) -> tuple[bool, str | None, int]:
    """碰撞检测。"""
    if earth_radius <= 0 or moon_radius <= 0:
        raise ValueError("Radii must be positive")
    positions = states[:, :3]
    earth_center = np.array([-mu, 0.0, 0.0])
    moon_center = np.array([1.0 - mu, 0.0, 0.0])
    d_earth = np.linalg.norm(positions - earth_center, axis=1)
    d_moon = np.linalg.norm(positions - moon_center, axis=1)
    e_col = np.where(d_earth < earth_radius)[0]
    m_col = np.where(d_moon < moon_radius)[0]
    if len(e_col) > 0:
        return True, "earth", int(e_col[0])
    if len(m_col) > 0:
        return True, "moon", int(m_col[0])
    return False, None, -1
