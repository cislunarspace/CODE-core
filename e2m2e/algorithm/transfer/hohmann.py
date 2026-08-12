"""HMN 霍曼直接转移：TLI 出发状态构造 + 霍曼转移物理量计算。

算法来源：
- 开普勒根数→笛卡尔：Curtis (2008) Algorithm 4.2
- 霍曼转移 Δv：Curtis §6.1 / Vallado
- 航迹角语义：Curtis Eq. 2.132
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e2m2e.data.constants.bodies import EARTH
from e2m2e.data.constants.datums import Datum

from .lambert import solve_lambert_batch

MU_EARTH: float = Datum.WGS84.earth_gm  # type: ignore[assignment]
R_EARTH: float = EARTH.gravity_ref_radius_km  # type: ignore[assignment]


@dataclass(frozen=True)
class TliParams:
    """Trans-Lunar Injection 出发参数。

    语义参照 Curtis (2008) Algorithm 4.2 + Lu et al. (2021) Eqs. 1-10。

    Attributes:
        parking_alt_km: 停泊轨道高度 (km)，圆轨道假设 (γ=0)
        inclination_deg: 停泊轨道倾角 (deg)
        flight_path_angle_deg: 航迹角 (deg)，霍曼转移出发条件为 0
        raan_deg: 升交点赤经 (deg)
        arg_perigee_deg: 近地点幅角 (deg)
        epoch: 出发历元（UTC 字符串或 JD_TDB 浮点数）
    """

    parking_alt_km: float
    inclination_deg: float
    flight_path_angle_deg: float = 0.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    epoch: float | str = 0.0


def _rotation_matrix(i_deg: float, omega_deg: float, raan_deg: float) -> NDArray[np.float64]:
    """Perifocal → ECI 旋转矩阵 R = R₃(-Ω)·R₁(-i)·R₃(-ω)。

    Curtis (2008) Algorithm 4.2, Step 3-5.
    """
    i = math.radians(i_deg)
    omega = math.radians(omega_deg)
    raan = math.radians(raan_deg)

    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(omega), math.sin(omega)
    co, so = math.cos(raan), math.sin(raan)

    # R₃(-ω) · R₁(-i) · R₃(-Ω)  — 先绕 z 轴转 -ω，再绕 x 轴转 -i，再绕 z 轴转 -Ω
    # 等价于 R = [col1 | col2 | col3] 其中
    # col1 = [cosΩcosω - sinΩsinωcosi,  sinΩcosω + cosΩsinωcosi,  sinωsini]
    # col2 = [-cosΩsinω - sinΩcosωcosi, -sinΩsinω + cosΩcosωcosi, cosωsini]
    # col3 = [sinΩsini,                  -cosΩsini,                 cosi    ]
    r = np.array(
        [
            [
                co * cw - so * sw * ci,
                -(co * sw + so * cw * ci),
                so * si,
            ],
            [
                so * cw + co * sw * ci,
                -(so * sw - co * cw * ci),
                -co * si,
            ],
            [
                sw * si,
                cw * si,
                ci,
            ],
        ],
        dtype=np.float64,
    )
    return r


def keplerian_to_cartesian(
    a_km: float,
    e: float,
    i_deg: float,
    omega_deg: float,
    raan_deg: float,
    nu_deg: float,
    mu: float = MU_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Curtis (2008) Algorithm 4.2: 开普勒根数 → ECI 笛卡尔状态。

    Args:
        a_km: 半长轴 (km)
        e: 偏心率
        i_deg: 倾角 (deg)
        omega_deg: 近地点幅角 (deg)
        raan_deg: 升交点赤经 (deg)
        nu_deg: 真近点角 (deg)
        mu: 引力参数 (km³/s²)

    Returns:
        (r_eci, v_eci) in (km, km/s)，shape 均为 (3,)
    """
    nu = math.radians(nu_deg)
    p = a_km * (1.0 - e * e)  # 半通径

    # Perifocal frame 位置与速度
    r_mag = p / (1.0 + e * math.cos(nu))
    r_pf = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])
    v_pf = np.array(
        [
            -math.sin(nu),
            e + math.cos(nu),
            0.0,
        ]
    ) * math.sqrt(mu / p)

    # 旋转到 ECI
    rot = _rotation_matrix(i_deg, omega_deg, raan_deg)
    r_eci = rot @ r_pf
    v_eci = rot @ v_pf
    return r_eci, v_eci


def construct_departure_state(
    params: TliParams,
    mu_earth: float = MU_EARTH,
    r_earth: float = R_EARTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TLI 参数 → ECI 出发状态向量。

    简化路径（γ=0，圆停泊轨道，霍曼转移标准出发条件）：
    1. r_park = r_earth + parking_alt_km
    2. v_park = √(mu_earth / r_park)
    3. r_pf = [r_park, 0, 0]ᵀ;  v_pf = [0, v_park, 0]ᵀ
    4. R = R₃(-Ω)·R₁(-i)·R₃(-ω)
    5. r_eci = R·r_pf;  v_eci = R·v_pf

    非零航迹角路径（γ≠0）：尚未实现，抛出 NotImplementedError。

    Args:
        params: TLI 出发参数
        mu_earth: 地球引力参数 (km³/s²)
        r_earth: 地球赤道半径 (km)

    Returns:
        (r0, v0) in (km, km/s)，shape 均为 (3,)
    """
    r_park = r_earth + params.parking_alt_km

    if abs(params.flight_path_angle_deg) < 1e-10:
        # γ=0：圆轨道，简化路径
        v_park = math.sqrt(mu_earth / r_park)
        r_pf = np.array([r_park, 0.0, 0.0])
        v_pf = np.array([0.0, v_park, 0.0])
        rot = _rotation_matrix(params.inclination_deg, params.arg_perigee_deg, params.raan_deg)
        return rot @ r_pf, rot @ v_pf
    raise NotImplementedError("非零航迹角路径尚未验证")


def hohmann_delta_v(r1: float, r2: float, mu: float = MU_EARTH) -> tuple[float, float]:
    """经典霍曼转移 Δv（Curtis §6.1 / Vallado）。

    Args:
        r1: 出发轨道半径 (km)
        r2: 到达轨道半径 (km)
        mu: 引力参数 (km³/s²)

    Returns:
        (dv1, dv2) in km/s
    """
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = math.sqrt(mu / r2) * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2)))
    return dv1, dv2


def hohmann_tof(r1: float, r2: float, mu: float = MU_EARTH) -> float:
    """霍曼转移飞行时间（半椭圆周期）。

    Args:
        r1: 出发轨道半径 (km)
        r2: 到达轨道半径 (km)
        mu: 引力参数 (km³/s²)

    Returns:
        飞行时间 (秒)
    """
    a_t = (r1 + r2) / 2.0
    return math.pi * math.sqrt(a_t**3 / mu)


def scan_lambert_delta_v(
    r0: NDArray[np.float64],
    v0_park: NDArray[np.float64],
    r_target: NDArray[np.float64],
    v_target: NDArray[np.float64],
    tof_grid: NDArray[np.float64],
    mu: float = MU_EARTH,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """用 Lambert 批量扫描找最优 tof。

    对每个 tof 调用 solve_lambert_batch，计算总
    ``Δv = |v0_lambert - v0_park| + |vf_lambert - v_target|``，
    返回 (最优 tof, 最优 v0_lambert, 最优 vf_lambert)。

    Args:
        r0: 出发位置 (3,) km
        v0_park: 停泊轨道速度 (3,) km/s
        r_target: 到达位置 (3,) km
        v_target: 目标轨道速度 (3,) km/s
        tof_grid: tof 网格 (M,) 秒
        mu: 引力参数 (km³/s²)

    Returns:
        (最优 tof, 最优 v0_lambert, 最优 vf_lambert)

    Raises:
        ValueError: 所有 tof 均无解。
    """
    result = solve_lambert_batch(
        r0.reshape(1, 3), r_target.reshape(1, 3), tof_grid, mu
    )  # (1, M, 2, 3)

    total_dv = np.full(tof_grid.shape[0], np.inf)
    v0_lam = result[0, :, 0, :]  # (M, 3)
    vf_lam = result[0, :, 1, :]  # (M, 3)
    nan_mask = np.any(np.isnan(v0_lam) | np.isnan(vf_lam), axis=1)
    total_dv[~nan_mask] = np.linalg.norm(v0_lam[~nan_mask] - v0_park, axis=1) + np.linalg.norm(
        vf_lam[~nan_mask] - v_target, axis=1
    )

    best_idx = int(np.argmin(total_dv))
    if np.isinf(total_dv[best_idx]):
        raise ValueError("scan_lambert_delta_v: 所有 tof 均无解")

    return (
        float(tof_grid[best_idx]),
        result[0, best_idx, 0, :].copy(),
        result[0, best_idx, 1, :].copy(),
    )


def ephemeris_shoot_transfer(
    dynamics: Any,
    t0: float,
    r0: NDArray[np.float64],
    v0: NDArray[np.float64],
    tof: float,
    n_patches: int = 5,
    max_iter: int = 30,
    tolerance: float = 1e-6,
) -> Any:
    """用 MultipleShooting 在给定动力学模型下修正 Lambert 初猜。

    步骤：

    1. 沿 Lambert 初猜弧段均匀采样 ``n_patches`` 个 patch point 时刻。
    2. 用动力学模型 ``dynamics.propagate`` 从 (r0, v0) 积分整条弧段，
       在各 patch point 时刻插值取状态作为初猜。
    3. 调用 ``MultipleShooting.correct()`` 收敛。
    4. 返回 ``e2m2e.algorithm.solver.multiple_shooting.MultipleShootingResult``。

    Args:
        dynamics: 动力学对象，需提供 ``propagate(state, t_span, with_stm=True)``
            和 ``equations_of_motion(t, state)`` 接口。
        t0: 出发时刻（秒，动力学模型的时间基准）。
        r0: 出发位置 (3,) km。
        v0: 出发速度 (3,) km/s，Lambert 解。
        tof: 飞行时间 (秒)。
        n_patches: patch point 数量，默认 5。
        max_iter: 多重打靶最大迭代次数，默认 30。
        tolerance: 收敛容差，默认 1e-6。

    Returns:
        :class:`~e2m2e.algorithm.solver.multiple_shooting.MultipleShootingResult`: 打靶修正结果。

    Raises:
        ValueError: n_patches < 2。
    """
    # 延迟导入，避免循环依赖
    from ..solver.multiple_shooting import MultipleShooting

    if n_patches < 2:
        raise ValueError("n_patches must be >= 2")

    # 1. 均匀采样 patch point 时刻
    t_end = t0 + tof
    t_patch = np.linspace(t0, t_end, n_patches)

    # 2. 用动力学模型从 (r0, v0) 积分整条弧段，获取初猜状态
    initial_state = np.concatenate([r0, v0])
    result = dynamics.propagate(initial_state, (t0, t_end), with_stm=True)
    states_traj = result["states"]  # (M, 6)
    time_traj = result["time"]  # (M,)

    # 在各 patch point 时刻线性插值取状态
    state_patch = np.empty((n_patches, 6))
    for j in range(6):
        state_patch[:, j] = np.interp(t_patch, time_traj, states_traj[:, j])

    # 3. 调用 MultipleShooting 收敛
    ms = MultipleShooting(dynamics=dynamics)
    return ms.correct(
        t_patch=t_patch,
        state_patch=state_patch,
        var_time=False,
        max_iter=max_iter,
        tolerance=tolerance,
    )
