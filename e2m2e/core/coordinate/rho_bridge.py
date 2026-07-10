"""rho 无量纲坐标 ↔ ECI（J2000, km）坐标桥接。

将 qiao ``rho_to_eci`` / ``eci_to_rho`` 的功能用 e2m2e 的 ``EphemerisSystem``
+ ``SynodicAxes`` 重写。rho 坐标系以选定平动点为原点，使用 CR3BP 归一化单位，
轴向与瞬时 EMR 会合系对齐。

数学关系::

    r_ECI = C @ rho_km + r_LP
    v_ECI = C @ rhodot_km + Cdot @ rho_km + C @ v_LP

其中 C 是 EMR→J2000 旋转矩阵（与 qiao ``Calc_MoonParam`` 约定一致），
r_LP/v_LP 是平动点在 J2000 中的位置/速度。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..cr3bp_system import LibrationPoint
from .synodic_axes import SynodicAxes

if TYPE_CHECKING:
    from .ephemeris_system import EphemerisSystem


@runtime_checkable
class RhoContext(Protocol):
    """rho↔ECI 桥接所需的最小上下文契约。

    core 层不应认识 algorithms 的类型；本 Protocol 描述 rho_bridge 实际用到的
    几个属性（归一化参数、平动点选择），让 :class:`NormalFormContext` 等上层
    类型按结构匹配，消除 core → algorithms 的反向依赖。
    """

    LU: float
    TU: float
    jd0: float
    gamma: float
    libration_point: LibrationPoint


def compute_emr_rotation(
    et: float, system: EphemerisSystem
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """从 SPICE 构造 EMR 会合系旋转矩阵 C(t) 及导数 Cdot(t)。

    约定与 qiao ``Calc_MoonParam`` 一致：C 满足 ``r_J2000 = C @ r_EMR``
    （EMR → J2000），其列向量在 J2000 中给出 EMR 基向量。
    ``SynodicAxes.rotation_matrix(et)`` 返回相同的矩阵。

    Args:
        et: 历书时（秒）。
        system: 星历系统，提供 SPICE 访问。

    Returns:
        ``(C, Cdot)``，各为 ``(3, 3)`` 数组。C 从 EMR→J2000，Cdot 为其时间导数。
    """
    syn_axes = SynodicAxes(system.spice)
    return syn_axes.rotation_and_rate(et)


def _jd_to_et(jd: float, system: EphemerisSystem) -> float:
    """儒略日 → SPICE 历书时（秒）。

    输入 jd 为 TDB 儒略日（与 qiao JD0 约定一致），使用 JDTDB 格式
    避免 UTC/TDB 的 ~64 秒偏差。
    """
    return system.spice.utc_to_et(f"{jd:.20f} JDTDB")


def _compute_lp_state_j2000(
    et: float,
    context: RhoContext,
    system: EphemerisSystem,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """计算平动点在 J2000 中的位置和速度。

    对共线点 (L1/L2/L3)：``r_LP = f_gamma * R_EM``，``v_LP = f_gamma * V_EM``。
    对三角点 (L4/L5)：通过旋转矩阵变换。

    Args:
        et: 历书时（秒）。
        context: 标准形上下文，提供平动点选择与归一化参数。
        system: 星历系统。

    Returns:
        ``(r_LP, v_LP)`` 各为 ``(3,)`` 数组，J2000 下 km / km/s。
    """
    moon_state = system.get_body_state("MOON", et)
    R_EM = moon_state[:3]
    V_EM = moon_state[3:]
    point = context.libration_point

    if point is LibrationPoint.L1:
        f = 1.0 - context.gamma
        return f * R_EM, f * V_EM
    if point is LibrationPoint.L2:
        f = 1.0 + context.gamma
        return f * R_EM, f * V_EM
    if point is LibrationPoint.L3:
        f = -context.gamma
        return f * R_EM, f * V_EM

    # L4 / L5：需要 C 矩阵做旋转（C 从 EMR→J2000）
    C, _Cdot = compute_emr_rotation(et, system)
    ang = -np.pi / 3.0 if point is LibrationPoint.L4 else np.pi / 3.0
    R_mat = np.array(
        [
            [np.cos(ang), np.sin(ang), 0.0],
            [-np.sin(ang), np.cos(ang), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    M = C @ R_mat @ C.T
    return M @ R_EM, M @ V_EM


def rho_to_eci(
    rho_nd: npt.ArrayLike,
    rhodot_nd: npt.ArrayLike,
    t_nd: float,
    context: RhoContext,
    system: EphemerisSystem,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """rho 无量纲坐标 → ECI（J2000, km, km/s）。

    Args:
        rho_nd: 无量纲位置 ``(3,)``，以平动点为原点。
        rhodot_nd: 无量纲速度 ``(3,)``。
        t_nd: 无量纲时间（TU）。
        context: 标准形上下文，提供 LU、TU、平动点选择等。
        system: 星历系统。

    Returns:
        ``(r_eci, v_eci)`` 各为 ``(3,)`` 数组，km 和 km/s。
    """
    rho_nd = np.asarray(rho_nd, dtype=float)
    rhodot_nd = np.asarray(rhodot_nd, dtype=float)

    LU = context.LU
    TU = context.TU

    jd = context.jd0 + t_nd * TU / 86400.0
    et = _jd_to_et(jd, system)

    C, Cdot = compute_emr_rotation(et, system)
    r_LP, v_LP = _compute_lp_state_j2000(et, context, system)

    rho_km = rho_nd * LU
    rhodot_km = rhodot_nd * LU / TU

    # EMR → J2000（C 从 EMR→J2000）
    r_eci = C @ rho_km + r_LP
    v_eci = C @ rhodot_km + Cdot @ rho_km + C @ v_LP

    return r_eci, v_eci


def eci_to_rho(
    r_eci: npt.ArrayLike,
    v_eci: npt.ArrayLike,
    t_nd: float,
    context: RhoContext,
    system: EphemerisSystem,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """ECI（J2000, km, km/s）→ rho 无量纲坐标。

    Args:
        r_eci: J2000 位置 ``(3,)``，km。
        v_eci: J2000 速度 ``(3,)``，km/s。
        t_nd: 无量纲时间（TU）。
        context: 标准形上下文。
        system: 星历系统。

    Returns:
        ``(rho_nd, rhodot_nd)`` 各为 ``(3,)`` 无量纲数组。
    """
    r_eci = np.asarray(r_eci, dtype=float)
    v_eci = np.asarray(v_eci, dtype=float)

    LU = context.LU
    TU = context.TU

    jd = context.jd0 + t_nd * TU / 86400.0
    et = _jd_to_et(jd, system)

    C, Cdot = compute_emr_rotation(et, system)
    r_LP, v_LP = _compute_lp_state_j2000(et, context, system)

    # J2000 → EMR（C.T 从 J2000→EMR）
    # rho_km = C.T @ (r_eci - r_LP), rhodot_km = C.T @ v_eci - C.T @ Cdot @ rho_km - v_LP
    rho_km = C.T @ (r_eci - r_LP)
    rho_nd = rho_km / LU

    rhodot_km = C.T @ v_eci - C.T @ Cdot @ rho_km - v_LP
    rhodot_nd = rhodot_km * TU / LU

    return rho_nd, rhodot_nd
