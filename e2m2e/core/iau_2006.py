"""IAU 2006 简化地球定向参数计算。

提供岁差矩阵与章动矩阵的纯 Python 实现，不依赖外部 EOP 文件。
本模块使用 IAU 2006 简化理论，适合教学、快速验证以及无 SPICE 内核
场景下的近似地固系计算。

参考
====

- IAU 2006 Precession-Nutation Model (Capitaine et al. 2003)
- Explanatory Supplement to the Astronomical Almanac, 3rd edition
- NASA NAIF SPICE Toolkit 文档中的 ``pxform`` 与 ``iau2000`` 说明
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# 儒略世纪长度（天）
_DAYS_PER_JULIAN_CENTURY = 36525.0


def _seconds_to_julian_centuries(et: float) -> float:
    """将 SPICE 历书时（秒，TDB）转换为 J2000.0 起算的儒略世纪数。

    Args:
        et: SPICE 历书时（秒）。

    Returns:
        从 J2000.0 起算的儒略世纪数 T。
    """
    # J2000.0 对应的 SPICE 历书时为 0 秒
    # 1 儒略世纪 = 36525 天 = 36525 * 86400 秒
    return et / (_DAYS_PER_JULIAN_CENTURY * 86400.0)


def _arcseconds_to_radians(arcsec: float) -> float:
    """将角秒转换为弧度。"""
    return arcsec * np.pi / (180.0 * 3600.0)


def precession_angles(t: float) -> tuple[float, float, float]:
    """计算 IAU 2006 岁差角。

    Args:
        t: 从 J2000.0 起算的儒略世纪数。

    Returns:
        三元组 ``(zeta_A, theta_A, z_A)``，单位为弧度。
        满足岁差矩阵 ``P = R3(-z_A) @ R2(theta_A) @ R3(-zeta_A)``。
    """
    zeta_arcsec = 2306.083227 * t + 0.2988500 * t**2 + 0.01803728 * t**3
    theta_arcsec = 2004.191903 * t - 0.4294934 * t**2 - 0.04180064 * t**3
    z_arcsec = 2306.077181 * t + 1.0927348 * t**2 + 0.01826837 * t**3
    return (
        _arcseconds_to_radians(zeta_arcsec),
        _arcseconds_to_radians(theta_arcsec),
        _arcseconds_to_radians(z_arcsec),
    )


def precession_matrix(t: float) -> npt.NDArray[np.floating]:
    """计算 IAU 2006 岁差矩阵 P(t)。

    Args:
        t: 从 J2000.0 起算的儒略世纪数。

    Returns:
        3x3 岁差矩阵，满足 ``r_J2000 = P @ r_mean_equator``。
    """
    zeta, theta, z = precession_angles(t)
    return _rotation3(-z) @ _rotation2(theta) @ _rotation3(-zeta)


def mean_obliquity(t: float) -> float:
    """计算 J2000.0 平黄赤交角 epsilon_0。

    Args:
        t: 从 J2000.0 起算的儒略世纪数。

    Returns:
        平黄赤交角，单位为弧度。
    """
    arcsec = 84381.406 - 46.836769 * t - 0.0001831 * t**2 + 0.00200340 * t**3
    return _arcseconds_to_radians(arcsec)


def nutation_angles(t: float) -> tuple[float, float, float]:
    """计算 IAU 2000A 简化章动角。

    本实现仅包含最大项（黄经章动 psi 与交角章动 epsilon 的主项），
    精度约为 0.1 角秒量级。对于更高精度需求，应使用 SPICE 内核。

    Args:
        t: 从 J2000.0 起算的儒略世纪数。

    Returns:
        三元组 ``(dpsi, deps, eps0)``，分别为黄经章动、交角章动和平黄赤
        交角，单位均为弧度。
    """
    # 黄经章动主项系数（角秒）
    # 基于 IAU 2000A 最大项近似
    dpsi_arcsec = -17.206424 * np.sin(_mean_moon_node(t)) + 0.003386 * np.sin(
        2 * _mean_moon_node(t)
    )
    deps_arcsec = 9.205233 * np.cos(_mean_moon_node(t))

    eps0 = mean_obliquity(t)
    return (_arcseconds_to_radians(dpsi_arcsec), _arcseconds_to_radians(deps_arcsec), eps0)


def nutation_matrix(t: float) -> npt.NDArray[np.floating]:
    """计算章动矩阵 N(t)。

    Args:
        t: 从 J2000.0 起算的儒略世纪数。

    Returns:
        3x3 章动矩阵，满足 ``r_true_equator = N @ r_J2000``。
    """
    dpsi, deps, eps0 = nutation_angles(t)
    return _rotation1(-(eps0 + deps)) @ _rotation3(-dpsi) @ _rotation1(eps0)


def iau2000eq_matrix(et: float) -> npt.NDArray[np.floating]:
    """计算从 IAU2000Eq 平赤道/平春分点到 ICRF/J2000 的旋转矩阵。

    该矩阵仅包含岁差效应：``R = P``。平赤道/平春分点系不包含章动。

    Args:
        et: SPICE 历书时（秒）。

    Returns:
        3x3 旋转矩阵，满足 ``r_ICRF = R @ r_IAU2000Eq``。
    """
    t = _seconds_to_julian_centuries(et)
    return precession_matrix(t)


def iau2000eq_true_matrix(et: float) -> npt.NDArray[np.floating]:
    """计算从真赤道/真春分点（含章动）到 ICRF/J2000 的旋转矩阵。

    该矩阵包含岁差与章动效应：``R = N @ P``。

    Args:
        et: SPICE 历书时（秒）。

    Returns:
        3x3 旋转矩阵，满足 ``r_ICRF = R @ r_true_equator``。
    """
    t = _seconds_to_julian_centuries(et)
    p = precession_matrix(t)
    n = nutation_matrix(t)
    return n @ p


def _mean_moon_node(t: float) -> float:
    """计算月球平均升交点黄经（弧度）。"""
    arcsec = 450160.398036 - 6962890.5431 * t + 7.4722 * t**2 + 0.007702 * t**3
    return _arcseconds_to_radians(arcsec)


def _rotation1(angle: float) -> npt.NDArray[np.floating]:
    """绕 x 轴的旋转矩阵。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation2(angle: float) -> npt.NDArray[np.floating]:
    """绕 y 轴的旋转矩阵。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation3(angle: float) -> npt.NDArray[np.floating]:
    """绕 z 轴的旋转矩阵。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
