"""地球潮汐修正(迁移 GMAT HarmonicGravity)。

迁移 GMAT R2026a ``src/base/forcemodel/harmonic/HarmonicGravity.cpp`` 的潮汐
修正能力,基于 IERS Technical Note 32(Conventions 2003)。本模块只做纯计算,
返回 ΔC/ΔS 系数修正,由 ``GravityField`` 在球谐展开前叠加(AC1-AC4)。

单位约定:位置 km、GM km³/s²、参考半径 km,与 ``gravity_field.py`` 一致。
GMAT 源码用混单位(pos=meters、FieldRadius=km)凑量级,本实现用一致单位
重写,物理公式等价。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ----------------------------------------------------------------------------
# Love 数(GMAT LM_SetDefaultEarthTide 默认值,硬编码)
# KEarth[n][m]:n=2,3 阶位移 Love 数;n=0,1,4 为零
# KPlusEarth[m]:n=2 时的弹性 3 阶位移(LoveMax=4,GMAT 数组)
# ----------------------------------------------------------------------------

_K_EARTH: npt.NDArray[np.floating] = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=0
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=1
        [0.30190, 0.29830, 0.30102, 0.0, 0.0],  # n=2(K20,K21,K22)
        [0.093, 0.093, 0.093, 0.094, 0.0],  # n=3
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=4
    ],
    dtype=float,
)

_K_PLUS_EARTH: npt.NDArray[np.floating] = np.array(
    [-0.00087, -0.00079, -0.00057, 0.0, 0.0], dtype=float
)


def _legendre_23(s: float, c: float) -> npt.NDArray[np.floating]:
    """计算 n=2,3 的完全正规化 associated Legendre(与 GMAT PolarToLegendre 一致)。

    Args:
        s: sin(地心纬度)。
        c: cos(地心纬度)。

    Returns:
        5×5 数组,P[n][m],n,m ∈ {0..4};未计算的项为零。
    """
    P = np.zeros((5, 5), dtype=float)
    sqrt5 = np.sqrt(5.0)
    sqrt5over3 = np.sqrt(5.0 / 3.0)
    sqrt7 = np.sqrt(7.0)
    sqrt7over6 = np.sqrt(7.0 / 6.0)
    sqrt7over15 = np.sqrt(7.0 / 15.0)
    sqrt_point7 = np.sqrt(0.7)

    # n=2
    P[2, 0] = sqrt5 * (1.5 * s * s - 0.5)
    P[2, 1] = 3.0 * sqrt5over3 * c * s
    P[2, 2] = 1.5 * sqrt5over3 * c * c
    # n=3
    P[3, 0] = sqrt7 * (2.5 * s * s * s - 1.5 * s)
    P[3, 1] = sqrt7over6 * c * (7.5 * s * s - 1.5)
    P[3, 2] = 7.5 * sqrt7over15 * c * c * s
    P[3, 3] = 2.5 * sqrt_point7 * c * c * c
    return P


def solid_tide_step1(
    pos_perturber: npt.ArrayLike,
    mu_perturber: float,
    mu_earth: float,
    r_earth: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """固体潮 Step 1(频率无关,迁移 GMAT IncrementSolidTide)。

    对单个扰动天体(Sun 或 Moon)计算 ΔC/ΔS。调用方需对 Sun、Moon 各调一次
    并累加(GMAT IncrementEarthTide 内部调 IncrementSolidTide 两次)。

    公式(IERS TN32 eqn 1, p.59;eqn 4, p.60):
        ΔC[n][m] += K[n][m]/(2n+1) · (μ_p/μ_e) · (R_e/r)^(n+1) · P_nm(sinφ) · cos(mλ)
        ΔS[n][m] += K[n][m]/(2n+1) · (μ_p/μ_e) · (R_e/r)^(n+1) · P_nm(sinφ) · sin(mλ)
    n=2 时额外(弹性 Love 数,3 阶位移):
        ΔC[4][m] += KPlus[m]/5 · ... ; ΔS[4][m] += KPlus[m]/5 · ...

    Args:
        pos_perturber: 扰动天体在 ITRF 下的位置,形状 (3,),单位 km。
        mu_perturber: 扰动天体 GM,km³/s²。
        mu_earth: 地球 GM,km³/s²。
        r_earth: 地球参考半径,km。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组。
    """
    pos = np.asarray(pos_perturber, dtype=float)
    r = np.linalg.norm(pos)
    if r == 0.0:
        raise ValueError("perturber position must be non-zero")

    # 地心纬度 φ、经度 λ
    xy = np.hypot(pos[0], pos[1])
    lat = np.arctan2(pos[2], xy)
    lon = np.arctan2(pos[1], pos[0])
    s = np.sin(lat)
    c = np.cos(lat)

    P = _legendre_23(s, c)
    massratio = mu_perturber / mu_earth
    rho = r_earth / r  # 无量纲

    deltaC = np.zeros((5, 5), dtype=float)
    deltaS = np.zeros((5, 5), dtype=float)

    for n in (2, 3):
        rho_n = rho ** (n + 1)
        for m in range(0, n + 1):
            f = massratio * rho_n * P[n, m]
            cm = np.cos(m * lon)
            sm = np.sin(m * lon)
            kn = _K_EARTH[n, m] / (2 * n + 1)
            deltaC[n, m] += kn * f * cm
            deltaS[n, m] += kn * f * sm
            if n == 2:
                kplus = _K_PLUS_EARTH[m] / (2 * n + 1)  # (2n+1)=5
                deltaC[4, m] += kplus * f * cm
                deltaS[4, m] += kplus * f * sm

    return deltaC, deltaS
