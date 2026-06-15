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

# ----------------------------------------------------------------------------
# 时间与角度常量
# ----------------------------------------------------------------------------

_JD_J2000 = 2451545.0
_DAYS_PER_JULIAN_CENTURY = 36525.0
_DAYS_PER_YEAR = 365.25
_RAD_PER_DEG = np.pi / 180.0

# ----------------------------------------------------------------------------
# Table 6.3(IERS TN32 p.64/66,迁移 GMAT HarmonicGravity.cpp 静态数组)
# 每行:[l, l', F, D, O, ...系数]
#   l,l' : Doodson 幅角阶数(用于幅角组合)
#   F,D,O: 月亮升交点经度 F、月亮平近点角... 实际每行前 5 列是 5 个 Delaunay
#          幅角的整数乘子(N1..N5 对应 F[0..4])
#   ip,op: C/S(或 cos/sin)的同相、正交系数(单位 1e-12)
# Table63c 只有 amp(1 列),(2,2) 公式不同
# ----------------------------------------------------------------------------

# Table63a:48 行,C21/S21。列 [N1,N2,N3,N4,N5, ip, op]
_TABLE_63A: npt.NDArray[np.floating] = np.array(
    [
        [2, 0, 2, 0, 2, -0.1, 0],
        [0, 0, 2, 2, 2, -0.1, 0],
        [1, 0, 2, 0, 1, -0.1, 0],
        [1, 0, 2, 0, 2, -0.7, 0.1],
        [-1, 0, 2, 2, 2, -0.1, 0],
        [0, 0, 2, 0, 1, -1.3, 0.1],
        [0, 0, 2, 0, 2, -6.8, 0.6],
        [0, 0, 0, 2, 0, 0.1, 0],
        [1, 0, 2, -2, 2, 0.1, 0],
        [-1, 0, 2, 0, 1, 0.1, 0],
        [-1, 0, 2, 0, 2, 0.4, 0],
        [1, 0, 0, 0, 0, 1.3, -0.1],
        [1, 0, 0, 0, 1, 0.3, 0],
        [-1, 0, 0, 2, 0, 0.3, 0],
        [-1, 0, 0, 2, 1, 0.1, 0],
        [0, 1, 2, -2, 2, -1.9, 0.1],
        [0, 0, 2, -2, 1, 0.5, 0],
        [0, 0, 2, -2, 2, -43.4, 2.9],
        [0, -1, 2, -2, 2, 0.6, 0],
        [0, 1, 0, 0, 0, 1.6, -0.1],
        [-2, 0, 2, 0, 1, 0.1, 0],
        [0, 0, 0, 0, -2, 0.1, 0],
        [0, 0, 0, 0, -1, -8.8, 0.5],
        [0, 0, 0, 0, 0, 470.9, -30.2],
        [0, 0, 0, 0, 1, 68.1, -4.6],
        [0, 0, 0, 0, 2, -1.6, 0.1],
        [-1, 0, 0, 1, 0, 0.1, 0],
        [0, -1, 0, 0, -1, -0.1, 0],
        [0, -1, 0, 0, 0, -20.6, -0.3],
        [0, 1, -2, 2, -2, 0.3, 0],
        [0, -1, 0, 0, 1, -0.3, 0],
        [-2, 0, 0, 2, 0, -0.2, 0],
        [-2, 0, 0, 2, 1, -0.1, 0],
        [0, 0, -2, 2, -2, -5.0, 0.3],
        [0, 0, -2, 2, -1, 0.2, 0],
        [0, -1, -2, 2, -2, -0.2, 0],
        [1, 0, 0, -2, 0, -0.5, 0],
        [1, 0, 0, -2, 1, -0.1, 0],
        [-1, 0, 0, 0, -1, 0.1, 0],
        [-1, 0, 0, 0, 0, -2.1, 0.1],
        [-1, 0, 0, 0, 1, -0.4, 0],
        [0, 0, 0, -2, 0, -0.2, 0],
        [-2, 0, 0, 0, 0, -0.1, 0],
        [0, 0, -2, 0, -2, -0.6, 0],
        [0, 0, -2, 0, -1, -0.4, 0],
        [0, 0, -2, 0, 0, -0.1, 0],
        [-1, 0, -2, 0, -2, -0.1, 0],
        [-1, 0, -2, 0, -1, -0.1, 0],
    ],
    dtype=float,
)

# Table63b:21 行,C20。列 [N1,N2,N3,N4,N5, ip, op]
_TABLE_63B: npt.NDArray[np.floating] = np.array(
    [
        [0, 0, 0, 0, 1, 16.6, -6.7],
        [0, 0, 0, 0, 2, -0.1, 0.1],
        [0, -1, 0, 0, 0, -1.2, 0.8],
        [0, 0, -2, 2, -2, -5.5, 4.3],
        [0, 0, -2, 2, -1, 0.1, -0.1],
        [0, -1, -2, 2, -2, -0.3, 0.2],
        [1, 0, 0, -2, 0, -0.3, 0.7],
        [-1, 0, 0, 0, -1, 0.1, -0.2],
        [-1, 0, 0, 0, 0, -1.2, 3.7],
        [-1, 0, 0, 0, 1, 0.1, -0.2],
        [1, 0, -2, 0, -2, 0.1, -0.2],
        [0, 0, 0, -2, 0, 0.0, 0.6],
        [-2, 0, 0, 0, 0, 0.0, 0.3],
        [0, 0, -2, 0, -2, 0.6, 6.3],
        [0, 0, -2, 0, -1, 0.2, 2.6],
        [0, 0, -2, 0, 0, 0.0, 0.2],
        [1, 0, -2, -2, -2, 0.1, 0.2],
        [-1, 0, -2, 0, -2, 0.4, 1.1],
        [-1, 0, -2, 0, -1, 0.2, 0.5],
        [0, 0, -2, -2, -2, 0.1, 0.2],
        [-2, 0, -2, 0, -2, 0.1, 0.1],
    ],
    dtype=float,
)

# Table63c:2 行,C22/S22。列 [N1,N2,N3,N4,N5, amp]
_TABLE_63C: npt.NDArray[np.floating] = np.array(
    [
        [1, 0, 2, 0, 2, -0.3],
        [0, 0, 2, 0, 2, -1.2],
    ],
    dtype=float,
)


def _delaunay_args_and_gmst(jd: float) -> tuple[npt.NDArray[np.floating], float]:
    """计算 5 个 Delaunay 幅角(度)与 GMST(度)。

    迁移 GMAT IncrementEarthTide 的幅角段(IERS TN32 p.48/60)。GMAT 注释
    "ignore difference between TDB and TDT",本实现用 et→JD(TDB)近似,
    框架层面可接受(潮汐是日级振荡)。

    Args:
        jd: 儒略日(TDB 近似)。

    Returns:
        (F, GMST):F 为长度 5 的数组(度),GMST 为标量(度)。
    """
    t = (jd - _JD_J2000) / _DAYS_PER_JULIAN_CENTURY
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    # 5 个 Delaunay 幅角(度)
    F = np.zeros(5, dtype=float)
    F[0] = (134.96340251e3 + 1717915923.2178 * t + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4) / 3600.0
    F[1] = (357.52910918e3 + 129596581.0481 * t - 0.5532 * t2 + 0.000136 * t3 - 0.00001149 * t4) / 3600.0
    F[2] = (93.27209062e3 + 1739527262.8478 * t - 12.7512 * t2 - 0.001037 * t3 + 0.00000417 * t4) / 3600.0
    F[3] = (297.85019547e3 + 1602961601.2090 * t - 6.3706 * t2 + 0.006593 * t3 - 0.00003169 * t4) / 3600.0
    F[4] = (125.04455501e3 - 6962890.5431 * t + 7.4722 * t2 + 0.007702 * t3 - 0.00005939 * t4) / 3600.0

    # GMST(IERS p.60):先算秒,再 /240 转度
    gmst_sec = 67310.54841 + 3164400184.812866 * t + 0.093104 * t2 - 6.2e-06 * t3
    gmst_deg = gmst_sec / 240.0

    return F, gmst_deg


def solid_tide_step2(et: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """固体潮 Step 2(频率相关,迁移 GMAT IncrementEarthTide 的 Delaunay 幅角段)。

    只影响 (2,0)/(2,1)/(2,2)。用 5 个 Delaunay 幅角 + GMST + Table6.3a/b/c。
    量级 ~1e-10(GMAT ``freq_dep * 1e-12`` 缩放)。

    Args:
        et: SPICE et 秒(past J2000)。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组;仅 (2,0)/(2,1)/(2,2) 非零。
    """
    jd = _JD_J2000 + et / 86400.0
    F, gmst = _delaunay_args_and_gmst(jd)

    deltaC = np.zeros((5, 5), dtype=float)
    deltaS = np.zeros((5, 5), dtype=float)

    # (2,0) 频率相关:IERS eqn 5a
    freq_c20 = 0.0
    for row in _TABLE_63B:
        theta = -np.dot(row[:5], F) * _RAD_PER_DEG  # 弧度
        freq_c20 += row[5] * np.cos(theta) - row[6] * np.sin(theta)
    deltaC[2, 0] += freq_c20 * 1e-12

    # (2,1) 频率相关:IERS eqn 5b,m=1
    freq_c21 = 0.0
    freq_s21 = 0.0
    m = 1
    for row in _TABLE_63A:
        theta = (m * (gmst + 180.0) - np.dot(row[:5], F)) * _RAD_PER_DEG
        freq_c21 += row[5] * np.sin(theta) + row[6] * np.cos(theta)
        freq_s21 += row[5] * np.cos(theta) - row[6] * np.sin(theta)
    deltaC[2, 1] += freq_c21 * 1e-12
    deltaS[2, 1] += freq_s21 * 1e-12

    # (2,2) 频率相关:m=2,Table63c 只有 amp
    freq_c22 = 0.0
    freq_s22 = 0.0
    m = 2
    for row in _TABLE_63C:
        theta = (m * (gmst + 180.0) - np.dot(row[:5], F)) * _RAD_PER_DEG
        freq_c22 += row[5] * np.cos(theta)
        freq_s22 += -row[5] * np.sin(theta)
    deltaC[2, 2] += freq_c22 * 1e-12
    deltaS[2, 2] += freq_s22 * 1e-12

    return deltaC, deltaS


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


def pole_tide(
    et: float,
    xp: float,
    yp: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """极潮(固体极潮 IERS p.65 + Desai 海洋极潮 TN32 §6.3)。

    迁移 GMAT IncrementEarthTide 的极潮段。只影响 (2,1)。对齐 GMAT
    ``ETide::SolidAndPole`` 档(固体极潮 + 海洋极潮都做);``Solid`` 档不做极潮。

    公式:
        ym2000 = (JD - JD_J2000) / 365.25
        xp_bar = 0.054 + ym2000·0.00083  (IERS p.84 mean pole)
        yp_bar = 0.357 + ym2000·0.00395
        m1 = xp - xp_bar;m2 = -(yp - yp_bar)
        固体极潮:ΔC21 -= 1.333e-9·(m1+0.0115·m2);ΔS21 -= 1.333e-9·(m2-0.0115·m1)
        海洋极潮:ΔC21 -= 2.2344e-10·(m1-0.01737·m2);ΔS21 -= 1.7680e-10·(m2-0.03351·m1)

    Args:
        et: SPICE et 秒(past J2000)。
        xp: 极移 x 分量(arcsec,IERS EOP C04)。
        yp: 极移 y 分量(arcsec)。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组;仅 (2,1) 非零。
    """
    jd = _JD_J2000 + et / 86400.0
    ym2000 = (jd - _JD_J2000) / _DAYS_PER_YEAR
    xp_bar = 0.054 + ym2000 * 0.00083
    yp_bar = 0.357 + ym2000 * 0.00395

    m1 = xp - xp_bar
    m2 = -(yp - yp_bar)

    deltaC = np.zeros((5, 5), dtype=float)
    deltaS = np.zeros((5, 5), dtype=float)

    # 固体极潮(IERS p.65)
    deltaC[2, 1] -= 1.333e-09 * (m1 + 0.0115 * m2)
    deltaS[2, 1] -= 1.333e-09 * (m2 - 0.0115 * m1)

    # 海洋极潮(Desai,TN32 §6.3)
    deltaC[2, 1] -= 2.2344e-10 * (m1 - 0.01737 * m2)
    deltaS[2, 1] -= 1.7680e-10 * (m2 - 0.03351 * m1)

    return deltaC, deltaS


def permanent_tide_correction(
    mu_sun: float,
    mu_moon: float,
    mu_earth: float,
    r_earth: float,
    a_sun: float,
    a_moon: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """永久潮汐修正(IERS TN32 Step 3,时间平均,AC3)。

    zero-tide 系数约定下,``GravityField`` 在叠加固体潮后减去此值——因为
    zero-tide 系数已含永久潮汐,运行时若再加完整固体潮(含永久分量)会重复。

    用 ``solid_tide_step1`` 在 Sun/Moon 半长轴距离 + 赤道(零纬度,时间平均
    近似)计算。GMAT 把永久潮汐处理放在系数加载 setup;e2m2e 用运行时减除,
    公式等价(GMAT 注释 "moved to model setup, correction to TideFree
    coefficients" 的本意)。

    Args:
        mu_sun, mu_moon: Sun/Moon GM,km³/s²。
        mu_earth: 地球 GM,km³/s²。
        r_earth: 地球参考半径,km。
        a_sun, a_moon: Sun/Moon 轨道半长轴,km(时间平均距离近似)。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组。
    """
    sun_pos = np.array([a_sun, 0.0, 0.0])
    moon_pos = np.array([a_moon, 0.0, 0.0])
    dC_sun, dS_sun = solid_tide_step1(sun_pos, mu_sun, mu_earth, r_earth)
    dC_moon, dS_moon = solid_tide_step1(moon_pos, mu_moon, mu_earth, r_earth)
    return dC_sun + dC_moon, dS_sun + dS_moon
