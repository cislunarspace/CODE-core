"""分区解析尺度（spatiography scales）。

Rosengren et al. 2026《The Astrodynamics Primer on Cislunar and Translunar
Space》§5 的全部闭式边界尺度，物理单位（km、km³/s²、rad/s）。纯函数形态：
显式数值参数 + 可选 ``PrimerConstants`` 缺省，无传播、无迭代（照
``algorithm/dynamics/potential.py`` 范式）。属 ADR 0014 二档查询；
分层契约与复现陷阱见 ADR 0041。

角度约定（Battin 面，陷阱④）：``psi`` 一律从**反主天体方向**量起
（月球版 psi=0 为背地、地球版 psi=0 为背日），只有按此约定才能复现
论文数值（朝地 52009 km / 背地 64201 km）。论文正文行内定义（月卫线
与月地线夹角）与其数值矛盾，以数值为准。
"""

from __future__ import annotations

import math

from .constants import PRIMER_DEFAULTS, PrimerConstants

__all__ = [
    "activity_surface_moon",
    "battin_soi_earth",
    "battin_soi_moon",
    "characteristic_rate_j2",
    "characteristic_rate_lunar_exterior",
    "characteristic_rate_solar_exterior",
    "chebotarev_radius_earth",
    "chebotarev_radius_moon",
    "geo_radius_km",
    "hill_radius_earth",
    "hill_radius_moon",
    "laplace_radius_geolunar",
    "laplace_radius_selenocentric",
    "soi_laplace_earth",
    "soi_laplace_moon",
    "tidal_parity_radius",
    "tisserand_parameter",
]


def characteristic_rate_j2(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球扁率特征进动频率标度 omega^+（论文式 95），rad/s。

    .. math:: \\omega^\\oplus = \\frac{3}{2}\\sqrt{\\mu_\\oplus}\\, J_2 R_\\oplus^2\\, a^{-7/2}

    这是标度系数而非真实拱线率（真实率含卫星根数因子，论文式 93/94）；
    Laplace 半径只由特征频率定义（与卫星根数无关）。
    """
    c = constants
    return 1.5 * math.sqrt(c.earth_gm) * c.earth_j2 * c.earth_ref_radius_km**2 * a_km ** (-3.5)


def characteristic_rate_lunar_exterior(
    a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """月球外支四极长期特征频率 omega^☾（论文式 96），rad/s。

    .. math:: \\omega^\\mathbb{C} = \\frac{3}{4}\\frac{\\mu_\\mathbb{C}}{\\sqrt{\\mu_\\oplus}}
        \\frac{1-\\sin^2 I_\\mathbb{C}/2}{a_\\mathbb{C}^3(1-e_\\mathbb{C}^2)^{3/2}}\\, a^{3/2}

    倾角因子按原文取 ``1 - sin²I☾/2``（= 0.99596）。原文此处存在排版歧义
    （另一读法 ``1 - 3/2 sin²I☾`` = 0.98788），两者经五次方根阻尼后对
    r_L 的影响仅 +0.11%（48812.40 → 48866.98 km）；本实现取字面读法以
    逐位复现论文 Table 1 值，见 ADR 0041 陷阱注记。
    """
    c = constants
    inc_rad = math.radians(c.moon_inc_deg)
    factor = 1.0 - 0.5 * math.sin(inc_rad) ** 2
    return (
        0.75
        * (c.moon_gm / math.sqrt(c.earth_gm))
        * factor
        / (c.moon_a_km**3 * (1.0 - c.moon_ecc**2) ** 1.5)
        * a_km**1.5
    )


def characteristic_rate_solar_exterior(
    a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """太阳外支四极长期特征频率 omega^☉（论文式 97），rad/s。"""
    c = constants
    return (
        0.75
        * (c.sun_gm / math.sqrt(c.earth_gm))
        / (c.sun_a_km**3 * (1.0 - c.sun_ecc**2) ** 1.5)
        * a_km**1.5
    )


def laplace_radius_geolunar(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地心（geolunar）Laplace 半径 r_L（论文式 98/99），km。

    .. math::
        r_L^5 = \\frac{2\\mu_\\oplus J_2 R_\\oplus^2}{D}, \\quad
        D = \\frac{\\mu_\\mathbb{C}}{a_\\mathbb{C}^3}
        \\frac{1-\\sin^2 I_\\mathbb{C}/2}{(1-e_\\mathbb{C}^2)^{3/2}}
        + \\frac{\\mu_\\odot}{a_\\odot^3}
        \\frac{1}{(1-e_\\odot^2)^{3/2}}

    terrestrial / cislunar 分界：日月力矩与地球扁率长期效应相当之处。
    黄金值 48812.40 km = 7.6531 R⊕ = 0.12732 a☾（论文 "≈7.7 R⊕ ≈ 0.13 a☾"）。
    r_L 是纯系统常数——式 95/96/97 均不含卫星根数，与所考察轨道无关。
    """
    c = constants
    numerator = 2.0 * c.earth_gm * c.earth_j2 * c.earth_ref_radius_km**2
    lunar = (
        (c.moon_gm / c.moon_a_km**3)
        * (1.0 - math.sin(math.radians(c.moon_inc_deg)) ** 2 / 2.0)
        / ((1.0 - c.moon_ecc**2) ** 1.5)
    )
    solar = c.sun_gm / (c.sun_a_km**3 * (1.0 - c.sun_ecc**2) ** 1.5)
    return (numerator / (lunar + solar)) ** 0.2


def laplace_radius_selenocentric(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月心 Laplace 半径 rho_L（论文式 124/125），km。

    月球扁率力矩与地日四极潮汐相当之处（轴对称小倾角近似，外力矩对月
    赤道平均、保留 e☾/e☉）。黄金值 3846 km = 2.21 R☾（高约 2109 km）。
    """
    c = constants
    numerator = 2.0 * c.moon_gm * c.moon_j2 * c.moon_radius_km**2
    earth = c.earth_gm / (c.moon_a_km**3 * (1.0 - c.moon_ecc**2) ** 1.5)
    solar = c.sun_gm / (c.sun_a_km**3 * (1.0 - c.sun_ecc**2) ** 1.5)
    return (numerator / (earth + solar)) ** 0.2


def hill_radius_moon(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球 Hill 尺度 rho_H（论文式 110 近似式），km。

    .. math::
        (\\rho_H)^\\mathbb{C} = a_\\mathbb{C}
        \\left(\\frac{\\mu_\\mathbb{C}}{3\\mu_\\oplus}\\right)^{1/3}

    复现陷阱：论文数值 61364 km（= 35.32 R☾）**只由近似式复现**；完整式
    ``a☾(μ☾/(3(μ⊕+μ☾)))^{1/3}`` 给 61114 km（差 −250 km）。一阶意义下
    L1/L2 位于距月 ±rho_H。
    """
    c = constants
    return c.moon_a_km * (c.moon_gm / (3.0 * c.earth_gm)) ** (1.0 / 3.0)


def hill_radius_earth(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球（Sun–(Earth+Moon) 问题）Hill 尺度 r_H（论文式 111 近似式），km。

    黄金值 ≈ 1.4966e6 km = 3.9034 a☾（T = 210.88 d，论文表值 3.90）。
    """
    c = constants
    return c.sun_a_km * (c.earth_gm / (3.0 * c.sun_gm)) ** (1.0 / 3.0)


def soi_laplace_moon(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球 Laplace–Tisserand 球形 SOI 代理 rho_SOI（论文式 116），km。

    黄金值 66010 km = 37.99 R☾。角依赖活动面（式 115）在方照 psi=π/2 处
    取该最大值。
    """
    c = constants
    return c.moon_a_km * (c.moon_gm / c.earth_gm) ** 0.4


def activity_surface_moon(psi_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """Laplace–Tisserand 角依赖活动面 rho_act(psi)（论文式 115），km。

    psi 从反地方向量起（与 :func:`battin_soi_moon` 同约定）。
    """
    c = constants
    return (
        c.moon_a_km
        * (c.moon_gm / c.earth_gm) ** 0.4
        * (1.0 + 3.0 * math.cos(psi_rad) ** 2) ** (-0.1)
    )


def chebotarev_radius_moon(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球 Chebotarev 直接力对等球 rho_Ch（论文式 117），km。

    直接力平衡 mu☾/rho² = mu⊕/a☾²。复算 42520.9 km = 24.47 R☾；
    论文 km 值 42499 差 0.05%（论文内部舍入），R☾ 比值一致。
    """
    c = constants
    return c.moon_a_km * math.sqrt(c.moon_gm / c.earth_gm)


def battin_soi_moon(psi_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球 Battin 一阶非对称 SOI 面 rho_B(psi)（论文式 118），km。

    .. math::
        (\\rho_B)^\\mathbb{C}(\\psi) = a_\\mathbb{C}\\left[
        \\left(\\tfrac{\\mu_\\mathbb{C}}{\\mu_\\oplus}\\right)^{-2/5}(1+3\\cos^2\\psi)^{1/10}
        - \\tfrac{2}{5}\\cos\\psi\\tfrac{1+6\\cos^2\\psi}{1+3\\cos^2\\psi}\\right]^{-1}

    **角度约定（陷阱④）**：psi 从反地方向量起——psi=0（背地）给最大
    64201 km，psi=π（朝地）给最小 52009 km。论文正文行内定义与数值矛盾，
    按数值复现约定实现（详见模块 docstring 与 ADR 0041）。
    """
    c = constants
    cos_psi = math.cos(psi_rad)
    bracket = (c.moon_gm / c.earth_gm) ** (-0.4) * (
        1.0 + 3.0 * cos_psi**2
    ) ** 0.1 - 0.4 * cos_psi * (1.0 + 6.0 * cos_psi**2) / (1.0 + 3.0 * cos_psi**2)
    return c.moon_a_km / bracket


def soi_laplace_earth(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球 Laplace–Tisserand SOI（论文式 120），km。黄金值 ≈ 2.41 a☾。"""
    c = constants
    return c.sun_a_km * (c.earth_gm / c.sun_gm) ** 0.4


def chebotarev_radius_earth(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球 Chebotarev 直接力对等球（论文式 121），km。黄金值 0.68 a☾。"""
    c = constants
    return c.sun_a_km * math.sqrt(c.earth_gm / c.sun_gm)


def battin_soi_earth(psi_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球 Battin 一阶 SOI 面（论文式 122），km。

    psi 从反日方向量起（太阳在 psi=π）；日地质量比下变形微弱，方照处
    回到 :func:`soi_laplace_earth`。
    """
    c = constants
    cos_psi = math.cos(psi_rad)
    bracket = (c.earth_gm / c.sun_gm) ** (-0.4) * (
        1.0 + 3.0 * cos_psi**2
    ) ** 0.1 - 0.4 * cos_psi * (1.0 + 6.0 * cos_psi**2) / (1.0 + 3.0 * cos_psi**2)
    return c.sun_a_km / bracket


def tidal_parity_radius(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """日月潮汐平权半长轴 a_TP（论文式 127/128），km。

    .. math::
        \\left(\\frac{a_{TP}}{a_\\mathbb{C}}\\right)^5 = \\frac{\\mu_\\mathbb{C}}{\\mu_\\odot}
        \\left(\\frac{a_\\odot}{a_\\mathbb{C}}\\right)^3
        \\left(1-\\tfrac{3}{2}\\sin^2 i_\\mathbb{C}\\right)
        \\left(1+\\tfrac{3}{2}e_\\mathbb{C}^2\\right)(1-e_\\odot^2)^{3/2}

    黄金值 447948 km = 1.1684 a☾（T = 34.53 d）。注意本式用月球**内支**
    因子 ``1 - 3/2 sin²i``，与 r_L 的外支因子 ``1 - sin²I/2`` 不同，勿混用。
    论文三条 caveat：非 gateway/零速度面/切换面；与 L2 距离贴近纯属巧合；
    非瞬时第三体加速度交接半径。
    """
    c = constants
    inc_rad = math.radians(c.moon_inc_deg)
    rhs = (
        (c.moon_gm / c.sun_gm)
        * (c.sun_a_km / c.moon_a_km) ** 3
        * (1.0 - 1.5 * math.sin(inc_rad) ** 2)
        * (1.0 + 1.5 * c.moon_ecc**2)
        * (1.0 - c.sun_ecc**2) ** 1.5
    )
    return c.moon_a_km * rhs**0.2


def tisserand_parameter(
    a_km: float,
    ecc: float,
    inc_rel_moon_rad: float = 0.0,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> float:
    """月球 Tisserand 参数 T☾（论文式 140）。

    .. math:: T_\\mathbb{C} = \\frac{a_\\mathbb{C}}{a}
        + 2\\cos I^\\mathbb{C}\\sqrt{\\frac{a}{a_\\mathbb{C}}(1-e^2)}}

    Jacobi 常数的轨道根数空间类比（共面退化即论文 Fig. 3 等值线）。
    参考值：共面 a=a☾ 圆轨道 T☾=3——它**低于**第一个地月颈口开启阈值，
    是参考值而非阈值本身。
    """
    c = constants
    ratio = a_km / c.moon_a_km
    return 1.0 / ratio + 2.0 * math.cos(inc_rel_moon_rad) * math.sqrt(ratio * (1.0 - ecc**2))


def geo_radius_km(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """地球静止轨道半径（地心 Kepler，恒星日自转），km。

    供 (a,e) 图 GEO 穿越线使用（论文 Fig. 8/11 走廊曲线，横轴量 r_GEO
    未给数值，此处按恒星日派生，≈42164 km）。GEO 在本框架中**不是**
    分区判据（论文 §2.3 明确批评以 GEO 为界），仅作走廊参考线。
    """
    c = constants
    n_earth = 2.0 * math.pi / c.earth_sidereal_day_s
    return (c.earth_gm / n_earth**2) ** (1.0 / 3.0)
