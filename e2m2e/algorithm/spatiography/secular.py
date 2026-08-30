"""长期共振 loci 与 vZLK 相图（spatiography secular）。

Rosengren et al. 2026 §4.2–§4.3 的解析骨架（ADR 0041 Phase 3a 第二、
三件交付）：

- **长期前置频率**（式 47/48、56/57）：外支 ω_ext^b(a) 与月内支
  ω_int^☾(a)。注意与 ``scales.characteristic_rate_*``（式 95–97，Laplace
  半径用的倾角因子 ``1 − sin²I/2``）是**不同式号**：本模块用 K_b =
  1 − (3/2)sin²I_b（式 48/57），勿混用。
- **拱线驻定 loci**（式 75–78）：cislunar 侧日月同为外支、角结构可合并，
  驻定条件 5cos²I = 1 − e²（式 76，(a,I) 平面水平线）；translunar 侧
  月内支 + 日外支混合，驻定倾角随 a 弯曲（式 78 闭式），两个极限：月内
  主导 cos²I → 1/5（式 79，逆 Kozai 63.4°/116.6°，式 80）、日外主导
  cos²I → (1−e²)/5（式 81）。
- **vZLK 相图与时间尺度**（式 64–71）：日月理想化（K=1、共参考面）
  四极 Hamiltonian（式 65）的可积流；第一积分 c1（式 67）、c2（式 68）
  的等值线即 (ω, sqrt(1−e²)) 平面的经典 vZLK 图；临界倾角 39.2°/
  140.8°（式 64）；特征频率 ν_vZLK（式 69）与时间尺度（式 70/71）；
  有效性筛选（J2 在 r_L 内抑制、a→a☾ 破坏双平均）。

这些 loci 是**最低阶 spatiographic 骨架**（论文 §4.3 结尾的定性声明），
不是月距附近的精确局部共振位置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import PRIMER_DEFAULTS, PrimerConstants
from .scales import characteristic_rate_j2, laplace_radius_geolunar

__all__ = [
    "VZLK_CRITICAL_INCLINATION_DEG",
    "SecularLocusCurve",
    "VzlkPortrait",
    "VzlkValidity",
    "apsidal_rate_cislunar",
    "apsidal_rate_translunar",
    "apsidal_stationary_inclination_translunar",
    "nodal_rate_ext_moon",
    "nodal_rate_int_moon",
    "secular_loci_curves",
    "secular_prefactor_ext_moon",
    "secular_prefactor_ext_sun",
    "secular_prefactor_int_moon",
    "vzlk_frequency_rad_s",
    "vzlk_phase_portrait",
    "vzlk_tidal_sum",
    "vzlk_timescale_days",
    "vzlk_validity",
]

#: vZLK 临界倾角（式 64）：arccos(sqrt(3/5)) = 39.2315°；补角 140.7685°。
VZLK_CRITICAL_INCLINATION_DEG = math.degrees(math.acos(math.sqrt(3.0 / 5.0)))


# ---------------------------------------------------------------------------
# 长期前置频率（式 47/48 外支、式 56/57 月内支）
# ---------------------------------------------------------------------------


def secular_prefactor_ext_moon(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球外支四极长期前置频率 ω_ext^☾(a)（式 47 + 48），rad/s。

    .. math:: \\omega_{\\mathrm{ext}}^{\\mathbb{C}}(a) = \\frac{3}{4}
        \\frac{\\mu_\\mathbb{C}}{\\sqrt{\\mu_\\oplus}} K_\\mathbb{C}
        \\frac{a^{3/2}}{a_\\mathbb{C}^3 (1-e_\\mathbb{C}^2)^{3/2}}

    倾角因子 K☾ = 1 − (3/2)sin²I☾（式 48）≈ 0.9879。与 Laplace 半径的
    特征频率（式 96，因子 1 − sin²I/2）不是同一式，勿混用。
    """
    c = constants
    k_moon = 1.0 - 1.5 * math.sin(math.radians(c.moon_inc_deg)) ** 2
    return (
        0.75
        * (c.moon_gm / math.sqrt(c.earth_gm))
        * k_moon
        * a_km**1.5
        / (c.moon_a_km**3 * (1.0 - c.moon_ecc**2) ** 1.5)
    )


def secular_prefactor_ext_sun(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """太阳外支四极长期前置频率 ω_ext^☉(a)（式 47，K_☉ = 1），rad/s。"""
    c = constants
    return (
        0.75
        * (c.sun_gm / math.sqrt(c.earth_gm))
        * a_km**1.5
        / (c.sun_a_km**3 * (1.0 - c.sun_ecc**2) ** 1.5)
    )


def secular_prefactor_int_moon(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """月球内支（translunar）四极长期前置频率 ω_int^☾(a)（式 56 + 57），rad/s。

    .. math:: \\omega_{\\mathrm{int}}^{\\mathbb{C}}(a) = \\frac{3}{4}
        \\frac{\\mu_\\mathbb{C}}{\\sqrt{\\mu_\\oplus}} K_\\mathbb{C}
        \\left(1+\\tfrac{3}{2}e_\\mathbb{C}^2\\right)
        \\frac{a_\\mathbb{C}^2}{a^{7/2}}

    与外支在 a = a☾ 处近似衔接（差 (1+3/2 e☾²)(1−e☾²)^{3/2} ≈ 1.0000），
    回归测试锁定该连续性。
    """
    c = constants
    k_moon = 1.0 - 1.5 * math.sin(math.radians(c.moon_inc_deg)) ** 2
    return (
        0.75
        * (c.moon_gm / math.sqrt(c.earth_gm))
        * k_moon
        * (1.0 + 1.5 * c.moon_ecc**2)
        * c.moon_a_km**2
        / a_km**3.5
    )


def nodal_rate_ext_moon(
    a_km: float, ecc: float, inc_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """月球外支交点进动率 Ω̇_ext^☾（式 54），rad/s（长期诊断量）。"""
    w_ext = secular_prefactor_ext_moon(a_km, constants)
    return -w_ext * (1.0 + 1.5 * ecc**2) / math.sqrt(1.0 - ecc**2) * math.cos(inc_rad)


def nodal_rate_int_moon(
    a_km: float, ecc: float, inc_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """月球内支交点进动率 Ω̇_int^☾（式 63），rad/s（长期诊断量）。"""
    w_int = secular_prefactor_int_moon(a_km, constants)
    return -w_int * math.cos(inc_rad) / (1.0 - ecc**2) ** 2


# ---------------------------------------------------------------------------
# 拱线驻定 loci（式 75–78）
# ---------------------------------------------------------------------------


def apsidal_rate_cislunar(
    a_km: float, ecc: float, inc_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """cislunar 侧合并拱线进动率 ω̇（式 75），rad/s。

    日月同为外支、角结构相同可合并：ω̇ = (ω_ext^☾ + ω_ext^☉)/2 ·
    (5cos²I − 1 + e²)/sqrt(1−e²)。零点即式 76 的水平线 5cos²I = 1−e²。
    """
    w_sum = secular_prefactor_ext_moon(a_km, constants) + secular_prefactor_ext_sun(a_km, constants)
    return 0.5 * w_sum * (5.0 * math.cos(inc_rad) ** 2 - 1.0 + ecc**2) / math.sqrt(1.0 - ecc**2)


def apsidal_rate_translunar(
    a_km: float, ecc: float, inc_rad: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float:
    """translunar 侧混合拱线进动率 ω̇（式 77），rad/s。

    月内支 + 日外支：ω̇ = ω_int^☾/2 · (5cos²I − 1)/(1−e²)² + ω_ext^☉/2 ·
    (5cos²I − 1 + e²)/sqrt(1−e²)。零点给随 a 弯曲的驻定倾角（式 78）。
    """
    w_int = secular_prefactor_int_moon(a_km, constants)
    w_sun = secular_prefactor_ext_sun(a_km, constants)
    cos2 = math.cos(inc_rad) ** 2
    return 0.5 * w_int * (5.0 * cos2 - 1.0) / (1.0 - ecc**2) ** 2 + 0.5 * w_sun * (
        5.0 * cos2 - 1.0 + ecc**2
    ) / math.sqrt(1.0 - ecc**2)


def apsidal_stationary_inclination_translunar(
    a_km: float, ecc: float, constants: PrimerConstants = PRIMER_DEFAULTS
) -> float | None:
    """translunar 侧拱线驻定倾角 I_ω̇=0(a; e)（式 78 闭式），rad。

    .. math:: \\cos^2 I = \\frac{\\omega_{\\mathrm{int}}^{\\mathbb{C}}
        + \\omega_{\\mathrm{ext}}^{\\odot}(1-e^2)^{5/2}}
        {5\\left[\\omega_{\\mathrm{int}}^{\\mathbb{C}}
        + \\omega_{\\mathrm{ext}}^{\\odot}(1-e^2)^{3/2}\\right]}

    Returns:
        驻定倾角（取 [0°, 90°] 支）；右端不在 (0, 1] 时返回 None
        （该 a/e 下无驻定解）。
    """
    w_int = secular_prefactor_int_moon(a_km, constants)
    w_sun = secular_prefactor_ext_sun(a_km, constants)
    one_me2 = 1.0 - ecc**2
    numerator = w_int + w_sun * one_me2**2.5
    denominator = 5.0 * (w_int + w_sun * one_me2**1.5)
    ratio = numerator / denominator
    if not 0.0 < ratio <= 1.0:
        return None
    return math.acos(math.sqrt(ratio))


@dataclass(frozen=True)
class SecularLocusCurve:
    """一条长期驻定 loci 曲线（(a, I) 平面）。

    Attributes:
        branch: ``"cislunar"``（式 76 水平线）或 ``"translunar"``（式 78）。
        eccentricity: 切片偏心率。
        a_km: 半长轴采样，km。
        inclination_rad: 驻定倾角（与 a_km 对齐；无解处为 NaN）。
        formula_id: 论文式号。
    """

    branch: str
    eccentricity: float
    a_km: np.ndarray
    inclination_rad: np.ndarray
    formula_id: str


def secular_loci_curves(
    a_grid_km: list[float] | tuple[float, ...] | np.ndarray | None = None,
    e_slices: list[float] | tuple[float, ...] = (0.0, 0.3, 0.6),
    branches: tuple[str, ...] = ("cislunar", "translunar"),
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> tuple[SecularLocusCurve, ...]:
    """拱线驻定 loci 曲线族（式 76 水平线 + 式 78 弯曲线）。

    Args:
        a_grid_km: translunar 支的半长轴采样；None = 1.02–3.90 a☾ 共 73 点
            （地球 Hill 界内）。cislunar 支为水平线，不受 a 网格影响，
            沿同一网格展开便于统一绘制。
        e_slices: 偏心率切片。
        branches: ``"cislunar"`` / ``"translunar"`` 子集。
        constants: Primer 常数集。

    Returns:
        曲线元组：每 (branch, e_slice) 一条；cislunar 支各点同值（式 76
        与 a 无关），translunar 支无解处置 NaN。
    """
    c = constants
    if a_grid_km is None:
        grid = np.linspace(1.02 * c.moon_a_km, 3.90 * c.moon_a_km, 73)
    else:
        grid = np.asarray(a_grid_km, dtype=float)
    curves: list[SecularLocusCurve] = []
    for branch in branches:
        if branch not in ("cislunar", "translunar"):
            raise ValueError(f"未知 branch={branch!r}，支持 cislunar/translunar")
        for e in e_slices:
            ecc = float(e)
            if branch == "cislunar":
                # 式 76：5cos²I = 1 − e²，水平线（与 a 无关）。
                ratio = (1.0 - ecc**2) / 5.0
                inc = math.acos(math.sqrt(ratio)) if 0.0 < ratio <= 1.0 else math.nan
                incs = np.full_like(grid, inc)
                curves.append(
                    SecularLocusCurve(
                        branch=branch,
                        eccentricity=ecc,
                        a_km=grid.copy(),
                        inclination_rad=incs,
                        formula_id="Eq.76",
                    )
                )
            else:
                incs = np.array(
                    [
                        (lambda v: math.nan if v is None else v)(
                            apsidal_stationary_inclination_translunar(float(a), ecc, c)
                        )
                        for a in grid
                    ]
                )
                curves.append(
                    SecularLocusCurve(
                        branch=branch,
                        eccentricity=ecc,
                        a_km=grid.copy(),
                        inclination_rad=incs,
                        formula_id="Eq.78",
                    )
                )
    return tuple(curves)


# ---------------------------------------------------------------------------
# vZLK（式 64–71）
# ---------------------------------------------------------------------------


def vzlk_tidal_sum(constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """理想化日月合并四极潮汐系数 Σ（式 65/69 方括号首项），1/s²。

    .. math:: \\Sigma = \\frac{\\mu_\\odot}{a_\\odot^3(1-e_\\odot^2)^{3/2}}
        + \\frac{\\mu_\\mathbb{C}}{a_\\mathbb{C}^3(1-e_\\mathbb{C}^2)^{3/2}}

    口径：K_☉ = K_☾ = 1（月轨面与黄道共面的理想化，论文 §4.2 的声明）。
    """
    c = constants
    return c.sun_gm / (c.sun_a_km**3 * (1.0 - c.sun_ecc**2) ** 1.5) + c.moon_gm / (
        c.moon_a_km**3 * (1.0 - c.moon_ecc**2) ** 1.5
    )


def vzlk_frequency_rad_s(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """vZLK 特征四极频率 ν_vZLK(a)（式 69），rad/s。"""
    return 0.75 * a_km**1.5 / math.sqrt(constants.earth_gm) * vzlk_tidal_sum(constants)


def vzlk_timescale_days(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> float:
    """vZLK 时间尺度 t_vZLK（式 71，Antognini 2015 归一化），天。

    .. math:: t_{\\mathrm{vZLK}} \\sim \\frac{16}{15}
        \\frac{\\sqrt{\\mu_\\oplus}}{a^{3/2}} \\Sigma^{-1}

    与式 70（t ~ ν^{-1}）差一个 16/20 固定因子，两式都是论文给出的
    量级估计；本函数取式 71 口径，式 70 可由 :func:`vzlk_frequency_rad_s`
    取倒数得到。
    """
    c = constants
    seconds = (16.0 / 15.0) * math.sqrt(c.earth_gm) / a_km**1.5 / vzlk_tidal_sum(c)
    return seconds / 86400.0


@dataclass(frozen=True)
class VzlkValidity:
    """vZLK 骨架在给定 a 处的有效性筛选（论文 §4.2 的两条 caveat）。

    Attributes:
        a_km: 评估半长轴。
        alpha: 展开比 a/a☾。
        j2_suppressed: True 表示 a < r_L，地球扁率进动抑制经典 vZLK
            （特征频率口径：ω^⊕(a) > Σ·a^{3/2} 侧）。
        j2_rate_ratio: J2 特征频率（式 95）与 ν_vZLK（式 69）之比。
        double_averaging_warning: True 表示 α > 0.8，双平均展开快速失效
            （阈值登记于 ADR 0041 Phase 3 增补，实现者定标）。
    """

    a_km: float
    alpha: float
    j2_suppressed: bool
    j2_rate_ratio: float
    double_averaging_warning: bool


def vzlk_validity(a_km: float, constants: PrimerConstants = PRIMER_DEFAULTS) -> VzlkValidity:
    """vZLK 骨架有效性筛选：J2 抑制（r_L 内）与双平均失效（a→a☾）。"""
    c = constants
    alpha = a_km / c.moon_a_km
    ratio = characteristic_rate_j2(a_km, c) / vzlk_frequency_rad_s(a_km, c)
    return VzlkValidity(
        a_km=a_km,
        alpha=alpha,
        j2_suppressed=a_km < laplace_radius_geolunar(c),
        j2_rate_ratio=ratio,
        double_averaging_warning=alpha > 0.8,
    )


@dataclass(frozen=True)
class VzlkPortrait:
    """vZLK 相图（(ω, sqrt(1−e²)) 平面的 c2 等值线族，式 65–68）。

    Attributes:
        c1: 第一积分 (1−e²)cos²I（式 67）。
        levels: 等值线 c2 值（式 68）。
        curves: 每条折线带其 c2 等值，点为 (ω_deg, sqrt(1−e²))。
        e_max: 分离线上最大偏心率 sqrt(1 − 5c1/3)（c1 < 3/5 时有限，
            在 ω = 90° 处达到；c1 ≥ 3/5 时全相图为环流量，取 NaN）。
    """

    c1: float
    levels: tuple[float, ...]
    curves: tuple[tuple[float, np.ndarray], ...]
    e_max: float

    @property
    def has_separatrix(self) -> bool:
        """c1 < 3/5 时存在分离线（ω 可振荡区）。"""
        return self.c1 < 0.6


def _c2_field(c1: float, omega_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """c2(ω, y) 场（式 68，y = sqrt(1−e²)）。

    由式 67 反解 cos²I = c1/y²（有效区 y ≥ sqrt(c1)），
    c2 = (1−y²)·(2/5 − (1 − c1/y²)·sin²ω)；y < sqrt(c1) 处 NaN。
    """
    sin2_i = 1.0 - c1 / y_grid**2
    valid = (y_grid >= math.sqrt(c1)) & (y_grid <= 1.0)
    e2 = np.where(valid, 1.0 - y_grid**2, np.nan)
    c2 = e2 * (0.4 - np.where(valid, sin2_i, 0.0) * np.sin(omega_grid) ** 2)
    return np.where(valid, c2, np.nan)


def _marching_squares(
    field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, level: float
) -> list[np.ndarray]:
    """简版 marching squares：提取 field = level 的等值线段并串成折线。

    网格节点 值；x 沿 axis=0、y 沿 axis=1。对 NaN 节点按无效处理
    （不穿过）。返回若干条折线，每条 (n, 2) 数组，坐标 (x, y)。
    本实现只做线性插值边点 + 端点贪心串接，满足相图数据层用途，
    不追求拓扑完备（相图曲线本身就是数据层，非物理面）。
    """
    f = field - level
    ny, nx = f.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00, v10 = f[j, i], f[j, i + 1]
            v01, v11 = f[j + 1, i], f[j + 1, i + 1]
            if not all(np.isfinite([v00, v10, v01, v11])):
                continue
            x0, x1 = x_grid[i], x_grid[i + 1]
            y0, y1 = y_grid[j], y_grid[j + 1]
            pts: list[tuple[float, float]] = []
            for va, vb, pa, pb in (
                (v00, v10, (x0, y0), (x1, y0)),
                (v10, v11, (x1, y0), (x1, y1)),
                (v11, v01, (x1, y1), (x0, y1)),
                (v01, v00, (x0, y1), (x0, y0)),
            ):
                if (va < 0.0) != (vb < 0.0):
                    t = va / (va - vb)
                    pts.append((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])))
            if len(pts) >= 2:
                segments.append((pts[0], pts[1]))
            elif len(pts) == 3:
                segments.append((pts[0], pts[1]))
                segments.append((pts[1], pts[2]))

    if not segments:
        return []
    # 贪心串接：从每条未用段出发按端点匹配延展（段可反向接入头/尾）。
    used = [False] * len(segments)
    polylines: list[np.ndarray] = []
    for idx in range(len(segments)):
        if used[idx]:
            continue
        used[idx] = True
        chain = [segments[idx][0], segments[idx][1]]
        changed = True
        while changed:
            changed = False
            for k, (p, q) in enumerate(segments):
                if used[k]:
                    continue
                for end in (chain[-1], chain[0]):
                    for a, b in ((p, q), (q, p)):
                        if (abs(end[0] - a[0]) < 1e-9) and (abs(end[1] - a[1]) < 1e-9):
                            if end is chain[-1]:
                                chain.append(b)
                            else:
                                chain.insert(0, b)
                            used[k] = True
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        polylines.append(np.array(chain))
    return polylines


def vzlk_phase_portrait(
    c1: float,
    levels: list[float] | tuple[float, ...] | None = None,
    n_omega: int = 181,
    n_inc: int = 161,
) -> VzlkPortrait:
    """vZLK 相图：固定 c1 的 c2 等值线族（式 67/68，(ω, sqrt(1−e²)) 平面）。

    理想化合并日月四极 Hamiltonian（式 65，K=1）可积，第一积分 c1、c2
    （式 67/68）沿轨迹守恒，故 c2 的等值线即轨迹。场在 (ω, y) 直接建：
    y = sqrt(1−e²) ∈ [sqrt(c1), 1]，cos²I = c1/y²（式 67 反解）。

    Args:
        c1: 第一积分 (1−e²)cos²I ∈ (0, 1]。c1 < 3/5 存在分离线。
        levels: c2 等值线值；None = 在场值域内取 9 条（含分离线 c2 = 0）。
        n_omega: ω 采样（0–360°）；n_y: y = sqrt(1−e²) 采样。

    Returns:
        :class:`VzlkPortrait`；curves 的每条折线点为 (ω_deg, sqrt(1−e²))。

    Raises:
        ValueError: c1 不在 (0, 1]。
    """
    if not 0.0 < c1 <= 1.0:
        raise ValueError(f"c1 须在 (0, 1]，得到 {c1}")
    omega_grid = np.radians(np.linspace(0.0, 360.0, n_omega))
    y_lo, y_hi = math.sqrt(c1), 1.0
    n_y = n_inc
    y_grid = np.linspace(y_lo, y_hi, n_y)
    c2 = _c2_field(c1, omega_grid[None, :], y_grid[:, None])
    omega_deg = np.degrees(omega_grid)

    if levels is None:
        finite = c2[np.isfinite(c2)]
        lo, hi = float(finite.min()), float(finite.max())
        span = max(hi, 0.4) - min(lo, 0.0)
        levels = sorted({round(min(lo, 0.0) + span * i / 8.0, 6) for i in range(9)} | {0.0})

    curves: list[tuple[float, np.ndarray]] = []
    for level in levels:
        for pts in _marching_squares(c2, omega_deg, y_grid, float(level)):
            if len(pts) >= 2:
                curves.append((float(level), pts))

    e_max = math.sqrt(max(0.0, 1.0 - 5.0 * c1 / 3.0)) if c1 < 0.6 else math.nan
    return VzlkPortrait(
        c1=float(c1),
        levels=tuple(float(v) for v in levels),
        curves=tuple(curves),
        e_max=float(e_max),
    )
