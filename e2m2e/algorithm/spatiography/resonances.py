"""共振名义中心阶梯与 Gallardo 半解析共振半宽（spatiography resonances）。

Table 1 / Table 2 全部低阶平运动共振名义中心与开普勒周期，物理单位。
标签约定（论文式 84/85）：``k:k_b``，第一整数属卫星、第二属摄动体；
共振条件 k_b·n − k·n_b ≈ 0，T/T_b ≈ k_b/k，名义中心 a/a_b = (k_b/k)^{2/3}
（乘质量因子）。

复现陷阱（ADR 0041）：
- 周期列由 T☾ = 2π·sqrt(a☾³/GM⊕) **解析派生**（≈27.34460 天），绝不硬
  编码 27.346，否则 Table 1 周期列逐位复现失败；
- Table 2 的 16 条月心外地球共振只能用质量因子 ``(GM☾/GM⊕)^{1/3}``
  逐位复现——论文式 (126) 字面的 ``mu_bar^{1/3}``（mu_bar = GM☾/(GM⊕+GM☾)）
  系统性低 0.4%，按论文数值口径实现并在 docstring 注记；
- Table 1 的 1:3☾ 行周期表值 82.00 与自洽派生值 82.03 差 0.03 天，属论
  文表内自身舍入不一致，回归测试对该行放宽容差。

Gallardo 半解析半宽（论文 §5.3 式 100–104，计算设置对齐论文 Fig. 8）：
- 共面切片（卫星相对月球倾角置零）、Simon 1994 月根数、2ρ_H 近遇截断
  （Gallardo et al 2021；截断样本不参与平均，余下样本重归一）；
- 共振角 σ = k☾λ − kλ☾ + γ（式 100）：k☾ 为摄动体整数、k 为卫星整数；
  λ 沿共振环由约束 σ = const 反解（式 101 的采样实现）；γ 是拱点/节点
  经度的缓变组合，共面问题里只水平平移 R(σ)——σ_s/σ_u 随之平移而
  ΔR（式 103）与半宽（式 104）γ 不变，故实现取 γ=0 并在 docstring 声明；
- 半宽 Δa = sqrt(8ΔR/3)/n（式 104，n 取名义中心处开普勒平均运动）；
- 稳定平衡点 σ_s 是数值平均 R(σ) 的**极小**、不稳定点 σ_u 是极大（由
  半长期 Hamiltonian 式 102 的 Hessian 判稳推出，测试锁定该约定）；
- 1:1 共振带宽在本方法下系统性高估（论文 §5.3 line 959 明确声明），
  交付文档须转述，不得把 1:1 包络当 gateway 边界用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...status import ConvergenceState, FailureCause, ResultStatus
from .constants import PRIMER_DEFAULTS, PrimerConstants
from .scales import hill_radius_moon

__all__ = [
    "PRIMER_RESONANCE_KINDS",
    "ResonanceCenter",
    "ResonanceLadderResult",
    "ResonanceWidthEnvelope",
    "ResonanceWidthProfile",
    "ResonanceWidthResult",
    "gallardo_resonance_width",
    "gallardo_width_envelopes",
    "resonance_centers",
]

#: 支持的共振族标识（kind 参数取值）。
PRIMER_RESONANCE_KINDS: tuple[str, ...] = (
    "interior_lunar",
    "exterior_lunar",
    "solar",
    "exterior_terrestrial_selenocentric",
)

# 各族 (k, k_b) 整数对，顺序即论文 Table 1 / Table 2 行序。
_INTERIOR_LUNAR: tuple[tuple[int, int], ...] = (
    (5, 1),
    (4, 1),
    (3, 1),
    (5, 2),
    (2, 1),
    (5, 3),
    (3, 2),
    (4, 3),
    (5, 4),
)
_EXTERIOR_LUNAR: tuple[tuple[int, int], ...] = (
    (4, 5),
    (3, 4),
    (2, 3),
    (3, 5),
    (1, 2),
    (2, 5),
    (1, 3),
    (1, 4),
    (1, 5),
)
_SOLAR: tuple[tuple[int, int], ...] = ((5, 1), (4, 1), (3, 1), (5, 2), (2, 1))
_SOLAR_SECONDARY: tuple[tuple[int, int], ...] = ((6, 1),)
_SELENOCENTRIC_EARTH: tuple[tuple[int, int], ...] = (
    (8, 1),
    (7, 1),
    (6, 1),
    (5, 1),
    (9, 2),
    (4, 1),
    (7, 2),
    (10, 3),
    (3, 1),
    (8, 3),
    (5, 2),
    (7, 3),
    (9, 4),
    (2, 1),
    (9, 5),
    (7, 4),
)

# 位置符号：地心系共振用 ☾/☉，月心系共振用 ⊕（与论文 Table 1/2 一致）。
_KIND_BODY: dict[str, str] = {
    "interior_lunar": "☾",
    "exterior_lunar": "☾",
    "solar": "☉",
    "exterior_terrestrial_selenocentric": "⊕",
}


@dataclass(frozen=True)
class ResonanceCenter:
    """单个共振名义中心。

    Attributes:
        label: 论文记号（如 ``"5:1☾"``、``"8:1⊕"``）。
        kind: 所属共振族（:data:`PRIMER_RESONANCE_KINDS` 之一）。
        k: 卫星侧整数（标签第一整数）。
        k_body: 摄动体侧整数（标签第二整数）。
        a_km: 名义中心半长轴，km（地心系共振为地心距；月心系为月心距）。
        a_over_a_moon: 以月球平均半长轴归一的位置（地心系共振）；
            月心系共振为 ``None``。
        rho_over_moon_radius: 以月球半径归一的位置（月心系共振）；
            地心系共振为 ``None``。
        period_days: 名义中心处圆轨道开普勒周期，天（口径随族：
            地心系绕 GM⊕、日支绕 GM☉+GM⊕、月心系绕 GM☾）。
        secondary: True 表示论文标注的低阶保守截断之外的次级项（6:1☉）。
    """

    label: str
    kind: str
    k: int
    k_body: int
    a_km: float
    a_over_a_moon: float | None
    rho_over_moon_radius: float | None
    period_days: float
    secondary: bool = field(default=False)


@dataclass(frozen=True)
class ResonanceLadderResult:
    """共振梯查询结果（状态契约：status/cause/message 三元组）。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    centers: tuple[ResonanceCenter, ...]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def __len__(self) -> int:
        return len(self.centers)

    def __iter__(self):
        return iter(self.centers)


def _kepler_period_days(a_km: float, gm_km3_s2: float) -> float:
    return 2.0 * math.pi * math.sqrt(a_km**3 / gm_km3_s2) / 86400.0


def _solve_kepler_eccentric(mean_anom: np.ndarray, ecc: float) -> np.ndarray:
    """向量化牛顿迭代解开普勒方程 M = E − e·sinE。

    M 先折到 [−π, π] 加速收敛；对 e ≤ 0.97 五次迭代残差达 1e-15 量级
    （回归测试锁定）。e ∈ [0, 1)。
    """
    m = np.mod(mean_anom + np.pi, 2.0 * math.pi) - np.pi
    ecc_anom = m.copy()
    for _ in range(8):
        residual = ecc_anom - ecc * np.sin(ecc_anom) - m
        ecc_anom = ecc_anom - residual / (1.0 - ecc * np.cos(ecc_anom))
    return ecc_anom


def _coplanar_positions(
    a_km: float, ecc: float, varpi_rad: float, mean_longitudes: np.ndarray
) -> np.ndarray:
    """共面（月球轨道面内）轨道位置，由平黄经经开普勒方程反解。

    Args:
        a_km: 半长轴，km；ecc: 偏心率；varpi_rad: 近日点黄经（自参考 x 轴起）。
        mean_longitudes: 平黄经数组（rad，任意范围）。

    Returns:
        位置数组 (n, 2)，km；平面即月球平均轨道面。
    """
    ecc_anom = _solve_kepler_eccentric(mean_longitudes - varpi_rad, ecc)
    r_km = a_km * (1.0 - ecc * np.cos(ecc_anom))
    true_long = varpi_rad + np.arctan2(
        math.sqrt(1.0 - ecc * ecc) * np.sin(ecc_anom), np.cos(ecc_anom) - ecc
    )
    return np.stack([r_km * np.cos(true_long), r_km * np.sin(true_long)], axis=-1)


@dataclass(frozen=True)
class ResonanceWidthProfile:
    """单个（共振， 偏心率切片）的 Gallardo 数值平均诊断（式 100–104）。

    Attributes:
        label: 共振标签（``"2:1☾"``）。
        k: 卫星侧整数；k_body: 摄动体侧整数；eccentricity: 切片偏心率。
        a_center_km: 名义中心半长轴（式 87 口径），km。
        sigma_rad: 共振角采样网格 [0, 2π)，rad。
        r_disturbing: 数值平均摄动函数 R(σ)（式 101），km²/s²。
        sigma_s_rad: 稳定平衡点（R 的极小），rad。
        sigma_u_rad: 不稳定平衡点（R 的极大），rad。
        delta_r_km2_s2: ΔR = R(σ_u) − R(σ_s)（式 103），km²/s²。
        delta_a_km: 半宽 Δa（式 104），km。
        n_truncated: 2ρ_H 近遇截断剔除的样本数。
        n_samples: 总采样数（n_sigma × n_lambda × k_body）。

    稳定/不稳定平衡点约定：半长期 Hamiltonian K（式 102）的 a 向 Hessian
    K_aa = −(3/4)μ⊕/a³ < 0，平衡点稳定要求 Hessian 定负，即 R''(σ) > 0，
    故 σ_s 取 R 的极小、σ_u 取极大，ΔR ≥ 0。
    """

    label: str
    k: int
    k_body: int
    eccentricity: float
    a_center_km: float
    sigma_rad: np.ndarray
    r_disturbing: np.ndarray
    sigma_s_rad: float
    sigma_u_rad: float
    delta_r_km2_s2: float
    delta_a_km: float
    n_truncated: int
    n_samples: int

    @property
    def truncated_fraction(self) -> float:
        """截断样本占比；接近 1 表示该切片近遇主导，半宽不可信应弃用。"""
        if self.n_samples == 0:
            return 0.0
        return self.n_truncated / self.n_samples


@dataclass(frozen=True)
class ResonanceWidthEnvelope:
    """单条共振在偏心率网格上的半宽包络（Fig. 8 阴影带的数据形态）。

    Attributes:
        label/k/k_body/a_center_km: 同 :class:`ResonanceWidthProfile`。
        eccentricities: 偏心率切片（给定顺序）。
        delta_a_km: 逐切片半宽，km。
    """

    label: str
    k: int
    k_body: int
    a_center_km: float
    eccentricities: tuple[float, ...]
    delta_a_km: tuple[float, ...]

    @property
    def lower_a_km(self) -> tuple[float, ...]:
        """包络下沿 a_center − Δa（逐切片），km。"""
        return tuple(self.a_center_km - d for d in self.delta_a_km)

    @property
    def upper_a_km(self) -> tuple[float, ...]:
        """包络上沿 a_center + Δa（逐切片），km。"""
        return tuple(self.a_center_km + d for d in self.delta_a_km)


@dataclass(frozen=True)
class ResonanceWidthResult:
    """半宽包络查询结果（状态契约三元组齐备）。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    envelopes: tuple[ResonanceWidthEnvelope, ...]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def __len__(self) -> int:
        return len(self.envelopes)

    def __iter__(self):
        return iter(self.envelopes)


def gallardo_resonance_width(
    k: int,
    k_body: int,
    eccentricity: float,
    a_km: float | None = None,
    varpi_offset_deg: float = 180.0,
    n_sigma: int = 72,
    n_lambda: int = 180,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> ResonanceWidthProfile:
    """Gallardo 半解析共振半宽（论文 §5.3 式 100–104，共面切片）。

    在名义中心 a_r = (k_body/k)^{2/3}·a☾ 处，把瞬时月球摄动函数（式 1
    地心口径：直接项 + 间接项，内/外支几何通用）沿共振环做数值平均
    （式 101）：λ☾ 均匀铺 [0, 2π·k_body)、卫星平黄经由约束
    ``k_body·λ − k·λ☾ + γ = σ`` 反解（γ=0，共面下 γ 只水平平移 R(σ)，
    见模块 docstring）。近遇截断：|r − r☾| < 2ρ_H 的样本剔除后对余下
    样本重归一（论文 Fig. 8 计算设置，Gallardo et al 2021）。

    Args:
        k: 卫星侧整数（标签第一整数）。
        k_body: 摄动体（月球）侧整数；与 k 须互素。
        eccentricity: 卫星偏心率切片（共面：相对月球倾角为零）。
        a_km: 评估半长轴；None = 名义中心（式 104 的 n 口径）。
        varpi_offset_deg: 卫星近日点黄经相对月球近日点黄经的夹角；缺省
            180°（反平行，与论文 §7.3 制图切片同约定）。改变它改变
            R(σ) 形状与 ΔR（物理上不同的拱线几何），不是 γ 平移。
        n_sigma: σ 网格点数；n_lambda: λ☾ 每 2π 的采样数（总采样
            数 = n_sigma × n_lambda × k_body）。
        constants: Primer 常数集。

    Returns:
        :class:`ResonanceWidthProfile`；Δa 由式 104
        ``Δa = sqrt(8ΔR/3)/n(a_center)``。

    Raises:
        ValueError: (k, k_body) 非正或不互素、偏心率越界 [0, 1)、采样过疏。
    """
    if k < 1 or k_body < 1:
        raise ValueError(f"k 与 k_body 须为正整数，得到 {k}:{k_body}")
    if math.gcd(k, k_body) != 1:
        raise ValueError(f"(k, k_body)=({k}, {k_body}) 须互素，否则共振环采样不闭合")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"eccentricity 须在 [0, 1)，得到 {eccentricity}")
    if n_sigma < 12 or n_lambda < 24:
        raise ValueError(f"采样过疏：n_sigma≥12、n_lambda≥24，得到 {n_sigma}/{n_lambda}")

    c = constants
    a_center = (k_body / k) ** (2.0 / 3.0) * c.moon_a_km
    a_eval = a_center if a_km is None else float(a_km)

    varpi_moon = 0.0  # 月球近日点黄经取参考 x 轴（只有差角进入物理）
    varpi_sat = math.radians(varpi_offset_deg)
    cutoff_km = 2.0 * hill_radius_moon(c)

    sigma = np.linspace(0.0, 2.0 * math.pi, n_sigma, endpoint=False)
    lam_moon = np.linspace(0.0, 2.0 * math.pi * k_body, n_lambda * k_body, endpoint=False)
    # 共振环约束 k_body·λ − k·λ☾ = σ（γ=0）→ λ = (σ + k·λ☾)/k_body
    lam_sat = (sigma[:, None] + k * lam_moon[None, :]) / k_body

    sat = _coplanar_positions(a_eval, eccentricity, varpi_sat, lam_sat)
    moon = _coplanar_positions(c.moon_a_km, c.moon_ecc, varpi_moon, np.tile(lam_moon, (n_sigma, 1)))

    dist = np.hypot(sat[:, :, 0] - moon[:, :, 0], sat[:, :, 1] - moon[:, :, 1])
    moon_r3 = np.linalg.norm(moon, axis=2) ** 3
    r_instant = c.moon_gm * (
        1.0 / dist - (sat[:, :, 0] * moon[:, :, 0] + sat[:, :, 1] * moon[:, :, 1]) / moon_r3
    )
    keep = dist >= cutoff_km
    n_truncated = int(np.count_nonzero(~keep))
    kept_per_sigma = keep.sum(axis=1)
    r_avg = np.zeros(n_sigma)
    np.divide(
        np.where(keep, r_instant, 0.0).sum(axis=1),
        kept_per_sigma,
        out=r_avg,
        where=kept_per_sigma > 0,
    )

    if np.all(kept_per_sigma == 0):
        # 整条共振环被近遇截断：ΔR 无定义，按零宽返回，截断占比暴露不可信。
        delta_r = 0.0
        i_s = i_u = 0
    else:
        i_s = int(np.argmin(r_avg))
        i_u = int(np.argmax(r_avg))
        delta_r = float(r_avg[i_u] - r_avg[i_s])
    n_mean_motion = math.sqrt(c.earth_gm / a_center**3)
    delta_a = math.sqrt(8.0 * delta_r / 3.0) / n_mean_motion

    return ResonanceWidthProfile(
        label=f"{k}:{k_body}☾",
        k=k,
        k_body=k_body,
        eccentricity=float(eccentricity),
        a_center_km=a_center,
        sigma_rad=sigma,
        r_disturbing=r_avg,
        sigma_s_rad=float(sigma[i_s]),
        sigma_u_rad=float(sigma[i_u]),
        delta_r_km2_s2=float(delta_r),
        delta_a_km=float(delta_a),
        n_truncated=n_truncated,
        n_samples=n_sigma * n_lambda * k_body,
    )


#: 缺省包络共振集：Table 1 内月 9 条 + 1:1（gateway 参考，半宽系统性
#: 高估，见模块 docstring）+ 外月 9 条。
_DEFAULT_ENVELOPE_PAIRS: tuple[tuple[int, int], ...] = (
    *_INTERIOR_LUNAR,
    (1, 1),
    *_EXTERIOR_LUNAR,
)


def gallardo_width_envelopes(
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    e_grid: list[float] | tuple[float, ...] | None = None,
    varpi_offset_deg: float = 180.0,
    n_sigma: int = 72,
    n_lambda: int = 180,
    constants: PrimerConstants = PRIMER_DEFAULTS,
) -> ResonanceWidthResult:
    """一批共振在偏心率网格上的 Gallardo 半宽包络（Fig. 8 数据层）。

    Args:
        pairs: (k, k_body) 互素对列表；None = 内月 9 条 + 1:1 + 外月 9 条。
        e_grid: 偏心率切片；None = 0.0–0.9 共 19 点（等距）。
        其余参数与 :func:`gallardo_resonance_width` 同名同义。

    Returns:
        :class:`ResonanceWidthResult`，envelopes 与 pairs 同序。
    """
    selected = _DEFAULT_ENVELOPE_PAIRS if pairs is None else tuple(pairs)
    if e_grid is None:
        grid = tuple(0.9 * i / 18.0 for i in range(19))
    else:
        grid = tuple(float(e) for e in e_grid)
    envelopes: list[ResonanceWidthEnvelope] = []
    for k, k_body in selected:
        widths = [
            gallardo_resonance_width(
                k,
                k_body,
                e,
                varpi_offset_deg=varpi_offset_deg,
                n_sigma=n_sigma,
                n_lambda=n_lambda,
                constants=constants,
            )
            for e in grid
        ]
        envelopes.append(
            ResonanceWidthEnvelope(
                label=widths[0].label,
                k=k,
                k_body=k_body,
                a_center_km=(k_body / k) ** (2.0 / 3.0) * constants.moon_a_km,
                eccentricities=grid,
                delta_a_km=tuple(w.delta_a_km for w in widths),
            )
        )
    return ResonanceWidthResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        envelopes=tuple(envelopes),
    )


def resonance_centers(
    kind: str = "all", constants: PrimerConstants = PRIMER_DEFAULTS
) -> ResonanceLadderResult:
    """计算共振名义中心阶梯（Table 1 / Table 2 全表）。

    Args:
        kind: ``"all"`` 或 :data:`PRIMER_RESONANCE_KINDS` 之一。
        constants: Primer 常数集。

    Returns:
        :class:`ResonanceLadderResult`，centers 按 Table 行序排列；
        ``kind="all"`` 时顺序为内月、外月、日、月心外地球。

    Raises:
        ValueError: kind 不受支持时（Facial 层翻译为 INVALID_PARAMS）。
    """
    if kind not in ("all", *PRIMER_RESONANCE_KINDS):
        raise ValueError(f"未知的共振族 kind={kind!r}，支持 all/{'/'.join(PRIMER_RESONANCE_KINDS)}")

    c = constants
    centers: list[ResonanceCenter] = []

    def _add(
        pairs,
        kind_key: str,
        *,
        mass_factor: float,
        base_km: float,
        gm_period: float,
        secondary_marker: tuple[tuple[int, int], ...] = (),
    ) -> None:
        body = _KIND_BODY[kind_key]
        seleno = kind_key == "exterior_terrestrial_selenocentric"
        for k, k_body in pairs:
            ratio = mass_factor * (k_body / k) ** (2.0 / 3.0)
            a_km = ratio * base_km
            centers.append(
                ResonanceCenter(
                    label=f"{k}:{k_body}{body}",
                    kind=kind_key,
                    k=k,
                    k_body=k_body,
                    a_km=a_km,
                    a_over_a_moon=(None if seleno else a_km / c.moon_a_km),
                    rho_over_moon_radius=(a_km / c.moon_radius_km if seleno else None),
                    period_days=_kepler_period_days(a_km, gm_period),
                    secondary=(k, k_body) in secondary_marker,
                )
            )

    if kind in ("all", "interior_lunar"):
        _add(
            _INTERIOR_LUNAR,
            "interior_lunar",
            mass_factor=1.0,
            base_km=c.moon_a_km,
            gm_period=c.earth_gm,
        )
    if kind in ("all", "exterior_lunar"):
        _add(
            _EXTERIOR_LUNAR,
            "exterior_lunar",
            mass_factor=1.0,
            base_km=c.moon_a_km,
            gm_period=c.earth_gm,
        )
    if kind in ("all", "solar"):
        # 日支周期口径：卫星仍绕地球（μ⊕）开普勒，与太阳视轨道平运动通约；
        # 名义中心即论文 Table 1 的日支 a/a☾ 与 T 列（5:1☉ → 73.05 d）。
        _add(
            _SOLAR,
            "solar",
            mass_factor=(c.earth_gm / (c.sun_gm + c.earth_gm)) ** (1.0 / 3.0),
            base_km=c.sun_a_km,
            gm_period=c.earth_gm,
        )
        _add(
            _SOLAR_SECONDARY,
            "solar",
            mass_factor=(c.earth_gm / (c.sun_gm + c.earth_gm)) ** (1.0 / 3.0),
            base_km=c.sun_a_km,
            gm_period=c.earth_gm,
            secondary_marker=((6, 1),),
        )
    if kind in ("all", "exterior_terrestrial_selenocentric"):
        # 复现陷阱：质量因子取 (GM☾/GM⊕)^{1/3}（逐位复现 Table 2），
        # 而非论文式 (126) 字面的 mu_bar^{1/3}（系统性低 0.4%）。
        _add(
            _SELENOCENTRIC_EARTH,
            "exterior_terrestrial_selenocentric",
            mass_factor=(c.moon_gm / c.earth_gm) ** (1.0 / 3.0),
            base_km=c.moon_a_km,
            gm_period=c.moon_gm,
        )

    return ResonanceLadderResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        centers=tuple(centers),
    )
