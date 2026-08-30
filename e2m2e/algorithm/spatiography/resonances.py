"""共振名义中心阶梯（spatiography resonances）。

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...status import ConvergenceState, FailureCause, ResultStatus
from .constants import PRIMER_DEFAULTS, PrimerConstants

__all__ = [
    "PRIMER_RESONANCE_KINDS",
    "ResonanceCenter",
    "ResonanceLadderResult",
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
