"""Primer 常数集装配（spatiography 子包自用）。

把 ``e2m2e.data.constants`` 的 ``[primer]`` 节装配为 frozen dataclass，
供本子包全部解析尺度函数显式传参使用——函数本体保持纯函数形态
（照 ``algorithm/dynamics/potential.py`` 范式），常数只经此结构注入。

常数纪律：``PrimerConstants`` 是 Rosengren et al. 2026 §5 的自洽集合，
不得与 ``Datum.DE421`` 的 GM 或 ``EARTH_MOON_DISTANCE_KM`` 混搭（见 ADR 0041）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...data.constants import (
    AU_KM,
    MOON,
    PRIMER_CITATION,
    PRIMER_EARTH_GM,
    PRIMER_EARTH_J2,
    PRIMER_EARTH_REF_RADIUS_KM,
    PRIMER_EARTH_SIDEREAL_DAY_S,
    PRIMER_MOON_A_KM,
    PRIMER_MOON_ECC,
    PRIMER_MOON_GM,
    PRIMER_MOON_INC_DEG,
    PRIMER_MOON_J2,
    PRIMER_SUN_A_AU,
    PRIMER_SUN_ECC,
    PRIMER_SUN_GM,
)

__all__ = ["PRIMER_CITATION", "PRIMER_DEFAULTS", "PrimerConstants"]


@dataclass(frozen=True)
class PrimerConstants:
    """Primer 分区常数集（Rosengren et al. 2026 §5，物理单位）。

    Attributes:
        earth_j2: 地球二阶带谐系数（GGM02）。
        moon_j2: 月球二阶带谐系数（Legnaro & Efthymiopoulos 2024）。
        moon_a_km: 月球平均轨道半长轴，km（Simon et al. 1994）。
        moon_ecc: 月球平均轨道偏心率。
        moon_inc_deg: 月球平均轨道倾角，deg（相对黄道）。
        sun_a_au: 太阳视地心轨道半长轴，au（Simon et al. 1994）。
        sun_ecc: 太阳视地心轨道偏心率。
        earth_gm: GM⊕，km³/s²（JPL/NAIF SPICE）。
        moon_gm: GM☾，km³/s²（JPL/NAIF SPICE）。
        sun_gm: GM☉，km³/s²（JPL/NAIF SPICE）。
        earth_ref_radius_km: 地球参考半径 R⊕，km（GGM02 归一）。
        moon_radius_km: 月球参考半径 R☾，km（IAU2015，与论文反推值一致）。
        earth_sidereal_day_s: 地球恒星日，s（IERS；供 GEO 派生线）。
        au_km: 天文单位，km（IAU2012；太阳视轨道换算用）。
    """

    earth_j2: float
    moon_j2: float
    moon_a_km: float
    moon_ecc: float
    moon_inc_deg: float
    sun_a_au: float
    sun_ecc: float
    earth_gm: float
    moon_gm: float
    sun_gm: float
    earth_ref_radius_km: float
    moon_radius_km: float
    earth_sidereal_day_s: float
    au_km: float

    @property
    def moon_mass_parameter(self) -> float:
        """地月 CR3BP 质量参数 mu_bar = GM☾ / (GM⊕ + GM☾)。"""
        return self.moon_gm / (self.earth_gm + self.moon_gm)

    @property
    def sun_a_km(self) -> float:
        """太阳视轨道半长轴，km（au 换算，IAU2012 au）。"""
        return self.sun_a_au * self.au_km

    @property
    def moon_mean_motion_rad_s(self) -> float:
        """月球（地心 Kepler，绕 μ⊕）平均运动 n☾ = sqrt(GM⊕ / a☾³)，rad/s。

        注意：这是 Table 1 周期列 T = 2π/n 的口径（绕地球二体）；与 CR3BP
        会合系旋转角速度（绕 GM⊕+GM☾）相差约 0.4%，两者不可混用。
        """
        return (self.earth_gm / self.moon_a_km**3) ** 0.5

    @property
    def moon_period_days(self) -> float:
        """月球轨道恒星周期 T☾ = 2π/n☾，天。

        由常数解析派生（约 27.34460 天）。**禁止硬编码 27.346**——Table 1
        周期列的逐位复现依赖本派生值（论文校验结论，见 ADR 0041）。
        """
        import math

        return 2.0 * math.pi / self.moon_mean_motion_rad_s / 86400.0

    @property
    def cr3bp_mean_motion_rad_s(self) -> float:
        """地月 CR3BP 会合系角速度 n = sqrt((GM⊕+GM☾)/a☾³)，rad/s。"""
        return ((self.earth_gm + self.moon_gm) / self.moon_a_km**3) ** 0.5


PRIMER_DEFAULTS = PrimerConstants(
    earth_j2=PRIMER_EARTH_J2,
    moon_j2=PRIMER_MOON_J2,
    moon_a_km=PRIMER_MOON_A_KM,
    moon_ecc=PRIMER_MOON_ECC,
    moon_inc_deg=PRIMER_MOON_INC_DEG,
    sun_a_au=PRIMER_SUN_A_AU,
    sun_ecc=PRIMER_SUN_ECC,
    earth_gm=PRIMER_EARTH_GM,
    moon_gm=PRIMER_MOON_GM,
    sun_gm=PRIMER_SUN_GM,
    earth_ref_radius_km=PRIMER_EARTH_REF_RADIUS_KM,
    moon_radius_km=MOON.require_mean_radius_km(),
    earth_sidereal_day_s=PRIMER_EARTH_SIDEREAL_DAY_S,
    au_km=AU_KM,
)
