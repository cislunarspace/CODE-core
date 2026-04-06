"""天体名称常量与预定义列表。

本模块定义 :class:`BodyName` 字符串常量类，与 :class:`~e2m2e.core.spice.SPICEManager`
及 :class:`~e2m2e.core.ephemeris_system.EphemerisSystem` 配合使用，避免在代码中
散落裸字符串，并获得 IDE 自动补全支持。

典型用法::

    from e2m2e.core import BodyName, EphemerisSystem, SPICEManager

    # 构造仅含内太阳系天体的星历系统
    system = EphemerisSystem(
        bodies=BodyName.INNER_SOLAR_SYSTEM,
        spice=spice_mgr,
        origin=BodyName.EARTH,
    )

    # 使用全太阳系天体
    system = EphemerisSystem(
        bodies=BodyName.ALL,
        spice=spice_mgr,
        origin=BodyName.EARTH,
    )

天体名称采用 NAIF 标准名称的大写形式，与 SPICE 内核的命名约定保持一致。
所有名称均与 :data:`~e2m2e.core.spice._GM_VALUES` 的键名对应，
保证 :meth:`~e2m2e.core.spice.SPICEManager.get_gm` 可直接命中缓存。
"""

from __future__ import annotations

from typing import Final, List


class BodyName:
    """NAIF 天体名称字符串常量类。

    提供太阳系主要天体的名称常量及常用分组列表。
    所有常量均为大写字符串，与 SPICE 内核命名约定一致。

    示例::

        from e2m2e.core import BodyName

        print(BodyName.EARTH)        # "EARTH"
        print(BodyName.MOON)         # "MOON"
        print(BodyName.ALL)          # ['SUN', 'MERCURY', ...]
    """

    # ------------------------------------------------------------------
    # 单体常量
    # ------------------------------------------------------------------

    SUN: Final[str] = "SUN"
    """太阳"""

    MERCURY: Final[str] = "MERCURY"
    """水星"""

    VENUS: Final[str] = "VENUS"
    """金星"""

    EARTH: Final[str] = "EARTH"
    """地球"""

    MOON: Final[str] = "MOON"
    """月球"""

    MARS: Final[str] = "MARS"
    """火星"""

    JUPITER: Final[str] = "JUPITER"
    """木星"""

    SATURN: Final[str] = "SATURN"
    """土星"""

    URANUS: Final[str] = "URANUS"
    """天王星"""

    NEPTUNE: Final[str] = "NEPTUNE"
    """海王星"""

    EMB: Final[str] = "EMB"
    """地月质心（Earth-Moon Barycenter，NAIF ID 3）"""

    PLUTO: Final[str] = "PLUTO"
    """冥王星（矮行星）"""

    # ------------------------------------------------------------------
    # 预定义分组列表
    # ------------------------------------------------------------------

    INNER_SOLAR_SYSTEM: Final[List[str]] = [
        "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS",
    ]
    """内太阳系天体：太阳 + 四颗内行星 + 月球。

    适用于近地/近月任务的引力摄动建模，计算量较小。
    """

    OUTER_SOLAR_SYSTEM: Final[List[str]] = [
        "JUPITER", "SATURN", "URANUS", "NEPTUNE",
    ]
    """外太阳系四大行星。

    用于长周期任务或需要考虑木星摄动的场景。
    """

    MAJOR_PLANETS: Final[List[str]] = [
        "SUN",
        "MERCURY", "VENUS", "EARTH", "MARS",
        "JUPITER", "SATURN", "URANUS", "NEPTUNE",
    ]
    """太阳 + 八大行星（不含月球）。"""

    ALL: Final[List[str]] = [
        "SUN",
        "MERCURY", "VENUS", "EARTH", "MOON", "MARS",
        "JUPITER", "SATURN", "URANUS", "NEPTUNE",
    ]
    """全太阳系常用天体：太阳 + 八大行星 + 月球。

    覆盖所有在 ``_GM_VALUES`` 中有硬编码 GM 值的天体（不含 EMB、Pluto），
    适合需要完整太阳系引力场的高精度建模。
    """

    EARTH_MOON: Final[List[str]] = ["EARTH", "MOON"]
    """地月系统：仅地球和月球。"""

    EARTH_MOON_SUN: Final[List[str]] = ["EARTH", "MOON", "SUN"]
    """CR3BP 星历修正标准组合：地球 + 月球 + 太阳。

    与 :class:`~e2m2e.core.ephemeris_dynamics.EphemerisDynamics` 的默认
    测试配置一致，适用于 DRO/LLO 等近月轨道的星历修正。
    """
