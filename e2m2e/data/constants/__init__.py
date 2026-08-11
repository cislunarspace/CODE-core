"""物理常数层：全库物理常数的唯一来源（阶段 1 骨架）。

本层与 ``e2m2e.data.templates`` 平级，职责分开：``templates`` 管任务/算法默认
参数，``constants`` 管物理量真值表。阶段 1 只建骨架，不迁移任何现有文件。
"""

from __future__ import annotations

from .bodies import (
    EARTH,
    EMB,
    JUPITER,
    MARS,
    MERCURY,
    MOON,
    NEPTUNE,
    PLUTO,
    SATURN,
    SUN,
    URANUS,
    VENUS,
    Body,
)
from .datums import Datum
from .sources import ConstantSource
from .universal import (
    AU_KM,
    DAYS_PER_JULIAN_CENTURY,
    DAYS_PER_JULIAN_YEAR,
    EARTH_MOON_DISTANCE_KM,
    GRAVITATIONAL_CONSTANT,
    KM_TO_M,
    SECONDS_PER_DAY,
    SECONDS_PER_JULIAN_YEAR,
    SOLAR_FLUX_TSI_W_M2,
    SOLAR_FLUX_W_M2,
    SOLAR_PRESSURE_1AU,
    SPEED_OF_LIGHT_KMS,
)

__all__ = [
    "AU_KM",
    "Body",
    "DAYS_PER_JULIAN_CENTURY",
    "DAYS_PER_JULIAN_YEAR",
    "EARTH",
    "EARTH_MOON_DISTANCE_KM",
    "EMB",
    "GRAVITATIONAL_CONSTANT",
    "JUPITER",
    "KM_TO_M",
    "MARS",
    "MERCURY",
    "MOON",
    "NEPTUNE",
    "PLUTO",
    "SATURN",
    "SECONDS_PER_DAY",
    "SECONDS_PER_JULIAN_YEAR",
    "SOLAR_FLUX_TSI_W_M2",
    "SOLAR_FLUX_W_M2",
    "SOLAR_PRESSURE_1AU",
    "SPEED_OF_LIGHT_KMS",
    "SUN",
    "ConstantSource",
    "Datum",
    "URANUS",
    "VENUS",
]
