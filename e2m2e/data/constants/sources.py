"""基准/来源枚举与元数据。

每个物理常量必须标注出处；本枚举提供规范化的来源标识。
"""

from __future__ import annotations

from enum import Enum


class ConstantSource(str, Enum):
    """物理常量来源标识（str Enum，可直接作为元数据值使用）。"""

    DE421 = "DE421"
    DE430 = "DE430"
    DE440 = "DE440"
    WGS84 = "WGS84"
    IAU2015 = "IAU2015"
    IAU2012 = "IAU2012"
    CODATA2018 = "CODATA2018"
    GMAT = "GMAT"
    GRGM900C = "GRGM900C"
    IERS = "IERS"
    LITERATURE = "literature"
    NAIF = "NAIF"
    Pesce2023 = "Pesce2023"
    SI = "SI"
    Vallado = "Vallado"
    Cui2025 = "Cui2025"
