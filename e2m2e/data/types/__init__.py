"""通用数据类型：状态向量、时间类型、轨道容器、轨迹容器。

State/Epoch 是类型别名（单值 → 别名）；Orbit/EphemerisTable/NominalOrbit 是
真容器类（多字段/多列 → 类）。算法层保持 numpy 不强制包装（ADR 0011）。

实现状态：已迁移（ADR 0011 第 1 批）。``Orbit``/``OrbitFamily`` 自
``core/orbit.py`` 迁入；``EphemerisTable`` 自 ``io/ephemeris.py`` 迁入；
``NominalOrbit`` 为新类型（FR1↔FR2 契约，插值器待 FR1 落地）。
"""

from .epoch import Epoch, EpochUtc, EtSec, JdTdb
from .orbit import Orbit
from .state import OrbitState, State
from .trajectory import EphemerisTable, NominalOrbit

__all__ = [
    "Epoch",
    "EpochUtc",
    "EtSec",
    "JdTdb",
    "Orbit",
    "OrbitState",
    "State",
    "EphemerisTable",
    "NominalOrbit",
]
