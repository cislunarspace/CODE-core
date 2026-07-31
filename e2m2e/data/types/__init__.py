"""通用数据类型：状态向量、时间类型、轨道容器、轨迹容器。

State/Epoch 是类型别名（单值 → 别名）；Orbit/EphemerisTable/NominalOrbit 是
真容器类（多字段/多列 → 类）。算法层保持 numpy 不强制包装（ADR 0011）。

实现状态：部分实现。``Orbit``/``EphemerisTable`` 为骨架（待从 core/io 迁入完整
实现），``NominalOrbit`` 为新类型接口待定稿。
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
