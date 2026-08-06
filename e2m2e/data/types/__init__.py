"""通用数据类型：状态向量、时间类型、轨道容器、轨迹容器。

State/Epoch 是类型别名（单值 → 别名）；Orbit/EphemerisTable/NominalOrbit 是
真容器类（多字段/多列 → 类）。算法层保持 numpy 不强制包装（ADR 0011）。

实现状态：已迁移（ADR 0011 第 1 批）。``Orbit``/``OrbitFamily`` 自
``core/orbit.py`` 迁入；``EphemerisTable`` 自 ``io/ephemeris.py`` 迁入；
``NominalOrbit`` 为新类型（FR1↔FR2 契约，插值器待 FR1 落地）。
DFH 文本格式序列化函数（parse/read/write）与容器同生命周期，也从此处导出。
"""

from .epoch import Epoch, EpochUtc, EtSec, JdTdb
from .maneuver import ManeuverTable, parse_maneuvers, read_maneuvers, write_maneuvers
from .orbit import Orbit
from .sk_statistic import (
    COLUMNS,
    SKStatistic,
    parse_sk_statistic,
    read_sk_statistic,
    write_sk_statistic,
)
from .state import OrbitState, State
from .trajectory import (
    EphemerisTable,
    NominalOrbit,
    parse_ephemeris,
    read_ephemeris,
    write_ephemeris,
)

__all__ = [
    "Epoch",
    "EpochUtc",
    "EtSec",
    "JdTdb",
    "ManeuverTable",
    "parse_maneuvers",
    "read_maneuvers",
    "write_maneuvers",
    "Orbit",
    "OrbitState",
    "SKStatistic",
    "COLUMNS",
    "parse_sk_statistic",
    "read_sk_statistic",
    "write_sk_statistic",
    "State",
    "EphemerisTable",
    "NominalOrbit",
    "parse_ephemeris",
    "read_ephemeris",
    "write_ephemeris",
]
