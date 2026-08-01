"""机动序列表容器。

通用数据容器（ADR 0011 迁移，源：``io/maneuvers.py`` 的
``ManeuverTable``）。DFH 格式解析/写出（parse/read/write）保留在 ``io/``
作临时脚本；算法层（站保）直接使用本容器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ManeuverTable:
    """机动序列表容器。

    Attributes:
        mjd_tdb: MJD 形式历元（TDB，天），形状 ``(n,)``
        delta_v_mps: 机动脉冲大小（m/s），形状 ``(n,)``
    """

    mjd_tdb: np.ndarray
    delta_v_mps: np.ndarray
    raw_text: str = field(default="", repr=False)

    def __len__(self) -> int:
        return len(self.mjd_tdb)
