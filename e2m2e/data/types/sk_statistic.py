"""蒙特卡洛统计表容器。

通用数据容器（ADR 0011 迁移，源：``io/sk_statistic.py`` 的
``SKStatistic``）。DFH 格式解析/写出（parse/read/write）保留在 ``io/``
作临时脚本；算法层（站保）直接使用本容器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 数据行各列含义（列数随是否含角动量管理而取舍）
COLUMNS = (
    "run_index",
    "total_delta_v",
    "max_delta_v",
    "attitude_delta_v",
    "attitude_delta_v_independent",
)


@dataclass
class SKStatistic:
    """SK_STATISTIC 统计表容器。

    Attributes:
        rows: 数据行矩阵，形状 ``(n, k)``，k 为 3（无角动量）或 4/5
            （含角动量）；列含义见 :data:`COLUMNS` 前 k 项，单位 m/s
        num_failed: 蒙特卡洛失败次数；文件无末行文字时为 ``None``
    """

    rows: np.ndarray
    num_failed: int | None
    raw_text: str = field(default="", repr=False)

    def __len__(self) -> int:
        return self.rows.shape[0]

    @property
    def has_attitude(self) -> bool:
        """是否含角动量管理（姿态 delta-V 列）。"""
        return self.rows.shape[1] >= 4
