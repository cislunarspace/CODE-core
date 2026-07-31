"""DFH SK_STATISTIC.TXT 蒙特卡洛统计文件解析与写入。

文件为以下两种形态之一（不含角动量管理 / 含角动量管理）::

    <仿真次数> <总delta-V> <最大delta-V>
    <仿真次数> <总delta-V> <最大delta-V> <姿态delta-V> <姿态delta-V(不影响轨道)>

末尾附一行文字 ``The number of failed Monte Carlo tests is N``（失败次数，
列数与数据行不同，必须单独识别）。照 MATLAB ``parse_sk_statistic.m``：
数据行取每行全部数值词元（>=3 个），按最大列数对齐、缺列补 NaN。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SKStatistic", "parse_sk_statistic", "read_sk_statistic", "write_sk_statistic"]

_NUM_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")
_FAILED_RE = re.compile(r"is\s+(\d+)")

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


def parse_sk_statistic(raw: str) -> SKStatistic:
    """解析 SK_STATISTIC.TXT 文本，返回 :class:`SKStatistic`。"""
    num_failed: int | None = None
    row_list: list[list[float]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "failed monte carlo" in line.lower():
            m = _FAILED_RE.search(line)
            if m:
                num_failed = int(m.group(1))
            continue
        toks = _NUM_RE.findall(line)
        if len(toks) >= 3:
            row_list.append([float(t) for t in toks])

    ncol = max((len(r) for r in row_list), default=0)
    rows = np.full((len(row_list), ncol), np.nan)
    for i, r in enumerate(row_list):
        rows[i, : len(r)] = r

    return SKStatistic(rows=rows, num_failed=num_failed, raw_text=raw)


def read_sk_statistic(path: str | Path) -> SKStatistic:
    """从文件读入 SK_STATISTIC.TXT。"""
    return parse_sk_statistic(Path(path).read_text(encoding="utf-8"))


def write_sk_statistic(stats: SKStatistic, path: str | Path) -> None:
    """把统计表写入 SK_STATISTIC.TXT。

    数据行按 DFH 输出风格：运行序号（12 宽整数）+ 各统计列（15 位小数
    定点），末尾一行 ``The number of failed Monte Carlo tests is N``。
    """
    lines = []
    for i, row in enumerate(stats.rows, start=1):
        cols = "".join(f"{v:>20.15f}" for v in row)
        lines.append(f"{i:>12d}{cols}")
    if stats.num_failed is not None:
        lines.append(f"The number of failed Monte Carlo tests is {stats.num_failed}")
    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
