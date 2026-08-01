"""DFH SK_STATISTIC.TXT 蒙特卡洛统计文件解析与写入（临时脚本，ADR 0011）。

``SKStatistic`` 通用容器已迁至 ``e2m2e.data.types.sk_statistic``；本模块
保留 DFH 专属格式的 parse/read/write，作为开发期临时脚本。

文件形态：数据行 ``<仿真次数> <总delta-V> <最大delta-V>``（含角动量管理时
追加姿态列），末尾一行 ``The number of failed Monte Carlo tests is N``。
照 MATLAB ``parse_sk_statistic.m``：数据行取每行全部数值词元（>=3 个），
按最大列数对齐、缺列补 NaN。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from e2m2e.data.types.sk_statistic import COLUMNS, SKStatistic

__all__ = [
    "SKStatistic", "COLUMNS", "parse_sk_statistic",
    "read_sk_statistic", "write_sk_statistic",
]

_NUM_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")
_FAILED_RE = re.compile(r"is\s+(\d+)")


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


def write_sk_statistic(stats: SKStatistic, path: str | Path) -> Path:
    """把统计表写入 SK_STATISTIC.TXT，返回写入的文件路径。"""
    lines = []
    for i, row in enumerate(stats.rows, start=1):
        cols = "".join(f"{v:>20.15f}" for v in row)
        lines.append(f"{i:>12d}{cols}")
    if stats.num_failed is not None:
        lines.append(f"The number of failed Monte Carlo tests is {stats.num_failed}")
    out = Path(path)
    out.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return out
