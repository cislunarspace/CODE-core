"""机动序列表容器。

通用数据容器（ADR 0011 迁移，源：``io/maneuvers.py`` 的
``ManeuverTable``）。DFH 文本格式序列化函数与本容器同生命周期，
算法层（站保）直接使用本容器与序列化函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["ManeuverTable", "parse_maneuvers", "read_maneuvers", "write_maneuvers"]

_NUM_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")


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


def parse_maneuvers(raw: str) -> ManeuverTable:
    """解析 MANEUVERS.TXT 文本，返回 :class:`ManeuverTable`。"""
    mjd: list[float] = []
    dv: list[float] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = _NUM_RE.findall(line)
        if len(toks) < 2:
            continue
        mjd.append(float(toks[0]))
        dv.append(float(toks[1]))
    return ManeuverTable(
        mjd_tdb=np.array(mjd, dtype=float),
        delta_v_mps=np.array(dv, dtype=float),
        raw_text=raw,
    )


def read_maneuvers(path: str | Path) -> ManeuverTable:
    """从文件读入 MANEUVERS.TXT。"""
    return parse_maneuvers(Path(path).read_text(encoding="utf-8"))


def _format_delta_v(v: float) -> str:
    """DFH 风格脉冲大小：|v| ≥ 0.1 用定点 15 位小数，否则科学计数。"""
    if abs(v) < 0.1:
        return f"{v:>22.15E}"
    return f"{v:>22.15f}"


def write_maneuvers(table: ManeuverTable, path: str | Path) -> Path:
    """把机动序列表写入 MANEUVERS.TXT，返回写入的文件路径。"""
    lines = [
        f"{mjd:>20.10f}{_format_delta_v(dv)}"
        for mjd, dv in zip(table.mjd_tdb, table.delta_v_mps, strict=True)
    ]
    out = Path(path)
    out.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return out
