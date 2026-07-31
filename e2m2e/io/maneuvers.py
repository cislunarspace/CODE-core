"""DFH MANEUVERS.TXT 机动序列文件解析与写入。

每个非空行包含两个空白分隔的实数：MJD 形式的历元（TDB）与机动脉冲
大小（m/s）。照 MATLAB ``parse_maneuvers.m``：按数值词元提取，不足两个
数值的行静默跳过。
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
    """DFH 风格脉冲大小：|v| ≥ 0.1 用定点 15 位小数，否则科学计数。

    与黄金样本中 Fortran 输出的数值写法一致（小量如 7.8e-2 用 E 记法），
    parse_maneuvers 的正则两种写法都能读。
    """
    if abs(v) < 0.1:
        return f"{v:>22.15E}"
    return f"{v:>22.15f}"


def write_maneuvers(table: ManeuverTable, path: str | Path) -> None:
    """把机动序列表写入 MANEUVERS.TXT。

    每行两个空白分隔数值：MJD（TDB，天）+ 脉冲大小（m/s）。
    """
    lines = [
        f"{mjd:>20.10f}{_format_delta_v(dv)}"
        for mjd, dv in zip(table.mjd_tdb, table.delta_v_mps, strict=True)
    ]
    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
