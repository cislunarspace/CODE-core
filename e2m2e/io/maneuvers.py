"""DFH MANEUVERS.TXT 机动序列文件解析与写入（临时脚本，ADR 0011）。

``ManeuverTable`` 通用容器已迁至 ``e2m2e.data.types.maneuver``；本模块保留
DFH 专属格式的 parse/read/write，作为开发期临时脚本（io/ 最终不进 e2m2e）。

文件行格式：每个非空行包含两个空白分隔的实数（MJD(TDB) 与机动脉冲 m/s）。
照 MATLAB ``parse_maneuvers.m``：按数值词元提取，不足两个数值的行静默跳过。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from e2m2e.data.types.maneuver import ManeuverTable

__all__ = ["ManeuverTable", "parse_maneuvers", "read_maneuvers", "write_maneuvers"]

_NUM_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")


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
