"""DFH EPHEMERIDES_*.TXT 星历文件读写（临时脚本，ADR 0011）。

``EphemerisTable`` 通用容器已迁至 ``e2m2e.data.types.trajectory``；本模块
保留 DFH 专属格式的 parse/read/write，作为开发期临时脚本（io/ 最终不进
e2m2e）。

文件行格式（空白分隔，每行 10 列或更多）::

    YYYY-MM-DD-HH-MM-SS.SS  pX pY pZ  vX vY vZ  sX sY sZ

- 时间戳：UTC，年-月-日-时-分-秒以 ``-`` 分隔（秒含两位小数）；
- pX/pY/pZ：GCRS 位置（km）；vX/vY/vZ：GCRS 速度（m/s）；
  sX/sY/sZ：地月会合系位置（无量纲）。

解析陷阱：时间戳自身以 ``-`` 分隔，不能对整行 ``split('-')``——数值列的
负号会被拆碎。照 MATLAB ``parse_ephemeris.m`` 的做法：先按空白分列取首列
作时间戳再拆 ``-``，其余列按浮点解析；WSB 等文件时间戳后多于 9 列时只取
前 9 列。解析不了时间戳或数值不足 9 列的行静默跳过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from e2m2e.data.types.trajectory import EphemerisTable

__all__ = ["EphemerisTable", "parse_ephemeris", "read_ephemeris", "write_ephemeris"]


def parse_ephemeris(raw: str) -> EphemerisTable:
    """解析 EPHEMERIDES_*.TXT 文本，返回 :class:`EphemerisTable`。"""
    epoch_rows: list[list[float]] = []
    pos_rows: list[list[float]] = []
    vel_rows: list[list[float]] = []
    sync_rows: list[list[float]] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        ts = parts[0].split("-")
        if len(ts) < 6:
            continue
        try:
            epoch = [float(t) for t in ts[:6]]
            vals = [float(v) for v in parts[1:10]]
        except ValueError:
            continue
        epoch_rows.append(epoch)
        pos_rows.append(vals[0:3])
        vel_rows.append(vals[3:6])
        sync_rows.append(vals[6:9])

    epochs = np.array(epoch_rows, dtype=float).reshape(-1, 6)
    ints = epochs[:, :5].astype(int) if len(epochs) else np.empty((0, 5), dtype=int)
    return EphemerisTable(
        year=ints[:, 0] if len(epochs) else np.array([], dtype=int),
        month=ints[:, 1] if len(epochs) else np.array([], dtype=int),
        day=ints[:, 2] if len(epochs) else np.array([], dtype=int),
        hour=ints[:, 3] if len(epochs) else np.array([], dtype=int),
        minute=ints[:, 4] if len(epochs) else np.array([], dtype=int),
        second=epochs[:, 5] if len(epochs) else np.array([], dtype=float),
        position_km=np.array(pos_rows, dtype=float).reshape(-1, 3),
        velocity_mps=np.array(vel_rows, dtype=float).reshape(-1, 3),
        synodic_position=np.array(sync_rows, dtype=float).reshape(-1, 3),
        raw_text=raw,
    )


def read_ephemeris(path: str | Path) -> EphemerisTable:
    """从文件读入 EPHEMERIDES_*.TXT。"""
    raw = Path(path).read_text(encoding="utf-8")
    return parse_ephemeris(raw)


def write_ephemeris(table: EphemerisTable, path: str | Path) -> None:
    """按 DFH 格式写出星历文件（CRLF 行尾，UTF-8 无 BOM）。"""
    lines = []
    for k in range(len(table)):
        head = (
            f"{table.year[k]:04d}-{table.month[k]:02d}-{table.day[k]:02d}"
            f"-{table.hour[k]:02d}-{table.minute[k]:02d}-{table.second[k]:05.2f}"
        )
        nums = (
            list(table.position_km[k])
            + list(table.velocity_mps[k])
            + list(table.synodic_position[k])
        )
        lines.append(head + "".join(f"{v:25.12f}" for v in nums))
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
