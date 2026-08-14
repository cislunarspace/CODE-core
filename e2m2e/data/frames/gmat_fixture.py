"""GMAT 兼容 ITRF 所需的裁剪 EOP/闰秒数据资源。

资源随 ``e2m2e.data.frames`` 发布，既供数据解析测试使用，也供
``algorithm.coordinate`` 的显式 GMAT-compatible 坐标轴提供默认输入。
这些文件是 IERS 公布物理数据的输入，不是其他软件的输出对照标准。
"""

from __future__ import annotations

import os
from pathlib import Path

from .eop import CoordinateDataError

_COMMITTED_GMAT_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "gmat"


def committed_gmat_fixture_dir() -> Path:
    """返回随数据层发布的 GMAT 裁剪数据目录。"""
    if not _COMMITTED_GMAT_FIXTURE_DIR.is_dir():
        raise CoordinateDataError(
            f"Committed GMAT fixture directory not found: {_COMMITTED_GMAT_FIXTURE_DIR}"
        )
    return _COMMITTED_GMAT_FIXTURE_DIR


def gmat_data_dir() -> Path | None:
    """返回 ``GMAT_DATA_DIR`` 指向的完整 GMAT data 目录。"""
    raw_path = os.environ.get("GMAT_DATA_DIR")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise CoordinateDataError(f"GMAT_DATA_DIR does not exist: {path}")
    return path


def gmat_fixture_path(name: str) -> Path:
    """返回随数据层发布的 GMAT 裁剪数据文件路径。"""
    path = committed_gmat_fixture_dir() / name
    if not path.is_file():
        raise CoordinateDataError(f"Committed GMAT fixture file not found: {path}")
    return path
