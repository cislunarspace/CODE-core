"""GMAT 裁剪 fixture 发现工具。"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_GMAT_FIXTURE_DIR = _REPO_ROOT / "tests" / "core" / "coordinate" / "fixtures" / "gmat"


class CoordinateDataError(RuntimeError):
    """坐标数据缺失、越界或格式错误。"""


def committed_gmat_fixture_dir() -> Path:
    """返回仓库内提交的 GMAT 裁剪 fixture 目录。"""
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
    """返回提交的 GMAT 裁剪 fixture 文件路径。"""
    path = committed_gmat_fixture_dir() / name
    if not path.is_file():
        raise CoordinateDataError(f"Committed GMAT fixture file not found: {path}")
    return path
