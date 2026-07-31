"""GMAT 裁剪 fixture 发现工具（含 CoordinateDataError shim）。

``CoordinateDataError`` 已迁至 ``e2m2e.data.frames.eop``（ADR 0011 迁移），
此处 re-export 保持旧路径可用；fixture 路径发现工具是开发期测试辅助，
暂留本模块。
"""

from __future__ import annotations

import os
from pathlib import Path

from e2m2e.data.frames.eop import CoordinateDataError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_GMAT_FIXTURE_DIR = _REPO_ROOT / "tests" / "core" / "coordinate" / "fixtures" / "gmat"


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
