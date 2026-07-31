"""GMAT fixture 发现工具 shim（ADR 0011 迁移）。

实现已迁至数据层：``e2m2e.data.frames.gmat_fixture``（路径发现）与
``e2m2e.data.frames.eop``（``CoordinateDataError``），旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.data.frames.eop import CoordinateDataError
from e2m2e.data.frames.gmat_fixture import (
    committed_gmat_fixture_dir,
    gmat_data_dir,
    gmat_fixture_path,
)

__all__ = [
    "CoordinateDataError",
    "committed_gmat_fixture_dir",
    "gmat_data_dir",
    "gmat_fixture_path",
]
