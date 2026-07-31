"""ForceModel 配置驱动 shim（ADR 0011 迁移）。

实现已迁至 ``e2m2e.algorithm.forces``，旧路径保持可用。
"""

from __future__ import annotations

from e2m2e.algorithm.forces.exceptions import NotSerializableError
from e2m2e.algorithm.forces.force_config import (
    build_force,
    dump_force_config,
    load_force_config,
    serialize_force,
)

__all__ = [
    "build_force",
    "serialize_force",
    "dump_force_config",
    "load_force_config",
    "NotSerializableError",
]
