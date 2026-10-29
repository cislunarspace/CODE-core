"""相对运动（交会对接，主题 3）。

领域算法与 station_keeping/family 同类（ADR 0011 迁移，源：``proximity/``）：
relative_dynamics（RLM/Encke/LVLH）、floquet_modes（Floquet 模态基础解集
与有界性判据）、phasing（调相）、safety（保持点安全）。

算法层能力，不设任务级工具。进程内调用方直接 import 使用，CLI、MCP、
GUI sidecar 不设对应工具（#835，见 ``.out-of-scope/relative-motion-tool-face.md``）。
"""

from __future__ import annotations

from .floquet_modes import (
    FloquetModeError,
    FloquetModeSet,
    FloquetPair,
    FloquetPairKind,
    floquet_modes,
)
from .phasing import PhasingManeuver, PhasingSolution, phasing_search
from .relative_dynamics import RelativeDynamics, RelativeState, TargetOrbit
from .safety import (
    KeepingPoint,
    SafetyRegion,
    SafetyReport,
    check_passive_safety,
    max_collision_probability,
)

__all__ = [
    "FloquetModeError",
    "FloquetModeSet",
    "FloquetPair",
    "FloquetPairKind",
    "RelativeDynamics",
    "RelativeState",
    "TargetOrbit",
    "KeepingPoint",
    "PhasingManeuver",
    "PhasingSolution",
    "SafetyRegion",
    "SafetyReport",
    "check_passive_safety",
    "floquet_modes",
    "max_collision_probability",
    "phasing_search",
]
