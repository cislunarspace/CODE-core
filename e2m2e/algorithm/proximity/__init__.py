"""相对运动（交会对接，主题 3）。

领域算法与 station_keeping/family 同类（ADR 0011 迁移，源：``proximity/``）：
relative_dynamics（RLM/Encke/LVLH）、phasing（调相）、safety（保持点安全）。
属二档三档扩展位（relative_motion 标二档、safety 标三档，ADR 0014）。
"""

from __future__ import annotations

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
    "RelativeDynamics",
    "RelativeState",
    "TargetOrbit",
    "KeepingPoint",
    "PhasingManeuver",
    "PhasingSolution",
    "phasing_search",
    "SafetyRegion",
    "SafetyReport",
    "check_passive_safety",
    "max_collision_probability",
]
