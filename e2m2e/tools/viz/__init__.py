"""可视化：PlotConfig/OrbitVisualizer/FamilyPlotter 等。

ADR 0011 迁移（源：``visualization/``），可选依赖 matplotlib（``[viz]``
extra）。MCP 无头部署不依赖可视化；核心库不 import tools/。
"""

from __future__ import annotations

from .base import OrbitVisualizer, ProjectionPlane
from .config import BODY_ICON_PATH_ENV, BODY_ICON_SCALE_ENV, PlotConfig
from .family import FamilyPlotter
from .transfer import TransferPlotter

__all__ = [
    "OrbitVisualizer",
    "ProjectionPlane",
    "PlotConfig",
    "BODY_ICON_PATH_ENV",
    "BODY_ICON_SCALE_ENV",
    "FamilyPlotter",
    "TransferPlotter",
]
