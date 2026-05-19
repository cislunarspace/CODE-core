"""e2m2e 可视化模块

提供轨道族、转移轨迹的绘图工具。
使用统一的 PlotConfig 配置类管理字体、颜色、尺寸等绘图参数。
稳定性分析请使用 e2m2e.algorithms.stability 模块。
"""

from .base import OrbitVisualizer, ProjectionPlane
from .config import BODY_ICON_SCALE_ENV, PlotConfig
from .family import FamilyPlotter
from .transfer import TransferPlotter

__all__ = [
    "BODY_ICON_SCALE_ENV",
    "PlotConfig",
    "OrbitVisualizer",
    "ProjectionPlane",
    "FamilyPlotter",
    "TransferPlotter",
]
