"""e2m2e 可视化模块

提供轨道族、转移轨迹、稳定性指标的绘图工具。
使用统一的 PlotConfig 配置类管理字体、颜色、尺寸等绘图参数。
"""

# 配置类、轨道可视化、轨道族绘图、转移轨迹绘图、稳定性计算可视化

from .base import OrbitVisualizer, ProjectionPlane
from .config import PlotConfig
from .family import FamilyPlotter
from .stability import compute_stability_for_family
from .transfer import TransferPlotter

__all__ = [
    "PlotConfig",
    "OrbitVisualizer",
    "ProjectionPlane",
    "FamilyPlotter",
    "TransferPlotter",
    "compute_stability_for_family",
]
