"""
e2m2e - Earth to Moon, Moon to Earth Transfer Orbit Design Library

一个用于设计和分析地月空间转移轨道的Python库，专注于圆型限制性三体问题（CR3BP）中的轨道动力学。

主要功能：
1. 地月系统三体动力学建模
2. 平动点轨道（Halo, Lyapunov等）设计
3. 微分修正算法
4. 轨道延拓算法
5. 转移轨道设计
6. 可视化工具

作者: 天疆说
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "天疆说"
__email__ = "ouyangjiahong22@nudt.edu.cn"

# 导入核心模块
from .core.system import CR3BP_System, LibrationPoint
from .core.dynamics import CR3BP_Dynamics
from .core.orbit import Orbit
from .core.coordinate import CoordinateTransformation

# 导入算法模块
from .algorithms.differential_correction import DifferentialCorrection
from .algorithms.continuation import Continuation
from .algorithms.stability import StabilityAnalysis

# 导入可视化模块
from .visualization.plotting import OrbitVisualizer

# 导入转移轨道模块
from .transfer.earth_moon import EarthMoonTransfer
from .transfer.moon_earth import MoonEarthTransfer
from .transfer.inter_orbit import InterOrbitTransfer

# 定义公共API
__all__ = [
    # 核心模块
    "CR3BP_System",
    "LibrationPoint",
    "CR3BP_Dynamics",
    "Orbit",
    "CoordinateTransformation",
    # 算法模块
    "DifferentialCorrection",
    "Continuation",
    "StabilityAnalysis",
    # 可视化模块
    "OrbitVisualizer",
    # 转移轨道模块
    "EarthMoonTransfer",
    "MoonEarthTransfer",
    "InterOrbitTransfer",
]
