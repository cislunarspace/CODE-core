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
版本: 2.0.0
"""

__version__ = "3.1.3"
__author__ = "天疆说"
__email__ = "ouyangjiahong22@nudt.edu.cn"

# 导入子包
from . import core
from . import algorithms
from . import visualization
from . import transfer

__all__ = [
    "core",
    "algorithms",
    "visualization",
    "transfer",
]
