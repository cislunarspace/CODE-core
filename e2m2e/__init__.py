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
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("e2m2e")
except PackageNotFoundError:
    try:
        from pathlib import Path

        import tomllib

        _pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if _pyproject.exists():
            with open(_pyproject, "rb") as _f:
                __version__ = tomllib.load(_f)["project"]["version"]
        else:
            __version__ = "0.0.0"
    except Exception:
        __version__ = "0.0.0"

__author__ = "天疆说"
__email__ = "ouyangjiahong22@nudt.edu.cn"

# 导入子包
from . import algorithms, core, dynamics, mbse, transfer, visualization

__all__ = [
    "core",
    "dynamics",
    "algorithms",
    "visualization",
    "transfer",
    "mbse",
]
