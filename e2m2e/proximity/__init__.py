"""e2m2e 交会对接与相对运动模块（主题 3）。

提供 CR3BP 相对运动动力学（RLM 线性化）、目标轨道包装、相对状态数据结构。
"""

from .relative_dynamics import RelativeDynamics, RelativeState, TargetOrbit

__all__ = [
    "RelativeDynamics",
    "RelativeState",
    "TargetOrbit",
]
