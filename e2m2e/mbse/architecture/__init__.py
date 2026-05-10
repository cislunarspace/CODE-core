"""架构模型子包

提供组件模型（ADR 0001: Protocol 接缝已移除，多态通过 Dynamics 基类实现）。
"""

from .components import Component, ComponentRegistry

__all__ = [
    "Component",
    "ComponentRegistry",
]
