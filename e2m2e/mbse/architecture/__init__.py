"""架构模型子包

提供 Protocol 接口定义和组件模型。
"""

from .components import Component, ComponentRegistry
from .ports import (
    CorrectorStrategy,
    EOMProvider,
    OrbitContainer,
    Optimizer,
    Propagator,
    SystemModel,
    Visualizer,
)

__all__ = [
    "CorrectorStrategy",
    "EOMProvider",
    "OrbitContainer",
    "Optimizer",
    "Propagator",
    "SystemModel",
    "Visualizer",
    "Component",
    "ComponentRegistry",
]
