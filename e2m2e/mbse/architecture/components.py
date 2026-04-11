"""组件模型

将 Protocol 接口与具体实现类关联，形成 SysML BDD 中的组件层次。

每个 Component 记录其满足的 Protocol、源代码位置和依赖关系。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Component:
    """系统组件定义

    对应 SysML BDD 中的一个块（Block），记录其接口契约和依赖。

    Attributes:
        name: 组件名称（如 "CR3BP_Dynamics"）
        module_path: 源代码模块路径（如 "e2m2e.core.dynamics"）
        protocols: 该组件满足的 Protocol 接口列表
        dependencies: 该组件依赖的其他组件名称
        layer: 所属架构层（core/algorithms/transfer/visualization）
        description: 组件功能简述
    """

    name: str
    module_path: str
    protocols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    layer: str = "core"
    description: str = ""

    def __post_init__(self):
        valid_layers = {"core", "algorithms", "transfer", "visualization", "mbse"}
        if self.layer not in valid_layers:
            raise ValueError(f"无效的架构层: {self.layer}，应为 {valid_layers}")


class ComponentRegistry:
    """组件注册表

    集中管理所有组件定义，支持按层、按接口查询。
    """

    _instance: Optional[ComponentRegistry] = None
    _components: dict[str, Component]

    def __new__(cls) -> ComponentRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components = {}
        return cls._instance

    def register(self, component: Component) -> None:
        """注册一个组件"""
        if component.name in self._components:
            raise ValueError(f"组件 {component.name} 已存在")
        self._components[component.name] = component

    def register_many(self, components: list[Component]) -> None:
        """批量注册组件"""
        for comp in components:
            self.register(comp)

    def get(self, name: str) -> Component:
        """按名称获取组件"""
        if name not in self._components:
            raise KeyError(f"组件 {name} 未注册")
        return self._components[name]

    def all(self) -> list[Component]:
        """获取所有已注册组件"""
        return list(self._components.values())

    def by_layer(self, layer: str) -> list[Component]:
        """按架构层筛选组件"""
        return [c for c in self._components.values() if c.layer == layer]

    def by_protocol(self, protocol_name: str) -> list[Component]:
        """按满足的 Protocol 接口筛选组件"""
        return [
            c for c in self._components.values() if protocol_name in c.protocols
        ]

    def dependency_graph(self) -> dict[str, list[str]]:
        """构建组件依赖图

        Returns:
            {component_name: [dependency_names]}
        """
        return {name: list(comp.dependencies) for name, comp in self._components.items()}

    def clear(self) -> None:
        """清空所有已注册组件（仅用于测试）"""
        self._components.clear()

    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def __iter__(self):
        return iter(self._components.values())
