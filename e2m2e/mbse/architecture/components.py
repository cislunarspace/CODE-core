"""组件模型

将具体实现类关联为 SysML BDD 中的组件层次。

每个 Component 记录其源代码位置和依赖关系。
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARCHITECTURE_LAYERS = ("data", "numerical", "algorithm", "api", "tools", "mbse")


@dataclass(frozen=True)
class Component:
    """系统组件定义

    对应 SysML BDD 中的一个块（Block），记录其接口契约和依赖。

    Attributes:
        name: 组件名称（如 "CR3BP_Dynamics"）
        module_path: 源代码模块路径（如 "e2m2e.algorithm.dynamics"）
        protocols: 预留字段（ADR 0001 后为空列表）
        dependencies: 该组件依赖的其他组件名称
        layer: 所属架构层（data/numerical/algorithm/api/tools/mbse）
        description: 组件功能简述
    """

    name: str
    module_path: str
    protocols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    layer: str = "mbse"
    description: str = ""

    def __post_init__(self):
        """校验架构层合法性"""
        if self.layer not in ARCHITECTURE_LAYERS:
            raise ValueError(f"无效的架构层: {self.layer}，应为 {ARCHITECTURE_LAYERS}")


class ComponentRegistry:
    """组件注册表。

    调用方持有注册表生命周期，避免模型装配和测试共享可变全局状态。
    """

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}

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
        """按接口名称筛选组件（ADR 0001 后 protocols 均为空，始终返回 []）"""
        return [c for c in self._components.values() if protocol_name in c.protocols]

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
        """返回已注册组件数量"""
        return len(self._components)

    def __contains__(self, name: str) -> bool:
        """检查组件是否已注册"""
        return name in self._components

    def __iter__(self):
        """遍历所有已注册组件"""
        return iter(self._components.values())
