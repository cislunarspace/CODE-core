"""
转移轨道设计基础模块

提供转移轨道设计的统一抽象基类和标准化的配置、结果数据结构。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.system import CR3BP_System
    from ..core.dynamics import CR3BP_Dynamics
    from ..core.orbit import Orbit


class TransferStrategy(Enum):
    GRID_SEARCH = "grid_search"
    NLP_OPTIMIZATION = "nlp_optimization"
    MANIFOLD = "manifold"
    DIRECT = "direct"


class TransferType(Enum):
    PLANAR = "planar"
    THREE_DIMENSIONAL = "3d"
    DIRECT = "direct"
    LGA = "lga"
    EXTERNAL = "external"


class BaseTransfer:
    """转移轨道设计基类

    定义转移轨道设计的统一接口。
    子类应实现具体的搜索和优化算法。

    使用方式:
        1. 创建实例: transfer = DROTransfer(system, dynamics)
        2. 设置轨道: transfer.set_departure_orbit(...) / set_arrival_orbit(...)
        3. 配置: transfer.configure_search(...) / configure_optimization(...)
        4. 执行: transfer.search() / transfer.optimize(...)
        5. 获取结果: transfer.results / transfer.optimized_result
    """

    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
    ):
        """初始化转移设计器

        参数:
            system: CR3BP系统对象
            dynamics: CR3BP动力学对象
            name: 转移名称
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu

        self._departure_orbit: Optional[Orbit] = None
        self._arrival_orbit: Optional[Orbit] = None

        self._search_results: Optional[List[Dict[str, Any]]] = None
        self._optimized_result: Any = None

    @property
    def departure_orbit(self) -> Optional[Orbit]:
        """出发点轨道"""
        return self._departure_orbit

    @property
    def arrival_orbit(self) -> Optional[Orbit]:
        """目标轨道"""
        return self._arrival_orbit

    @property
    def search_results(self) -> Optional[List[Dict[str, Any]]]:
        """搜索结果"""
        return self._search_results

    @property
    def optimized_result(self) -> Any:
        """优化结果"""
        return self._optimized_result

    def set_departure_orbit(self, orbit: Orbit) -> "BaseTransfer":
        """设置出发点轨道

        参数:
            orbit: 出发点轨道

        返回:
            self: 支持链式调用
        """
        self._departure_orbit = orbit
        return self

    def set_arrival_orbit(self, orbit: Orbit) -> "BaseTransfer":
        """设置目标轨道

        参数:
            orbit: 目标轨道

        返回:
            self: 支持链式调用
        """
        self._arrival_orbit = orbit
        return self

    def search(self, **kwargs) -> List[Dict[str, Any]]:
        """执行网格搜索

        子类应重写此方法实现具体搜索算法。

        参数:
            **kwargs: 搜索参数

        返回:
            搜索结果列表
        """
        raise NotImplementedError("Subclass must implement search()")

    def optimize(self, initial_guess: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行优化

        子类应重写此方法实现具体优化算法。

        参数:
            initial_guess: 优化初始猜测（通常来自搜索结果）

        返回:
            优化结果
        """
        raise NotImplementedError("Subclass must implement optimize()")

    def _is_feasible(self, result: Dict[str, Any]) -> bool:
        """判断搜索结果是否为可行候选解"""
        has_approach = (
            result.get("intersection_found", False)
            or result.get("min_distance", float("inf")) < 0.05
            or result.get("local_minimum_found", False)
        )
        no_collision = not result.get("collision_found", False)
        return has_approach and no_collision

    def get_feasible_results(self) -> List[Dict[str, Any]]:
        """获取所有可行搜索结果"""
        if self._search_results is None:
            return []
        return [r for r in self._search_results if self._is_feasible(r)]

    def info(self) -> None:
        """输出转移设计器信息"""
        print("=" * 60)
        print(f"{self.name}")
        print("=" * 60)
        print(f"System: {self.system}")
        print(f"μ: {self.mu:.6f}")
        print(f"Departure orbit: {self._departure_orbit}")
        print(f"Arrival orbit: {self._arrival_orbit}")
        if self._search_results:
            feasible = self.get_feasible_results()
            print(f"Search results: {len(self._search_results)} total, {len(feasible)} feasible")
        print("=" * 60)

    def __str__(self) -> str:
        return f"{self.name}(mu={self.mu})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system={self.system}, dynamics={self.dynamics})"
