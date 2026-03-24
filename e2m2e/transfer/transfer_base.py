"""
转移轨道设计基础模块

提供转移轨道设计的统一抽象基类和标准化的配置、结果数据结构。
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.system import CR3BP_System
    from ..core.dynamics import CR3BP_Dynamics
    from ..core.orbit import Orbit


class TransferStrategy(Enum):
    """转移策略枚举"""

    GRID_SEARCH = "grid_search"  # 网格搜索
    NLP_OPTIMIZATION = "nlp_optimization"  # NLP优化
    MANIFOLD = "manifold"  # 流形转移
    DIRECT = "direct"  # 直接转移


class TransferType(Enum):
    """转移类型枚举"""

    PLANAR = "planar"  # 平面转移
    THREE_DIMENSIONAL = "3d"  # 三维转移
    DIRECT = "direct"  # 直接转移
    LGA = "lga"  # 月球引力辅助转移
    EXTERNAL = "external"  # 外部转移


@dataclass
class TransferConfig:
    """转移配置基类

    包含转移设计所需的通用配置参数。
    子类应继承并扩展此配置。

    属性:
        max_transfer_time: 最大转移时间 (CR3BP无量纲时间)
        collision_earth_radius: 地球碰撞检测半径 (无量纲)
        collision_moon_radius: 月球碰撞检测半径 (无量纲)
        integration_dt: 积分时间步长
    """

    max_transfer_time: float = 15.0
    collision_earth_radius: float = 0.02
    collision_moon_radius: float = 0.005
    integration_dt: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class SearchConfig(TransferConfig):
    """网格搜索配置

    专门用于网格搜索阶段的配置参数。

    属性:
        alpha_min: α(切向速度比)最小值
        alpha_max: α(切向速度比)最大值
        n_alpha: α方向网格点数
        n_departure: 出发点采样数量
        intersection_threshold: 相交检测阈值
        min_distance_threshold: 最小距离阈值
    """

    alpha_min: float = 0.5
    alpha_max: float = 2.5
    n_alpha: int = 101
    n_departure: int = 200
    intersection_threshold: float = 0.001
    min_distance_threshold: float = 0.05

    @property
    def alpha_grid(self) -> np.ndarray:
        """α网格点"""
        return np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict["alpha_grid"] = self.alpha_grid.tolist()
        return base_dict


@dataclass
class OptimizationConfig(TransferConfig):
    """优化配置

    专门用于优化阶段的配置参数。

    属性:
        alpha_range: α搜索范围
        transfer_time_range: 转移时间范围
        t_ins_range: 插入时间范围
        velocity_angle_tolerance: 速度平行性容差
    """

    alpha_range: Tuple[float, float] = (0.5, 2.5)
    transfer_time_range: Tuple[float, float] = (1.0, 30.0)
    t_ins_range: Tuple[float, float] = (0.0, 10.0)
    velocity_angle_tolerance: float = 1e-6


@dataclass
class TransferResult:
    """转移结果基类

    存储转移设计的结果数据。
    子类应继承并扩展此结果类。

    属性:
        success: 是否成功
        message: 结果消息
        departure_orbit_name: 出发点轨道名称
        arrival_orbit_name: 目标轨道名称
        transfer_trajectory: 转移轨迹状态序列 [n_steps, 6]
        transfer_times: 转移轨迹时间序列 [n_steps]
        transfer_time: 转移时间
        total_delta_v: 总速度增量
    """

    success: bool = False
    message: str = ""
    departure_orbit_name: str = ""
    arrival_orbit_name: str = ""
    transfer_trajectory: Optional[np.ndarray] = None
    transfer_times: Optional[np.ndarray] = None
    transfer_time: float = 0.0
    total_delta_v: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        if self.transfer_trajectory is not None:
            result["transfer_trajectory_shape"] = self.transfer_trajectory.shape
        return result


@dataclass
class SearchResult(TransferResult):
    """网格搜索结果

    存储网格搜索阶段的结果。

    属性:
        departure_state: 出发点状态 [6]
        departure_time: 出发点时间
        alpha: 切向速度比
        min_distance: 到目标轨道最小距离
        min_distance_idx: 最小距离对应索引
        intersection_found: 是否与目标轨道相交
        intersection_point: 相交点状态
        local_minimum_found: 是否找到局部最小
        collision_found: 是否发生碰撞
        collision_body: 碰撞天体 ('earth' or 'moon')
        status: 状态标识
    """

    departure_state: Optional[np.ndarray] = None
    departure_time: float = 0.0
    alpha: float = 0.0
    min_distance: float = np.inf
    min_distance_idx: int = -1
    intersection_found: bool = False
    intersection_point: Optional[np.ndarray] = None
    intersection_idx: int = -1
    local_minimum_found: bool = False
    local_minimum_distance: float = np.inf
    local_minimum_idx: int = -1
    collision_found: bool = False
    collision_body: Optional[str] = None
    collision_idx: int = -1
    status: str = "pending"

    @property
    def is_feasible(self) -> bool:
        """判断是否为可行候选解"""
        has_approach = (
            self.intersection_found
            or self.min_distance < 0.05
            or self.local_minimum_found
        )
        no_collision = not self.collision_found
        return has_approach and no_collision

    @property
    def dv_departure(self) -> float:
        """计算departure impulse"""
        if self.departure_state is None or self.transfer_trajectory is None:
            return 0.0
        dv = self.transfer_trajectory[0, 3:] - self.departure_state[3:]
        return np.linalg.norm(dv)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict["is_feasible"] = self.is_feasible
        base_dict["dv_departure"] = self.dv_departure
        return base_dict


@dataclass
class OptimizationResult(TransferResult):
    """优化结果

    存储优化阶段的结果。

    属性:
        alpha: 优化后的切向速度比
        transfer_time: 优化后的转移时间
        t_ins: 优化后的插入时间
        delta_v1: 出发脉冲
        delta_v2: 插入脉冲
        insertion_state: 插入点状态
        departure_state: 出发点状态
        transfer_type: 转移类型
        constraints_violation: 约束违反量
    """

    alpha: float = 0.0
    transfer_time: float = 0.0
    t_ins: float = 0.0
    delta_v1: float = 0.0
    delta_v2: float = 0.0
    insertion_state: Optional[np.ndarray] = None
    departure_state: Optional[np.ndarray] = None
    transfer_type: TransferType = TransferType.DIRECT
    constraints_violation: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict["transfer_type"] = self.transfer_type.value
        return base_dict


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
        name: str = "BaseTransfer",
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
        self.name = name

        self._departure_orbit: Optional[Orbit] = None
        self._arrival_orbit: Optional[Orbit] = None
        self._search_config: Optional[SearchConfig] = None
        self._optimization_config: Optional[OptimizationConfig] = None
        self._search_results: Optional[List[SearchResult]] = None
        self._optimized_result: Optional[OptimizationResult] = None

    @property
    def departure_orbit(self) -> Optional[Orbit]:
        """出发点轨道"""
        return self._departure_orbit

    @property
    def arrival_orbit(self) -> Optional[Orbit]:
        """目标轨道"""
        return self._arrival_orbit

    @property
    def search_config(self) -> Optional[SearchConfig]:
        """搜索配置"""
        return self._search_config

    @property
    def optimization_config(self) -> Optional[OptimizationConfig]:
        """优化配置"""
        return self._optimization_config

    @property
    def search_results(self) -> Optional[List[SearchResult]]:
        """搜索结果"""
        return self._search_results

    @property
    def optimized_result(self) -> Optional[OptimizationResult]:
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

    def configure_search(self, **kwargs) -> "BaseTransfer":
        """配置搜索参数

        参数:
            **kwargs: SearchConfig 字段

        返回:
            self: 支持链式调用
        """
        if self._search_config is None:
            self._search_config = SearchConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(self._search_config, key):
                    setattr(self._search_config, key, value)
        return self

    def configure_optimization(self, **kwargs) -> "BaseTransfer":
        """配置优化参数

        参数:
            **kwargs: OptimizationConfig 字段

        返回:
            self: 支持链式调用
        """
        if self._optimization_config is None:
            self._optimization_config = OptimizationConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(self._optimization_config, key):
                    setattr(self._optimization_config, key, value)
        return self

    def search(self, **kwargs) -> List[SearchResult]:
        """执行网格搜索

        子类应重写此方法实现具体搜索算法。

        参数:
            **kwargs: 搜索参数

        返回:
            搜索结果列表
        """
        raise NotImplementedError("Subclass must implement search()")

    def optimize(self, initial_guess: Optional[SearchResult] = None) -> OptimizationResult:
        """执行优化

        子类应重写此方法实现具体优化算法。

        参数:
            initial_guess: 优化初始猜测（通常来自搜索结果）

        返回:
            优化结果
        """
        raise NotImplementedError("Subclass must implement optimize()")

    def get_feasible_results(self) -> List[SearchResult]:
        """获取所有可行搜索结果"""
        if self._search_results is None:
            return []
        return [r for r in self._search_results if r.is_feasible]

    def info(self) -> None:
        """输出转移设计器信息"""
        print("=" * 60)
        print(f"{self.name}")
        print("=" * 60)
        print(f"System: {self.system}")
        print(f"μ: {self.mu:.6f}")
        print(f"Departure orbit: {self._departure_orbit}")
        print(f"Arrival orbit: {self._arrival_orbit}")
        if self._search_config:
            print(f"Search config: {self._search_config}")
        if self._optimization_config:
            print(f"Optimization config: {self._optimization_config}")
        if self._search_results:
            feasible = self.get_feasible_results()
            print(f"Search results: {len(self._search_results)} total, {len(feasible)} feasible")
        print("=" * 60)

    def __str__(self) -> str:
        return f"{self.name}(mu={self.mu})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system={self.system}, dynamics={self.dynamics})"
