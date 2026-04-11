"""Protocol 接口定义

定义 e2m2e 系统中所有形式化的接口契约（对应 SysML BDD/IBD 中的端口）。

每个 Protocol 代表系统中一个关键抽象，具体实现类须满足这些接口。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SystemModel(Protocol):
    """系统物理模型接口

    提供天体系统的基本物理参数，如质量参数、平动点位置等。
    对应 SysML BDD 中的 <<system>> 块。
    """

    @property
    def mu(self) -> float:
        """系统质量参数 μ = m₂/(m₁+m₂)"""
        ...

    def get_jacobi_constant(self, state: np.ndarray) -> float:
        """计算给定状态的 Jacobi 常数

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            Jacobi 常数 C
        """
        ...


@runtime_checkable
class EOMProvider(Protocol):
    """运动方程提供者接口

    提供天体动力学运动方程（纯数学，不涉及数值积分）。
    对应 SysML IBD 中 Dynamics 块的 <<equation>> 端口。
    """

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """计算状态导数 d(state)/dt"""
        ...


@runtime_checkable
class Propagator(Protocol):
    """数值传播器接口

    关键契约：propagate() 返回的 states 形状必须为 (n_points, 6)。
    """

    def propagate(
        self,
        initial_state: np.ndarray,
        t_span: tuple,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """数值传播状态向量

        Returns:
            包含 'time', 'states'（形状 (n, 6)）的字典，
            可选 'stm'（形状 (n, 6, 6)）和 'jacobi'。
        """
        ...


@runtime_checkable
class OrbitContainer(Protocol):
    """轨道数据容器接口"""

    @property
    def states(self) -> np.ndarray:
        """状态序列，形状 (n, 6)"""
        ...

    @property
    def times(self) -> np.ndarray:
        """时间序列，形状 (n,)"""
        ...

    @property
    def period(self) -> float | None:
        """轨道周期（无量纲），未知时为 None"""
        ...


@runtime_checkable
class CorrectorStrategy(Protocol):
    """微分修正策略接口

    封装一种特定的微分修正配置（约束、自由变量、对称性条件）。
    策略函数返回 CorrectionConfig 不可变对象，由 DifferentialCorrection 使用。
    """

    setup_type: str
    symmetry_condition: str
    free_variable_indices: list[int]
    constraint_indices: list[int]
    target_conditions: dict[str, float]
    fixed_parameters: dict[str, float]


@runtime_checkable
class Optimizer(Protocol):
    """NLP 优化器接口"""

    def optimize(self, initial_guess: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """执行优化"""
        ...


@runtime_checkable
class Visualizer(Protocol):
    """可视化器接口"""

    def plot(self, data: object, config: object, **kwargs: Any) -> object:
        """绘制图表"""
        ...
