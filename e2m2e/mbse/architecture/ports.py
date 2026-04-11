"""Protocol 接口定义

定义 e2m2e 系统中所有形式化的接口契约（对应 SysML BDD/IBD 中的端口）。

每个 Protocol 代表系统中一个关键抽象，具体实现类须满足这些接口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from mbse.data.core_models import PropagationResult


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
        """计算状态导数 d(state)/dt

        Args:
            t: 时间（无量纲）
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        ...

    def equations_with_stm(
        self, t: float, augmented_state: np.ndarray
    ) -> np.ndarray:
        """计算增广状态导数（含 STM）

        Args:
            t: 时间（无量纲）
            augmented_state: 42 维增广状态 [state(6), STM_flat(36)]

        Returns:
            42 维增广状态导数
        """
        ...


@runtime_checkable
class Propagator(Protocol):
    """数值传播器接口

    能够将状态向量沿时间前向/后向积分。
    对应 SysML BDD 中的 <<propagator>> 块。

    关键契约：
    - propagate() 返回的 states 形状必须为 (n_points, 6)
    - time 数组必须单调递增
    """

    def propagate(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> dict:
        """数值传播状态向量

        Args:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            t_span: 积分时间范围 (t_start, t_end)
            t_eval: 输出时间点（None 则使用积分器内部步长）
            with_stm: 是否同时计算状态转移矩阵
            with_jacobi: 是否同时计算 Jacobi 常数

        Returns:
            包含 time, states, 可选 stm/jacobi 的字典。
            states 形状为 (n_points, 6)。
        """
        ...


@runtime_checkable
class OrbitContainer(Protocol):
    """轨道数据容器接口

    存储一条轨道的状态序列和时间序列。
    对应 SysML BDD 中的 <<orbit>> 块。
    """

    @property
    def states(self) -> np.ndarray:
        """状态序列，形状 (n, 6)，每行为 [x, y, z, vx, vy, vz]"""
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
    对应 SysML BDD 中的 <<strategy>> 块，采用策略模式。
    """

    def configure(self, **kwargs) -> CorrectorStrategy:
        """配置修正策略参数

        Returns:
            self（支持链式调用）
        """
        ...

    def compute_error(
        self,
        orbit: OrbitContainer,
        dynamics: Propagator,
    ) -> np.ndarray:
        """计算约束误差向量

        Args:
            orbit: 待修正轨道
            dynamics: 动力学模型

        Returns:
            误差向量
        """
        ...

    def get_free_variable_indices(self) -> list[int]:
        """获取自由变量索引列表"""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """NLP 优化器接口

    对应 SysML BDD 中的 <<optimizer>> 块。
    """

    def optimize(self, initial_guess: np.ndarray, **kwargs) -> dict:
        """执行优化

        Args:
            initial_guess: 初始决策变量

        Returns:
            优化结果字典
        """
        ...


@runtime_checkable
class Visualizer(Protocol):
    """可视化器接口

    对应 SysML BDD 中的 <<visualizer>> 块。
    """

    def plot(self, data: object, config: object, **kwargs) -> object:
        """绘制图表

        Args:
            data: 待绘制的数据
            config: 绘图配置

        Returns:
            matplotlib Figure 对象
        """
        ...
