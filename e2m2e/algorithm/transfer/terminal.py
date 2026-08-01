"""终端条件模块

定义转移优化中出发/到达终端条件的抽象接口与具体实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ...data.types.orbit import Orbit
    from ..dynamics import CR3BP_Dynamics


class TerminalCondition(ABC):
    """转移终端条件抽象基类

    定义出发状态与到达状态的获取契约。优化器通过此接口
    与不同类型的终端（轨道、固定状态、事件触发等）交互。
    """

    @abstractmethod
    def get_initial_state(self) -> npt.NDArray[np.floating]:
        """返回出发状态向量 ``[x, y, z, vx, vy, vz]``"""
        ...

    @abstractmethod
    def get_arrival_state(
        self, t_ins: float, dynamics: CR3BP_Dynamics
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """返回到达时刻的位置与速度

        Args:
            t_ins: 到达时刻（与轨道时间坐标一致）
            dynamics: 动力学对象，用于状态传播

        Returns:
            (position, velocity) 其中 position 形状 ``(3,)``，velocity 形状 ``(3,)``
        """
        ...


class OrbitTerminal(TerminalCondition):
    """基于周期轨道的终端条件

    出发状态取轨道首点状态；到达状态通过动力学传播到指定相位获取。
    """

    def __init__(self, orbit: Orbit) -> None:
        """初始化

        Args:
            orbit: 周期轨道数据
        """
        self.orbit = orbit

    def get_initial_state(self) -> npt.NDArray[np.floating]:
        """返回轨道首点状态"""
        return np.array(self.orbit.states[0], copy=True)

    def get_arrival_state(
        self, t_ins: float, dynamics: CR3BP_Dynamics
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """通过动力学传播获取到达相位状态"""
        state = dynamics.propagate_orbit_state_at_time(self.orbit, float(t_ins))
        return state[:3], state[3:6]


class StateTerminal(TerminalCondition):
    """基于固定状态与时刻的终端条件

    出发与到达状态均为固定值，不依赖动力学传播。
    """

    def __init__(
        self,
        state: npt.ArrayLike,
        time: float,
    ) -> None:
        """初始化

        Args:
            state: 固定状态向量 ``[x, y, z, vx, vy, vz]``
            time: 固定时刻（仅用于记录，不影响状态提取）
        """
        self.state = np.array(state, dtype=float)
        self.time = float(time)

    def get_initial_state(self) -> npt.NDArray[np.floating]:
        """返回固定状态的副本"""
        return np.array(self.state, copy=True)

    def get_arrival_state(
        self, t_ins: float, dynamics: CR3BP_Dynamics
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """返回固定状态的位置与速度（忽略 t_ins）"""
        return self.state[:3], self.state[3:6]
