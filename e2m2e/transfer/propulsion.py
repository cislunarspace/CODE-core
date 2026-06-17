"""推进模型抽象与脉冲推进实现。

提供 ``PropulsionModel`` 抽象基类及 ``ImpulsivePropulsion`` 具体实现，
用于计算转移轨道的出发注入速度与代价。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .cost import TransferCost, compute_transfer_cost


class PropulsionModel(ABC):
    """推进模型抽象基类。

    定义转移轨道出发注入速度计算与代价评估的接口。
    """

    @abstractmethod
    def compute_departure_velocity(self, state: np.ndarray, **params: Any) -> np.ndarray:
        """根据出发点状态与参数计算出发注入速度。

        Args:
            state: 出发点六维状态 ``[x, y, z, vx, vy, vz]``。
            **params: 模型特定参数（如 ``alpha``、``beta`` 等）。

        Returns:
            注入速度向量 ``[vx, vy, vz]``。
        """
        ...

    @abstractmethod
    def compute_cost(
        self,
        departure_state: np.ndarray,
        initial_velocity: np.ndarray,
        final_velocity: np.ndarray,
        insertion_velocity: np.ndarray,
    ) -> TransferCost:
        """计算转移代价。

        Args:
            departure_state: 出发点六维状态 ``[x, y, z, vx, vy, vz]``。
            initial_velocity: 出发注入速度（调整后） ``[vx, vy, vz]``。
            final_velocity: 转移轨迹末端速度 ``[vx, vy, vz]``。
            insertion_velocity: 目标轨道插入点速度 ``[vx, vy, vz]``。

        Returns:
            ``TransferCost`` 含 ``dv1``、``dv2`` 及 ``total``。
        """
        ...


class ImpulsivePropulsion(PropulsionModel):
    """脉冲推进模型。

    将出发速度分解为切向与法向分量：
    ``v = alpha * |v| * t_hat + beta * |v| * n_hat``，
    其中 ``t_hat`` 为原始速度方向，``n_hat`` 为轨道面法向。

    Attributes:
        normal: 轨道面法向量，默认 ``[0, 0, 1]`` （z轴）。
    """

    def __init__(self, normal: np.ndarray | None = None):
        """初始化脉冲推进模型。

        Args:
            normal: 轨道面法向量；``None`` 时使用 ``[0.0, 0.0, 1.0]``。
        """
        if normal is None:
            self.normal = np.array([0.0, 0.0, 1.0])
        else:
            self.normal = np.asarray(normal, dtype=float)

    def compute_departure_velocity(
        self, state: np.ndarray, alpha: float = 1.0, beta: float = 0.0, **_: Any
    ) -> np.ndarray:
        """根据 ``alpha`` 和 ``beta`` 计算出发注入速度。

        速度分解为切向和法向分量：
        ``v = alpha * |v| * t_hat + beta * |v| * n_hat``，
        其中 ``t_hat`` 为原始速度方向，``n_hat`` 为轨道面法向。

        Args:
            state: 出发点状态 ``[x, y, z, vx, vy, vz]``。
            alpha: 切向速度比（缩放切向分量）。
            beta: 法向速度比（缩放法向分量），默认 ``0.0`` （纯切向）。
            **_: 忽略其他关键字参数。

        Returns:
            注入速度向量 ``[vx, vy, vz]``。
        """
        vel = state[3:]

        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-10:
            warnings.warn("出发点速度接近零", stacklevel=2)
            return vel.copy()

        tangential = vel / v_mag

        normal_dir = np.cross(tangential, self.normal)
        norm_nd = np.linalg.norm(normal_dir)
        normal_dir = (
            np.array([1.0, 0.0, 0.0]) if norm_nd < 1e-10 else normal_dir / norm_nd
        )

        v_injection = alpha * v_mag * tangential + beta * v_mag * normal_dir

        return v_injection

    def compute_cost(
        self,
        departure_state: np.ndarray,
        initial_velocity: np.ndarray,
        final_velocity: np.ndarray,
        insertion_velocity: np.ndarray,
    ) -> TransferCost:
        """计算转移代价。

        委托给 ``e2m2e.transfer.cost.compute_transfer_cost``。

        Args:
            departure_state: 出发点六维状态 ``[x, y, z, vx, vy, vz]``。
            initial_velocity: 出发注入速度（调整后） ``[vx, vy, vz]``。
            final_velocity: 转移轨迹末端速度 ``[vx, vy, vz]``。
            insertion_velocity: 目标轨道插入点速度 ``[vx, vy, vz]``。

        Returns:
            ``TransferCost`` 含 ``dv1``、``dv2`` 及 ``total``。
        """
        return compute_transfer_cost(
            departure_state, initial_velocity, final_velocity, insertion_velocity
        )
