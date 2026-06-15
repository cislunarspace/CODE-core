"""推力与机动模型。

Slice 6 实现：
- ``ImpulsiveBurn``：瞬时 Δv 机动事件（不继承 ``PhysicalModel``），
  由 :meth:`ForceModel.propagate_maneuvers` 在 epoch 处中断传播并施加。
- ``FiniteBurn``：连续推力加速度（``PhysicalModel``），后续测试加入。

演进路径（对应 GMAT R2026a 三层架构）：本模块合并了 GMAT 的
``FiniteBurn``（配置）与 ``FiniteThrust``（力模型），未引入 ``Thruster``
硬件层；VNB/LVLH burn 坐标系推迟到 Slice 12。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel


@dataclass(frozen=True)
class ImpulsiveBurn:
    """瞬时 Δv 机动事件。

    ``delta_v`` 在传播（惯性）坐标系内解释，由
    :meth:`e2m2e.core.forces.force_model.ForceModel.propagate_maneuvers`
    在 ``epoch`` 处施加 ``state[3:6] += delta_v``。

    VNB/LVLH burn frame 推迟到 Slice 12（届时加 ``frame`` 字段，转换走
    :meth:`CoordinateSystem.transform_vector`，对应 GMAT
    ``Burn::ConvertDeltaVToInertial`` 的 ``coincident=true`` 纯旋转）。

    Args:
        epoch: 施加时刻，SPICE et 秒，与 ``ForceModel.propagate`` 的
            ``t_span`` 一致。
        delta_v: 速度增量，传播坐标系，形状 ``(3,)``。
    """

    epoch: float
    delta_v: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        """以拷贝方式存储 delta_v，避免外部数组变更影响 burn。"""
        object.__setattr__(
            self, "delta_v", np.asarray(self.delta_v, dtype=float).copy()
        )


class FiniteBurn(PhysicalModel):
    """连续推力加速度力模型。

    合并了 GMAT R2026a 的 ``FiniteBurn``（配置）与 ``FiniteThrust``（力模型），
    未引入 ``Thruster`` 硬件层（见模块 docstring 的演进路径）。

    推力大小与方向解耦：``thrust_profile(t)`` 返回标量推力（牛顿），
    ``direction`` 给出传播坐标系内的方向（固定向量或随状态更新的可调用），
    内部归一化为单位向量。质量为常量（Slice 6 不支持推进剂消耗）。

    Args:
        thrust_profile: ``t -> thrust``（N，标量；``0`` 表示关机）。
        direction: 固定方向向量 ``(3,)``，或 ``(t, state) -> (3,)`` 可调用，
            均在传播（惯性）坐标系内。
        mass: 航天器质量（kg，常量）。
    """

    def __init__(
        self,
        thrust_profile: Callable[[float], float],
        direction: npt.ArrayLike | Callable[[float, npt.NDArray[np.floating]], npt.ArrayLike],
        mass: float,
    ) -> None:
        self._thrust_profile = thrust_profile
        self._direction = direction
        self._mass = float(mass)
        if self._mass <= 0:
            raise ValueError(f"mass must be positive, got {self._mass}")
        if not callable(direction):
            direction_arr = np.asarray(direction, dtype=float)
            if np.linalg.norm(direction_arr) < 1e-15:
                raise ValueError("direction must be a non-zero vector")

    @property
    def thrust_profile(self) -> Callable[[float], float]:
        """推力大小随时间变化的可调用（N）。"""
        return self._thrust_profile

    @property
    def direction(
        self,
    ) -> npt.ArrayLike | Callable[[float, npt.NDArray[np.floating]], npt.ArrayLike]:
        """推力方向：固定向量或 ``(t, state) -> (3,)`` 可调用。"""
        return self._direction

    @property
    def mass(self) -> float:
        """航天器质量（kg，常量）。"""
        return self._mass

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: object,
    ) -> npt.NDArray[np.floating]:
        """返回推力加速度，km/s²。

        ``thrust_profile(t) == 0`` 时返回 ``zeros(3)``；负推力 raise。
        """
        magnitude = float(self._thrust_profile(t))
        if magnitude < 0.0:
            raise ValueError(
                f"thrust_profile returned negative value {magnitude}"
            )
        if magnitude == 0.0:
            return np.zeros(3)
        direction = self._direction
        if callable(direction):
            direction = direction(t, np.asarray(state, dtype=float))
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-15:
            raise ValueError("direction must be a non-zero vector")
        direction_hat = direction / norm
        return (magnitude / self._mass) * direction_hat / 1000.0


@dataclass(frozen=True)
class BurnApplication:
    """单次脉冲机动在 ``propagate_maneuvers`` 输出中的记录。

    Args:
        index: post-burn 行在输出 ``states`` 中的行号。
        epoch: 施加时刻，SPICE et 秒。
        delta_v: 施加的 Δv，传播坐标系，``(3,)``。
        velocity_before: 施加前速度，``(3,)``。
        velocity_after: 施加后速度，``(3,)``。
    """

    index: int
    epoch: float
    delta_v: npt.NDArray[np.floating]
    velocity_before: npt.NDArray[np.floating]
    velocity_after: npt.NDArray[np.floating]
