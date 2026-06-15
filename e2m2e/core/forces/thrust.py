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
    ``direction`` 给出方向向量（固定向量或随状态更新的可调用），
    内部归一化为单位向量。质量为常量（Slice 6 不支持推进剂消耗）。

    ``direction_frame`` 支持 ``"VNB"``、``"LVLH"`` 与 ``None``：
    - ``None``：``direction`` 直接在传播（惯性）坐标系内解释。
    - ``"VNB"``：``direction`` 在 VNB 坐标系下解释，其中
      V = v/|v|（速度方向），N = (r x v)/|r x v|（角动量方向），
      B = V x N（副法向）。
    - ``"LVLH"``：``direction`` 在 LVLH 坐标系下解释，其中
      R = r/|r|（径向），V = v/|v|（沿迹方向），N = R x V（轨道面法向）。

    Args:
        thrust_profile: ``t -> thrust``（N，标量；``0`` 表示关机）。
        direction: 固定方向向量 ``(3,)``，或 ``(t, state) -> (3,)`` 可调用。
        mass: 航天器质量（kg，常量）。
        direction_frame: 方向解释坐标系，``"VNB"`` / ``"LVLH"`` / ``None``。
    """

    def __init__(
        self,
        thrust_profile: Callable[[float], float],
        direction: npt.ArrayLike | Callable[[float, npt.NDArray[np.floating]], npt.ArrayLike],
        mass: float,
        direction_frame: str | None = None,
    ) -> None:
        self._thrust_profile = thrust_profile
        self._direction = direction
        self._mass = float(mass)
        if self._mass <= 0:
            raise ValueError(f"mass must be positive, got {self._mass}")
        if direction_frame not in (None, "VNB", "LVLH"):
            raise ValueError(
                f"direction_frame must be 'VNB', 'LVLH', or None, got {direction_frame!r}"
            )
        self._direction_frame = direction_frame
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
    def direction_frame(self) -> str | None:
        """方向解释坐标系：'VNB'、'LVLH' 或 None。"""
        return self._direction_frame

    @property
    def mass(self) -> float:
        """航天器质量（kg，常量）。"""
        return self._mass

    def _resolve_direction_in_frame(
        self,
        direction_local: npt.NDArray[np.floating],
        state: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """把 direction_local 从 burn 坐标系转换到传播坐标系。

        Args:
            direction_local: 在 direction_frame 下的方向向量（未归一化）。
            state: 完整状态向量 (6,)，在传播坐标系下。

        Returns:
            传播坐标系下的方向向量（未归一化）。
        """
        frame = self._direction_frame
        if frame is None:
            return direction_local

        r = state[:3]
        v = state[3:6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        if frame == "VNB":
            if v_norm < 1e-15:
                raise ValueError("VNB frame requires non-zero velocity")
            V = v / v_norm
            h = np.cross(r, v)
            h_norm = np.linalg.norm(h)
            if h_norm < 1e-15:
                raise ValueError("VNB frame requires non-zero angular momentum")
            N = h / h_norm
            B = np.cross(V, N)
            # direction_local = [a_V, a_N, a_B] -> a_V * V + a_N * N + a_B * B
            return (
                direction_local[0] * V
                + direction_local[1] * N
                + direction_local[2] * B
            )

        if frame == "LVLH":
            if r_norm < 1e-15:
                raise ValueError("LVLH frame requires non-zero position")
            R = r / r_norm
            if v_norm < 1e-15:
                raise ValueError("LVLH frame requires non-zero velocity")
            V = v / v_norm
            N = np.cross(R, V)
            # direction_local = [a_R, a_V, a_N] -> a_R * R + a_V * V + a_N * N
            return (
                direction_local[0] * R
                + direction_local[1] * V
                + direction_local[2] * N
            )

        # unreachable: validated in __init__
        return direction_local

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

        if self._direction_frame is not None:
            state_arr = np.asarray(state, dtype=float)
            direction_hat = self._resolve_direction_in_frame(direction_hat, state_arr)
            # 重新归一化（坐标转换可能改变长度）
            new_norm = np.linalg.norm(direction_hat)
            if new_norm < 1e-15:
                raise ValueError("resolved direction is zero vector")
            direction_hat = direction_hat / new_norm

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
