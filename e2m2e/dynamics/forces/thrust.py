r"""推力与机动模型。

提供两种推力/机动表示：

- ``ImpulsiveBurn``：瞬时 Δv 机动事件，由 ``ForceModel.propagate_maneuvers``
  在指定 epoch 处中断传播并施加速度增量。
- ``FiniteBurn``：连续推力加速度力模型，继承 ``PhysicalModel``，
  在传播过程中实时参与加速度计算。

``FiniteBurn`` 合并了 GMAT R2026a 的 ``FiniteBurn``（配置）与
``FiniteThrust``（力模型）两层，未引入 ``Thruster`` 硬件层。
推力大小与方向解耦：``thrust_profile(t)`` 返回标量推力（N），
``direction`` 给出方向向量（固定向量或随状态更新的可调用），
内部归一化为单位向量。质量为常量（不支持推进剂消耗）。

``direction_frame`` 支持 ``"VNB"``、``"LVLH"`` 与 ``None``：

- ``None``：``direction`` 直接在传播（惯性）坐标系内解释。
- ``"VNB"``：``direction`` 在 VNB 坐标系下解释，其中
  :math:`V = v/\\|v\\|` (速度方向)，
  :math:`N = (r \\times v)/\\|r \\times v\\|` (角动量方向)，
  :math:`B = V \\times N` (副法向)。
- ``"LVLH"``：``direction`` 在 LVLH 坐标系下解释，其中
  :math:`R = r/\\|r\\|` (径向)，
  :math:`V = v/\\|v\\|` (沿迹方向)，
  :math:`N = R \\times V` (轨道面法向)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..standard_dynamic_axes import LVLHAxes, VNBAxes
from .physical_model import PhysicalModel


@dataclass(frozen=True)
class ImpulsiveBurn:
    """瞬时 Δv 机动事件。

    ``delta_v`` 在传播（惯性）坐标系内解释，由
    :meth:`e2m2e.core.forces.force_model.ForceModel.propagate_maneuvers`
    在 ``epoch`` 处施加 ``state[3:6] += delta_v``。

    VNB/LVLH burn 坐标系暂不支持（届时加 ``frame`` 字段，转换走
    :meth:`CoordinateSystem.transform_vector`，对应 GMAT
    ``Burn::ConvertDeltaVToInertial`` 的 ``coincident=true`` 纯旋转）。

    Args:
        epoch: 施加时刻，SPICE et 秒，与 ``ForceModel.propagate`` 的
            ``t_span`` 一致。
        delta_v: 速度增量，参考系，形状 ``(3,)``。
    """

    epoch: float
    delta_v: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        """以拷贝方式存储 delta_v，避免外部数组变更影响 burn。"""
        object.__setattr__(self, "delta_v", np.asarray(self.delta_v, dtype=float).copy())


class FiniteBurn(PhysicalModel):
    """连续推力加速度力模型。

    推力大小与方向解耦：``thrust_profile(t)`` 返回标量推力（牛顿），
    ``direction`` 给出方向向量（固定向量或随状态更新的可调用），
    内部归一化为单位向量。质量为常量（不支持推进剂消耗）。

    ``direction_frame`` 支持 ``"VNB"``、``"LVLH"`` 与 ``None``：

    - ``None``：``direction`` 直接在传播（惯性）坐标系内解释。
    - ``"VNB"``：``direction`` 在 VNB 坐标系下解释，其中
      :math:`V = v/\\|v\\|` (速度方向)，
      :math:`N = (r \\times v)/\\|r \\times v\\|` (角动量方向)，
      :math:`B = V \\times N` (副法向)。
    - ``"LVLH"``：``direction`` 在 LVLH 坐标系下解释，其中
      :math:`R = r/\\|r\\|` (径向)，
      :math:`V = v/\\|v\\|` (沿迹方向)，
      :math:`N = R \\times V` (轨道面法向)。

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
        if direction_frame == "VNB":
            self._axes: VNBAxes | LVLHAxes | None = VNBAxes()
        elif direction_frame == "LVLH":
            self._axes = LVLHAxes()
        else:
            self._axes = None
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
        t: float,
        direction_local: npt.NDArray[np.floating],
        state: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """把 direction_local 从 burn 坐标系转换到传播坐标系。

        复用 :class:`~e2m2e.core.standard_dynamic_axes.VNBAxes` /
        :class:`~e2m2e.core.standard_dynamic_axes.LVLHAxes` 构造旋转矩阵，
        再用 ``rotation @ direction_local`` 完成变换。轴向定义沿用
        ``standard_dynamic_axes``（VNB/LVLH 按 GMAT 约定，见 CONTEXT.md）。

        动态坐标轴类本身不校验状态退化情形，这里保留原手搓逻辑的边界
        检查（零速度/零位置/共线 r-v），以抛出含义清晰的 ``ValueError``
        并避免 ``LVLHAxes`` 在角动量为零时产生 NaN。

        Args:
            t: 当前历元（秒），透传给 ``axes.update``/``rotation_matrix``。
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
        h_norm = np.linalg.norm(np.cross(r, v))

        if frame == "VNB":
            if v_norm < 1e-15:
                raise ValueError("VNB frame requires non-zero velocity")
            if h_norm < 1e-15:
                raise ValueError("VNB frame requires non-zero angular momentum")
        else:  # LVLH
            if r_norm < 1e-15:
                raise ValueError("LVLH frame requires non-zero position")
            if v_norm < 1e-15:
                raise ValueError("LVLH frame requires non-zero velocity")
            if h_norm < 1e-15:
                # 退化的径向（直线）轨道：r 与 v 共线，LVLHAxes 的 h_hat 未定义
                # (h/|h| → NaN)。沿用原手搓逻辑在此情形下的稳健行为：
                # N = R × V = 0，仅径向/沿迹分量有意义。
                R = r / r_norm
                V = v / v_norm
                return direction_local[0] * R + direction_local[1] * V

        assert self._axes is not None  # direction_frame validated in __init__
        self._axes.update(t, state)
        rotation = self._axes.rotation_matrix(t)
        return rotation @ direction_local

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
            raise ValueError(f"thrust_profile returned negative value {magnitude}")
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
            direction_hat = self._resolve_direction_in_frame(t, direction_hat, state_arr)
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
