r"""推力与机动模型。

提供两种推力/机动表示：

- ``ImpulsiveBurn``：瞬时 Δv 机动事件，由 ``ForceModel.propagate_maneuvers``
  在指定 epoch 处中断传播并施加速度增量。
- ``FiniteBurn``：连续推力加速度力模型，继承 ``PhysicalModel``，
  在传播过程中实时参与加速度计算。

``FiniteBurn`` 合并了 GMAT R2026a 的 ``FiniteBurn`` （配置）与
``FiniteThrust`` （力模型）两层，未引入 ``Thruster`` 硬件层。
推力大小与方向解耦：``thrust_profile(t)`` 返回标量推力（N），
``direction`` 给出方向向量（固定向量或随状态更新的可调用），
内部归一化为单位向量。质量为常量（不支持推进剂消耗）。

:py:class:`VariableMassFiniteBurn` 是其可变质量对应物：质量作为状态量
``state[6]`` 随推力消耗（``ṁ = −T/(Isp·g₀)``），是低推力最优控制与
月面动力下降的受控动力学基座。详见
``docs/plans/lowthrust-foundation-prd.md``。

``direction_frame`` 支持 ``"VNB"``、``"LVLH"`` 与 ``None``：

- ``None``：``direction`` 直接在传播（惯性）坐标系内解释。
- ``"VNB"``：``direction`` 在 VNB 坐标系下解释，其中
  :math:`V = v/\\|v\\|` (速度方向)，
  :math:`N = (r \\times v)/\\|r \\times v\\|` (角动量方向)，
  :math:`B = V \\times N` (副法向)。
- ``"LVLH"``：``direction`` 在 LVLH 坐标系下解释，其中
  :math:`R = r/\\|r\\|` (径向)，
  :math:`N = (r \\times v)/\\|r \\times v\\|` (法向)，
  :math:`T = N \\times R` (沿迹方向)。
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
    :meth:`e2m2e.algorithm.forces.force_model.ForceModel.propagate_maneuvers`
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
    """恒质量连续推力加速度力模型。

    6D 状态传播由 Rust 编译路径执行。配置 DSL 构造的常量或 pulse 推力曲线和
    固定方向可下沉；任意 Python callable 无法进入 Rust RK 内循环，在传播入口会
    显式报能力错误。需要推进剂消耗时使用 VariableMassFiniteBurn（变质量，7D 状态）。

    ``direction`` 给出方向向量（固定向量或随状态更新的可调用），
    内部归一化为单位向量。质量为常量（不支持推进剂消耗）。

    ``direction_frame`` 支持 ``"VNB"``、``"LVLH"`` 与 ``None``：

    - ``None``：``direction`` 直接在传播（惯性）坐标系内解释。
    - ``"VNB"``：三个分量依次对应速度单位向量、角动量单位向量和副法向量。
    - ``"LVLH"``：三个分量依次对应径向单位向量、沿迹单位向量和轨道面法向量；
      沿迹单位向量由法向量叉径向量得到。

    Args:
        thrust_profile: ``t -> thrust`` （N，标量；``0`` 表示关机）。
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

    def to_rust_spec(self, system: object) -> tuple | None:
        """序列化为恒质量 6D 编译传播接受的推力规格。

        只有配置 DSL 构造的 constant/pulse 推力 profile 和固定方向可以下沉
        到 Rust；任意 Python callable 无法在 Rust RK 内安全求值，返回 ``None``。
        返回规格为 ``("low_thrust", mass, thrust, t_start, t_end, direction,
        direction_frame)``，其中 constant profile 的起止时间为 ``None``。
        """
        profile_info = getattr(self._thrust_profile, "_e2m2e_config_kind", None)
        if profile_info is None or callable(self._direction):
            return None
        _kind, profile = profile_info
        kind = profile["kind"]
        if kind == "constant":
            t_start = None
            t_end = None
        elif kind == "pulse":
            t_start = float(profile["t_start"])
            t_end = float(profile["t_end"])
        else:
            raise NotImplementedError(f"FiniteBurn 不支持编译推力 profile: {kind!r}")
        direction = np.asarray(self._direction, dtype=float)
        return (
            "low_thrust",
            self._mass,
            float(profile["thrust"]),
            t_start,
            t_end,
            direction.tolist(),
            self._direction_frame,
        )


class VariableMassFiniteBurn(PhysicalModel):
    """连续推力加速度力模型（质量随推力消耗）。

    与 :class:`FiniteBurn` 的唯一区别：质量不是常量，而是状态量
    ``state[6]``。低推力转移与月面动力下降等最优控制问题中，质量演化
    是燃耗最优的基本变量（``ṁ = −T/(Isp·g₀)``），必须纳入状态向量。

    配套的 7D 传播在 :class:`~e2m2e.algorithm.forces.force_model.ForceModel.propagate`
    中走 Rust 快速路径 ``propagate_compiled_lowthrust``：状态
    ``[x, y, z, vx, vy, vz, m]``，受控动力学在 Rust 侧（``augmented_state``
    的 ``augmented_eom_7d``）。详见 ``docs/plans/lowthrust-foundation-prd.md``。

    推力大小与方向解耦，语义同 :class:`FiniteBurn`。``direction`` 支持
    固定向量或 ``(t, state) -> (3,)`` 可调用；``state`` 为 7D 时可调用方向
    可读取 ``state[6]`` 中的质量。``direction_frame`` 支持 ``"VNB"`` /
    ``"LVLH"`` / ``None``，帧解析与 :class:`FiniteBurn` 一致。

    Args:
        thrust: 推力幅值（N，常量）。
        isp: 比冲（s）。
        initial_mass: 初始质量（kg），用于初始化状态第 7 维与校验。
        direction: 固定方向向量 ``(3,)``，或 ``(t, state) -> (3,)`` 可调用。
        direction_frame: 方向解释坐标系，``"VNB"`` / ``"LVLH"`` / ``None``。
    """

    def __init__(
        self,
        thrust: float,
        isp: float,
        initial_mass: float,
        direction: npt.ArrayLike | Callable[[float, npt.NDArray[np.floating]], npt.ArrayLike],
        direction_frame: str | None = None,
    ) -> None:
        if thrust < 0.0:
            raise ValueError(f"thrust must be non-negative, got {thrust}")
        if isp <= 0.0:
            raise ValueError(f"isp must be positive, got {isp}")
        if initial_mass <= 0.0:
            raise ValueError(f"initial_mass must be positive, got {initial_mass}")
        if direction_frame not in (None, "VNB", "LVLH"):
            raise ValueError(
                f"direction_frame must be 'VNB', 'LVLH', or None, got {direction_frame!r}"
            )
        self._thrust = float(thrust)
        self._isp = float(isp)
        self._initial_mass = float(initial_mass)
        self._direction = direction
        self._direction_frame = direction_frame
        if not callable(direction):
            direction_arr = np.asarray(direction, dtype=float)
            if np.linalg.norm(direction_arr) < 1e-15:
                raise ValueError("direction must be a non-zero vector")

    @property
    def thrust(self) -> float:
        """推力幅值（N，常量）。"""
        return self._thrust

    @property
    def isp(self) -> float:
        """比冲（s）。"""
        return self._isp

    @property
    def initial_mass(self) -> float:
        """初始质量（kg），用于初始化状态第 7 维。"""
        return self._initial_mass

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

    def to_rust_spec(self, system: object) -> tuple | None:
        """序列化为低推力 7D 传播路径接受的推力规格。

        仅当 ``direction`` 为固定向量时返回元组（可调用方向需 Python 求值，
        无法下沉到 Rust）；常量推力映射成满油门（``throttle = 1.0``），
        ``t_max = thrust``。返回元组会被 ``ForceModel`` 的低推力分支拆出，
        交给 ``propagate_compiled_lowthrust``，不经过 6D 的 ``CompiledForce``
        路径。
        """
        if callable(self._direction):
            return None
        direction = np.asarray(self._direction, dtype=float)
        return (
            "low_thrust_variable",
            self._thrust,
            self._isp,
            1.0,  # throttle：常量推力即满推力
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        )


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
