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
  :math:`V = v/\\|v\\|` (沿迹方向)，
  :math:`N = R \\times V` (轨道面法向)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..coordinate.standard_dynamic_axes import LVLHAxes, VNBAxes
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


def _resolve_thrust_direction(
    t: float,
    direction_local: npt.NDArray[np.floating],
    state: npt.NDArray[np.floating],
    direction_frame: str | None,
    axes: LVLHAxes | VNBAxes | None,
) -> npt.NDArray[np.floating]:
    """把 direction_local 从 burn 坐标系转换到传播坐标系。

    复用 :class:`~e2m2e.core.standard_dynamic_axes.VNBAxes` /
    :class:`~e2m2e.core.standard_dynamic_axes.LVLHAxes` 构造旋转矩阵，
    再用 ``rotation @ direction_local`` 完成变换。轴向定义沿用
    ``standard_dynamic_axes`` （VNB/LVLH 按 GMAT 约定）。

    动态坐标轴类本身不校验状态退化情形，这里保留边界检查（零速度/零
    位置/共线 r-v），抛出含义清晰的 ``ValueError`` 并避免 ``LVLHAxes``
    在角动量为零时产生 NaN。``FiniteBurn`` 与 ``VariableMassFiniteBurn``
    共用此函数。

    Args:
        t: 当前历元（秒），透传给 ``axes.update``/``rotation_matrix``。
        direction_local: 在 direction_frame 下的方向向量（未归一化）。
        state: 状态向量（至少 6 维），在传播坐标系下。
        direction_frame: 方向解释坐标系，``"VNB"`` / ``"LVLH"`` / ``None``。
        axes: 与 direction_frame 配套的动态坐标轴实例，``None`` 时 frame
            必须也为 ``None``。

    Returns:
        传播坐标系下的方向向量（未归一化）。
    """
    if direction_frame is None:
        return direction_local

    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_norm = np.linalg.norm(np.cross(r, v))

    if direction_frame == "VNB":
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

    assert axes is not None  # direction_frame validated in __init__
    axes.update(t, state)
    rotation = axes.rotation_matrix(t)
    return rotation @ direction_local


class FiniteBurn(PhysicalModel):
    """连续推力加速度力模型（实现状态：预留，见 issue #407）。

    恒质量低推力从未有效接入传播：``compute_acceleration`` 已按 #378 删除，
    ``to_rust_spec`` 尚未接入 Rust 侧已实现的 ``CompiledForce::LowThrust``
    变体。方向帧解析逻辑（``_resolve_thrust_direction``）代码已就绪但无消费者。
    需要推力传播请改用 :class:`VariableMassFiniteBurn`（变质量，7D 状态）。

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
        """把 direction_local 从 burn 坐标系转换到传播坐标系（委托模块函数）。"""
        return _resolve_thrust_direction(
            t, direction_local, state, self._direction_frame, self._axes
        )

    def to_rust_spec(self, system: object) -> tuple | None:
        """``FiniteBurn`` 尚未接入 Rust 编译路径，显式抛 ``NotImplementedError``。

        Rust 侧 ``CompiledForce::LowThrust`` 变体已实现（恒质量小推力，元组格式
        ``("low_thrust", t_max, isp, throttle, direction)``），但 Python 侧
        ``FiniteBurn`` 尚未接入——时变 ``thrust_profile`` 到固定 t_max/throttle
        的映射、isp 来源等接口设计待定（issue #407）。需要推力传播请改用
        :class:`VariableMassFiniteBurn` （7D 状态，走
        ``propagate_compiled_lowthrust``）。issue #378：不允许静默回退 Python。
        """
        raise NotImplementedError(
            "FiniteBurn 恒质量低推力尚未接入 Rust 编译传播"
            "（CompiledForce::LowThrust 变体已实现，Python 侧待接入，见 #407）；"
            "如需推力传播请改用 VariableMassFiniteBurn。"
        )


class VariableMassFiniteBurn(PhysicalModel):
    """连续推力加速度力模型（质量随推力消耗）。

    与 :class:`FiniteBurn` 的唯一区别：质量不是常量，而是状态量
    ``state[6]``。低推力转移与月面动力下降等最优控制问题中，质量演化
    是燃耗最优的基本变量（``ṁ = −T/(Isp·g₀)``），必须纳入状态向量。

    配套的 7D 传播在 :class:`~e2m2e.core.forces.force_model.ForceModel.propagate`
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
