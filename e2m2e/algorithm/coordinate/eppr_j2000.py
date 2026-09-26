"""基于 ``CoordinateSystem`` 的 EPPR ↔ J2000 转换器。

EPPR 坐标系 = (EPPRAxes, CelestialBodyOrigin("EARTH"))
J2000 坐标系 = (ICRSAxes, CelestialBodyOrigin("EARTH"))

两系共享同一原点实例，``CoordinateSystem.transform_state`` 的原点项代数对消，
只剩轴向旋转（含旋转角速度项）；EPPR 独有脉动项（长度尺度 d(t) 与 ḋ/d）在本类
显式处理。EPPR 位置为**质心原点**无量纲量（地球在 (−μ,0,0)、月球在 (1−μ,0,0)），
故转换时经偏移 ``[μ,0,0]`` 在地心↔质心间平移；速度的无量纲化因子为 t_c/d(t)。

无量纲时间与 CR3BP 约定一致：``t_c`` 取 ``cr3bp_system.characteristic_time``，缺省
回退 ``Datum.DE421.char_time_s``。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.constants import Datum
from ..dynamics.cr3bp_system import CR3BP_System
from .coordinate_system import CoordinateSystem
from .eppr_axes import EPPRAxes
from .eppr_frame import EPPRFrameModel
from .standard_axes import ICRSAxes
from .standard_origins import CelestialBodyOrigin, InertialOrigin


class EPPRJ2000System:
    """基于 ``CoordinateSystem`` 的 EPPR ↔ J2000 双向转换器。"""

    def __init__(
        self,
        cr3bp_system: CR3BP_System,
        spice=None,
        *,
        frame_model: EPPRFrameModel | None = None,
    ) -> None:
        """初始化转换器。

        Args:
            cr3bp_system: 提供 μ 与特征时间／长度的 CR3BP 系统。
            spice: ``SPICEManager``；``frame_model`` 未显式给出时必填（否则
                ``TypeError``），用于推导帧量并为地心原点查询地球状态。
            frame_model: 显式帧量模型（测试注入合成星历用）；给出时 ``spice``
                可省略（原点退化为 ``InertialOrigin`` —— 两系共享同一原点，
                原点项代数对消，数值不变）。
        """
        if frame_model is None:
            if spice is None:
                raise TypeError("EPPRJ2000System 需要 spice，或显式提供 frame_model")
            frame_model = EPPRFrameModel(spice, cr3bp_system.mu)

        self.cr3bp_system = cr3bp_system
        self.spice = spice
        self.frame_model = frame_model
        self.eppr_axes = EPPRAxes(frame_model)
        origin = CelestialBodyOrigin("EARTH", spice) if spice is not None else InertialOrigin()
        self.eppr_cs = CoordinateSystem(self.eppr_axes, origin)
        self.j2000_cs = CoordinateSystem(ICRSAxes(), origin)

    def _get_time_unit(self) -> float:
        characteristic_time = getattr(self.cr3bp_system, "characteristic_time", None)
        if characteristic_time is not None:
            return float(characteristic_time)
        return float(Datum.DE421.char_time_s)

    def _bary_to_earth_offset(self) -> npt.NDArray[np.floating]:
        return np.array([float(self.cr3bp_system.mu), 0.0, 0.0])

    def j2000_to_eppr(
        self, state_j2000: npt.ArrayLike, t_nd: float, et0: float
    ) -> npt.NDArray[np.floating]:
        """地心 J2000 状态 → EPPR 无量纲状态（质心原点）。

        Args:
            state_j2000: 地心 J2000 状态 [km, km/s]。
            t_nd: 相对 ``et0`` 的无量纲时间（τ）。
            et0: 参考历元 SPICE ET 秒。
        """
        state_j2000 = np.asarray(state_j2000, dtype=float)
        t_c = self._get_time_unit()
        et = et0 + t_nd * t_c
        frame_state = self.frame_model.frame_state(et)

        # 地心系旋转结果：s[:3] = Rᵀ·x = d·x̃；s[3:] = ḋ·x̃ + d·ẋ̃（旋转项代数对消）。
        s = self.j2000_cs.transform_state(state_j2000, self.j2000_cs, self.eppr_cs, et)
        x_tilde = s[:3] / frame_state.distance
        r_eppr = x_tilde - self._bary_to_earth_offset()
        v_eppr = (t_c / frame_state.distance) * (s[3:] - frame_state.distance_rate * x_tilde)
        return np.concatenate([r_eppr, v_eppr])

    def eppr_to_j2000(
        self, state_eppr: npt.ArrayLike, t_nd: float, et0: float
    ) -> npt.NDArray[np.floating]:
        """EPPR 无量纲状态（质心原点）→ 地心 J2000 状态 [km, km/s]。"""
        state_eppr = np.asarray(state_eppr, dtype=float)
        t_c = self._get_time_unit()
        et = et0 + t_nd * t_c
        frame_state = self.frame_model.frame_state(et)

        x_tilde = state_eppr[:3] + self._bary_to_earth_offset()
        # 轴向位置 d·x̃ 与其时间导数 ḋ·x̃ + d·ẋ̃（ẋ̃ = ξ'/t_c）。
        state_in = np.concatenate(
            [
                frame_state.distance * x_tilde,
                frame_state.distance * state_eppr[3:] / t_c + frame_state.distance_rate * x_tilde,
            ]
        )
        return self.j2000_cs.transform_state(state_in, self.eppr_cs, self.j2000_cs, et)

    def batch_j2000_to_eppr(
        self,
        states_j2000: npt.ArrayLike,
        t_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray[np.floating]:
        """批量 J2000→EPPR（逐条调用标量版；无 Rust 通道）。"""
        states = np.asarray(states_j2000, dtype=float)
        times = np.asarray(t_arr, dtype=float)
        out = [
            self.j2000_to_eppr(state, float(t), et0) for state, t in zip(states, times, strict=True)
        ]
        return np.asarray(out, dtype=float).reshape(-1, 6)

    def batch_eppr_to_j2000(
        self,
        states_eppr: npt.ArrayLike,
        t_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray[np.floating]:
        """批量 EPPR→J2000（逐条调用标量版；无 Rust 通道）。"""
        states = np.asarray(states_eppr, dtype=float)
        times = np.asarray(t_arr, dtype=float)
        out = [
            self.eppr_to_j2000(state, float(t), et0) for state, t in zip(states, times, strict=True)
        ]
        return np.asarray(out, dtype=float).reshape(-1, 6)
