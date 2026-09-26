"""EPPR（星历脉动旋转）系统：CR3BP 无量纲槽位 + 星历驱动帧量。

``EPPRSystem`` 与 ``EPPR_Dynamics`` 构成 ADR 0027 模型阶梯的一对新槽位：状态与
CR3BP 同为无量纲（时间单位 t_c、长度单位 lstar、总质量 1），但坐标系为真实星历
驱动的 EPPR（x̂ 地球→月球、ẑ 瞬时轨道角动量、长度尺度 = 瞬时地月距离 d(t)）。
主星固定在 EPPR 无量纲 (−μ,0,0) 与 (1−μ,0,0)，帧量由 ``EPPRFrameModel`` 从同一
星历查询路径供给。

帧量推导复用 ``algorithm/coordinate/eppr_frame.py``（该模块不 import dynamics，
无环）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.constants import Datum
from ...data.templates.enums import ReferenceFrame, UnitSystem
from ..coordinate.eppr_frame import EPPRFrameModel
from .cr3bp_system import CR3BP_System
from .system import System


class EPPRSystem(System):
    """Earth-Moon EPPR 系统（星历脉动旋转系，无量纲）。"""

    def __init__(
        self,
        cr3bp_system: CR3BP_System,
        spice=None,
        et0: float = 0.0,
        *,
        frame_model: EPPRFrameModel | None = None,
    ) -> None:
        """初始化 EPPR 系统。

        Args:
            cr3bp_system: 提供 μ、特征尺度与天体名的 CR3BP 系统。
            spice: ``SPICEManager``；``frame_model`` 未给出时必填（否则
                ``TypeError``）。
            et0: 参考历元 SPICE ET 秒（无因次时间 τ = 0 对应时刻）。
            frame_model: 显式帧量模型（测试注入合成星历用）。
        """
        if frame_model is None:
            if spice is None:
                raise TypeError("EPPRSystem 需要 spice，或显式提供 frame_model")
            frame_model = EPPRFrameModel(spice, cr3bp_system.mu)

        self.cr3bp_system = cr3bp_system
        self.spice = spice
        self.et0 = float(et0)
        self.frame_model = frame_model

    # ---- System 抽象接口 ----

    @property
    def frame(self) -> ReferenceFrame:
        """EPPR 星历脉动旋转框架。"""
        return ReferenceFrame.EPPR

    @property
    def unit_system(self) -> UnitSystem:
        """与 CR3BP 一致的无量纲单位。"""
        return UnitSystem.DIMENSIONLESS

    def gravitational_parameter(self, body: str) -> float:
        """无量纲引力参数（委托 ``cr3bp_system``）。"""
        return self.cr3bp_system.gravitational_parameter(body)

    # ---- 委托属性（碰撞检测等按 CR3BP 口径消费） ----

    @property
    def mu(self) -> float:
        """质量参数 μ。"""
        return self.cr3bp_system.mu

    @property
    def primary_body(self) -> str:
        """主天体名称。"""
        return self.cr3bp_system.primary_body

    @property
    def secondary_body(self) -> str:
        """次天体名称。"""
        return self.cr3bp_system.secondary_body

    @property
    def primary_radius_km(self) -> float | None:
        """主天体半径（km），供碰撞终止。"""
        return self.cr3bp_system.primary_radius_km

    @property
    def secondary_radius_km(self) -> float | None:
        """次天体半径（km），供碰撞终止。"""
        return self.cr3bp_system.secondary_radius_km

    @property
    def characteristic_time(self) -> float:
        """特征时间 t_c (s)；未设置时回退 ``Datum.DE421.char_time_s``。"""
        characteristic_time = self.cr3bp_system.characteristic_time
        if characteristic_time is None:
            return float(Datum.DE421.char_time_s)
        return float(characteristic_time)

    @property
    def characteristic_length(self) -> float:
        """特征长度 lstar (km)；未设置时回退 ``Datum.DE421.char_length_km``。"""
        characteristic_length = self.cr3bp_system.characteristic_length
        if characteristic_length is None:
            return float(Datum.DE421.char_length_km)
        return float(characteristic_length)

    def get_body_position(self, body: str, t: float) -> npt.NDArray[np.floating]:
        """返回主／次天体在 EPPR 无量纲坐标中的位置（固定，与 t 无关）。

        Args:
            body: ``"primary"`` 或 ``"secondary"``。
            t: 无量纲时间（EPPR 中主星位置不随时间变化，仅为接口兼容保留）。
        """
        mu = self.mu
        if body == "primary":
            return np.array([-mu, 0.0, 0.0])
        if body == "secondary":
            return np.array([1.0 - mu, 0.0, 0.0])
        raise ValueError(f"EPPR 系统的 body 须为 'primary' 或 'secondary'，得到 {body!r}")
