"""动力学系统抽象基类模块。

定义 ``System`` 抽象基类，统一 CR3BP 系统与星历系统的公共接口。
具体实现见 ``cr3bp_system.py``（``CR3BP_System``）和 ``ephemeris_system.py``
（``EphemerisSystem``）。
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import numpy.typing as npt

from .enums import ReferenceFrame, UnitSystem


class System(abc.ABC):
    """动力学系统抽象基类。

    描述天体的几何、引力与运动学模型，是后续一切计算的上下文。
    所有具体系统（CR3BP、星历）都应实现以下最小接口：

    - ``frame``：坐标框架（``ReferenceFrame``）
    - ``unit_system``：单位系统（``UnitSystem``）
    - ``gravitational_parameter(body)``：天体引力参数

    注意：``mu``、``body_state(body, t)`` 等不在基类中，它们属于
    特定系统实现的概念。
    """

    @property
    @abc.abstractmethod
    def frame(self) -> ReferenceFrame:
        """系统的坐标框架。"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def unit_system(self) -> UnitSystem:
        """系统的单位系统。"""
        raise NotImplementedError

    @property
    def coordinate_system(self) -> Any:
        """系统默认坐标系；具体系统可覆盖。"""
        return None

    def update_coordinate_systems(self, t: float, state: npt.ArrayLike) -> None:
        """更新动态坐标系。

        若 ``coordinate_system.axes`` 为 ``DynamicAxes`` 实例，
        调用 ``axes.update(t, state)``。默认 no-op。
        """
        cs = self.coordinate_system
        if cs is None:
            return
        axes = getattr(cs, "axes", None)
        if axes is None:
            return
        from .coordinate.dynamic_axes import DynamicAxes

        if isinstance(axes, DynamicAxes):
            axes.update(t, np.asarray(state, dtype=float))

    @abc.abstractmethod
    def gravitational_parameter(self, body: str) -> float:
        """获取指定天体的引力参数。

        Args:
            body: 天体标识。CR3BP 系统接受 "primary"/"secondary"；
                星历系统接受 SPICE 天体名称（如 "EARTH"、"MOON"）。

        Returns:
            引力参数。CR3BP 系统返回无量纲值，星历系统返回 km³/s²。
        """
        raise NotImplementedError
