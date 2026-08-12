"""动力学系统抽象基类模块。

定义 ``System`` 抽象基类，统一 CR3BP 系统与星历系统的公共接口。
具体实现见 ``cr3bp_system.py`` （``CR3BP_System``）和 ``ephemeris_system.py``
（``EphemerisSystem``）。
"""

from __future__ import annotations

import abc

from ...data.templates.enums import ReferenceFrame, UnitSystem


class System(abc.ABC):
    """动力学系统抽象基类。

    描述天体的几何、引力与运动学模型，是后续一切计算的上下文。
    所有具体系统（CR3BP、星历）都应实现以下最小接口：

    - ``frame``：坐标框架（``ReferenceFrame``）
    - ``unit_system``：单位系统（``UnitSystem``）
    - ``gravitational_parameter(body)``：天体引力参数

    注意：``mu``、``body_state(body, t)``、``coordinate_system`` 等不在基类中，
    它们属于特定系统实现的概念。
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

    def get_body_position(self, body: str, t: float):
        """获取天体在时刻 t 的位置（仅星历系统实现）。

        CR3BP 等非星历系统不使用此方法，调用时抛出 NotImplementedError。

        Args:
            body: 天体名称（如 "MOON"、"SUN"）。
            t: SPICE et 秒。

        Returns:
            天体位置向量。
        """
        raise NotImplementedError(f"{type(self).__name__} does not support get_body_position")
