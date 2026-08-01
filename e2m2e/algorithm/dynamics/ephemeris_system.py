"""星历系统模块。

提供天体位置、引力常数等星历信息的统一查询接口，
底层通过 SPICE 工具包获取数据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.kernels.manager import SPICEManager
from ...data.templates.enums import ReferenceFrame, UnitSystem
from .system import System

if TYPE_CHECKING:
    from ..coordinate.coordinate_system import CoordinateSystem


class EphemerisSystem(System):
    """星历系统，管理一组天体的星历查询。

    封装 SPICE 工具包，为轨道设计流程提供统一的天体数据访问层。
    支持自定义参考原点和坐标框架。

    Attributes:
        bodies: 天体名称列表，如 ["EARTH", "MOON", "SUN"]。
        spice: SPICE 管理器实例，负责底层星历数据读取。
        origin: 参考原点天体名称，默认为 "EARTH"。
        frame: 坐标系名称，默认为 "J2000"。
    """

    def __init__(
        self,
        bodies: list[str],
        spice: SPICEManager,
        origin: str = "EARTH",
        frame: ReferenceFrame = ReferenceFrame.J2000,
        coordinate_system: CoordinateSystem | None = None,
    ) -> None:
        """初始化星历系统。

        Args:
            bodies: 需要纳入计算的天体名称列表。
            spice: 已完成内核加载的 SPICE 管理器实例。
            origin: 参考原点天体，所有位置矢量将相对于此天体计算。
            frame: 参考坐标系名称，用于确定位置矢量的坐标框架。
            coordinate_system: 可选的默认坐标系；用于 ForceModel 传播。
        """
        self.bodies = list(bodies)
        self.spice = spice
        self.origin = origin
        self._frame = frame
        self._coordinate_system = coordinate_system

    @property
    def frame(self) -> ReferenceFrame:
        """星历系统的坐标框架。"""
        return self._frame

    @property
    def unit_system(self) -> UnitSystem:
        """星历系统使用物理单位。"""
        return UnitSystem.SI

    @property
    def coordinate_system(self) -> CoordinateSystem | None:
        """星历系统的默认坐标系。"""
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value: CoordinateSystem | None) -> None:
        self._coordinate_system = value

    def update_coordinate_systems(self, t: float, state: npt.ArrayLike) -> None:
        """更新动态坐标系。

        若 ``coordinate_system.axes`` 为 ``DynamicAxes`` 实例，
        调用 ``axes.update(t, state)``。
        """
        cs = self.coordinate_system
        if cs is None:
            return
        axes = getattr(cs, "axes", None)
        if axes is None:
            return
        from ..coordinate.dynamic_axes import DynamicAxes

        if isinstance(axes, DynamicAxes):
            axes.update(t, np.asarray(state, dtype=float))

    def gravitational_parameter(self, body: str) -> float:
        """获取天体的引力参数 GM。

        Args:
            body: 天体名称。

        Returns:
            GM 值，单位 km³/s²。
        """
        return self.spice.get_gm(body)

    def get_body_position(self, body: str, et: float) -> npt.NDArray[np.floating]:
        """获取天体相对于原点的位置向量。

        自动使用初始化时设定的 frame 和 origin。

        Args:
            body: 天体名称，如 "MOON"、"SUN"。
            et: 历书时（秒）。

        Returns:
            位置向量，形状 (3,)，单位 km。
        """
        return self.spice.get_body_position(body, et, self.frame.value, self.origin)

    def get_body_state(self, body: str, et: float) -> npt.NDArray[np.floating]:
        """获取天体相对于原点的状态向量。

        自动使用初始化时设定的 frame 和 origin。

        Args:
            body: 天体名称，如 "MOON"、"SUN"。
            et: 历书时（秒）。

        Returns:
            状态向量，形状 (6,)，前 3 元素为位置 [km]，后 3 元素为速度 [km/s]。
        """
        return self.spice.get_body_state(body, et, self.frame.value, self.origin)

    def get_gm(self, body: str) -> float:
        """获取天体的引力参数 GM。

        Args:
            body: 天体名称。

        Returns:
            GM 值，单位 km³/s²。
        """
        return self.spice.get_gm(body)

    def get_gm_values(self) -> npt.NDArray[np.floating]:
        """获取所有管理天体的引力常数 (GM) 值。

        Returns:
            与 self.bodies 顺序对应的 GM 值数组，单位通常为 km³/s²。
        """
        return np.array([self.spice.get_gm(body) for body in self.bodies])
