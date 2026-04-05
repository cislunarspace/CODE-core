"""星历系统模块。

提供天体位置、引力常数等星历信息的统一查询接口，
底层通过 SPICE 工具包获取数据。
"""

from __future__ import annotations

from typing import List

from .spice import SPICEManager


class EphemerisSystem:
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
        bodies: List[str],
        spice: SPICEManager,
        origin: str = "EARTH",
        frame: str = "J2000",
    ) -> None:
        """初始化星历系统。

        Args:
            bodies: 需要纳入计算的天体名称列表。
            spice: 已完成内核加载的 SPICE 管理器实例。
            origin: 参考原点天体，所有位置矢量将相对于此天体计算。
            frame: 参考坐标系名称，用于确定位置矢量的坐标框架。
        """
        self.bodies = list(bodies)
        self.spice = spice
        self.origin = origin
        self.frame = frame

    def get_gm_values(self) -> List[float]:
        """获取所有管理天体的引力常数 (GM) 值。

        Returns:
            与 self.bodies 顺序对应的 GM 值列表，单位通常为 km³/s²。
        """
        return [self.spice.get_gm(body) for body in self.bodies]
