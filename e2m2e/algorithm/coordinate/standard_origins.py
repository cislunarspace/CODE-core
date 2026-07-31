"""常用坐标原点实现。

提供基于 SPICE 的天体中心原点等标准原点实现。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.kernels.manager import SPICEManager
from .origin import Origin


class CelestialBodyOrigin(Origin):
    """天体中心原点。

    表示某一大天体（如地球、月球、太阳）的中心，其在 ICRF 中的绝对状态
    通过 SPICE 实时查询。
    """

    def __init__(self, body: str, spice: SPICEManager) -> None:
        """初始化天体中心原点。

        Args:
            body: 天体名称，如 ``'EARTH'``、``'MOON'``、``'SUN'``。
            spice: SPICE 管理器实例，用于查询天体状态。
        """
        self._body = body.upper()
        self._spice = spice

    @property
    def body(self) -> str:
        """天体名称（大写）。"""
        return self._body

    def state(self, et: float) -> npt.NDArray[np.floating]:
        """返回该天体中心在 ICRF 中的绝对状态。

        Args:
            et: SPICE 历书时（秒）。

        Returns:
            长度为 6 的数组，前 3 个元素为位置（km），后 3 个为速度（km/s）。
        """
        return self._spice.get_body_state(self._body, et, "J2000", "SOLAR SYSTEM BARYCENTER")


class InertialOrigin(Origin):
    """惯性原点。

    表示 ICRF 原点本身，相对 ICRF 无平移。
    """

    def state(self, et: float) -> npt.NDArray[np.floating]:
        """返回零状态。"""
        return np.zeros(6)

