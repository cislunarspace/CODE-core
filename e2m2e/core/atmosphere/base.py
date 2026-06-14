"""大气密度模型抽象基类。"""

from __future__ import annotations

import abc


class AtmosphereModel(abc.ABC):
    """大气密度模型抽象基类。

    力模型（如 ``DragModel``）通过 ``density(altitude)`` 获取大气密度，
    不关心具体实现（指数模型、MSISE-00 等）。
    """

    @abc.abstractmethod
    def density(self, altitude: float) -> float:
        """返回指定高度处的大气密度。

        Args:
            altitude: 几何高度，单位 km。

        Returns:
            大气密度，单位 kg/m³。
        """
        raise NotImplementedError
