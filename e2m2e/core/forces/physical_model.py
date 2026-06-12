"""力模型抽象基类。"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from e2m2e.core.system import System


class PhysicalModel(abc.ABC):
    """物理力模型抽象基类。

    力模型以纯函数接口提供加速度。所有坐标约定都在
    ``system.coordinate_system`` 下完成；需要非默认坐标系计算的子类
    应自行调用 ``system.transform()``。
    """

    @abc.abstractmethod
    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """返回状态在 ``system.coordinate_system`` 下的加速度。

        Args:
            t: 时间，单位为 SPICE et（秒 past J2000）。
            state: 状态向量，形状至少为 ``(6,)``，前三个元素为位置。
            system: 当前动力学系统，提供坐标系与天体参数。

        Returns:
            加速度向量，形状 ``(3,)``，单位与 ``system.unit_system`` 一致。
        """
        raise NotImplementedError
