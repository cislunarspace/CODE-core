"""坐标原点抽象基类。

定义参考坐标系位置基准的抽象接口。具体子类（如天体中心、航天器位置）
通过 ``state`` 方法给出该原点在 ICRF 中的绝对状态。
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt


class Origin(abc.ABC):
    """坐标原点抽象基类。

    描述一个参考坐标系的位置基准。子类必须实现 ``state``，返回该原点
    在 ICRF 中的绝对六维状态 ``[r, v]``。

    对于相对 ICRF 没有平移的原点（例如太阳系质心），返回零向量即可。
    """

    @abc.abstractmethod
    def state(self, et: float) -> npt.NDArray[np.floating]:
        """返回该原点在 ICRF 中的绝对状态。

        Args:
            et: SPICE 历书时（秒）。

        Returns:
            长度为 6 的数组，前 3 个元素为位置（km），后 3 个为速度（km/s）。
        """
        raise NotImplementedError

