"""动态坐标轴抽象基类。"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from .axes import Axes


class DynamicAxes(Axes, abc.ABC):
    """动态坐标轴抽象基类。

    依赖外部状态（位置、速度）的坐标轴。子类须先调用 ``update(t, state)``
    后，再使用 ``rotation_matrix(et)`` 或 ``rotation_and_rate(et)``。

    ``rotation_and_rate`` 沿用 Axes 基类的中心差分默认实现。
    """

    def __init__(self) -> None:
        self._updated = False

    @abc.abstractmethod
    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        """更新坐标轴的内部状态。

        Parameters
        ----------
        t : float
            历元（秒，自 J2000）。
        state : ndarray
            6 元素状态向量 [r, v]（ICRF）。
        """
        raise NotImplementedError

    def _require_updated(self) -> None:
        if not self._updated:
            raise RuntimeError(
                "DynamicAxes: 必须先调用 update(t, state) 后方能使用 "
                "rotation_matrix/rotation_and_rate。"
            )

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        self._require_updated()
        return self._compute_rotation_matrix(et)

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        self._require_updated()
        return super().rotation_and_rate(et)

    @abc.abstractmethod
    def _compute_rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """子类实现：在已 update 的前提下计算旋转矩阵。"""
        raise NotImplementedError

