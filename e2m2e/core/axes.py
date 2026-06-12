"""坐标轴抽象基类。"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt


class Axes(abc.ABC):
    """坐标轴抽象基类。

    ``rotation_matrix(et)`` 返回的矩阵 ``R`` 满足 ``r_icrf = R @ r_axes``。
    ``rotation_and_rate(et)`` 返回 ``(R, Rdot)``，满足
    ``v_icrf = R @ v_axes + Rdot @ r_axes``。
    """

    _DEFAULT_RATE_STEP = 1.0e-5

    @abc.abstractmethod
    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """返回从该坐标轴到 ICRF/J2000 的 3x3 旋转矩阵。"""
        raise NotImplementedError

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """返回从该坐标轴到 ICRF/J2000 的旋转矩阵及其时间导数。

        子类可覆盖该方法提供解析或内核给出的 ``Rdot``。若旧子类只覆盖
        ``angular_velocity``，默认实现会按兼容角速度语义构造 ``Rdot``。
        否则使用中心差分作为通用路径。
        """
        rotation = self.rotation_matrix(et)
        if type(self).angular_velocity is not Axes.angular_velocity:
            omega = self.angular_velocity(et)
            omega_skew = np.array(
                [
                    [0.0, -omega[2], omega[1]],
                    [omega[2], 0.0, -omega[0]],
                    [-omega[1], omega[0], 0.0],
                ]
            )
            return rotation, omega_skew @ rotation

        step = float(getattr(self, "_time_step", self._DEFAULT_RATE_STEP))
        before = self.rotation_matrix(et - step)
        after = self.rotation_matrix(et + step)
        rate = (after - before) / (2.0 * step)
        return rotation, rate

    def state_transform_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """返回 6x6 状态变换矩阵。"""
        rotation, rate = self.rotation_and_rate(et)
        transform = np.zeros((6, 6))
        transform[:3, :3] = rotation
        transform[3:, :3] = rate
        transform[3:, 3:] = rotation
        return transform

    def angular_velocity(self, et: float) -> npt.NDArray[np.floating]:
        """返回 ICRF 中观测到的该坐标轴角速度。

        这是兼容 API；高精度状态转换优先使用 ``rotation_and_rate``。
        """
        rotation, rate = self.rotation_and_rate(et)
        omega_skew = rate @ rotation.T
        return np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
