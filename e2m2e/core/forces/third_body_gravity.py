"""第三体引力摄动模型。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from ..system import System


class ThirdBodyGravity(PhysicalModel):
    """第三体引力摄动加速度模型。

    对一个摄动天体 :math:`i`，返回相对原点天体 :math:`P_0` 的第三体摄动加速度：

    .. math::

        a = -\\mu_i \\left[
              \\frac{r - r_i}{|r - r_i|^3} + \\frac{r_i}{|r_i|^3}
            \\right]

    其中 :math:`r` 为航天器相对原点的位置，:math:`r_i` 为摄动天体相对原点
    的位置（由 ``system.get_body_position`` 自动以 ``system.origin`` 为观察者
    计算）。第一项为直接项（摄动天体对航天器的引力），第二项为间接项
    （扣除摄动天体对原点的引力），与 ``EphemerisDynamics`` 的第三体分支
    逐字对齐。

    Args:
        body: 摄动天体名称（如 ``'MOON'``、``'SUN'``）。
        mu: 引力参数（km³/s²）。为 ``None`` 时，在 ``compute_acceleration`` 中
            从 ``system.gravitational_parameter(body)`` 获取。
    """

    #: 防止除零的最小距离钳位（km，约 1 米），与 EphemerisDynamics.MIN_DISTANCE 一致。
    MIN_DISTANCE = 1e-6

    def __init__(self, body: str, mu: float | None = None) -> None:
        self._body = body.upper()
        self._mu = float(mu) if mu is not None else None

    @property
    def body(self) -> str:
        """摄动天体名称。"""
        return self._body

    @property
    def mu(self) -> float | None:
        """显式设置的引力参数；``None`` 表示从 system 获取。"""
        return self._mu

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """返回第三体摄动加速度，km/s²。

        Args:
            t: 时间，单位为 SPICE et（秒 past J2000）。
            state: 状态向量，形状至少为 ``(6,)``，前三个元素为位置。
            system: 当前动力学系统，提供 ``gravitational_parameter``、
                ``get_body_position`` 与 ``origin``。

        Returns:
            加速度向量，形状 ``(3,)``。

        距离低于 ``MIN_DISTANCE`` 时钳位，避免除零（发出 ``UserWarning``）。
        """
        mu = self._mu
        if mu is None:
            if system is None:
                raise ValueError(
                    "mu is None and system is None; cannot resolve gravitational_parameter"
                )
            mu = system.gravitational_parameter(self._body)

        r_sc = np.asarray(state, dtype=float)[:3]
        # 摄动天体相对原点（get_body_position 自动用 system.origin 作 observer）
        r_ob = np.asarray(system.get_body_position(self._body, t), dtype=float)
        # 摄动天体→航天器
        r_bsc = r_sc - r_ob

        r_bsc_norm = float(np.linalg.norm(r_bsc))
        r_ob_norm = float(np.linalg.norm(r_ob))
        if r_bsc_norm < self.MIN_DISTANCE:
            warnings.warn(
                f"Spacecraft at perturbing body {self._body} center "
                f"(|r_bsc|={r_bsc_norm:.2e} km), "
                f"clamping to MIN_DISTANCE={self.MIN_DISTANCE} km",
                stacklevel=2,
            )
            r_bsc_norm = float(self.MIN_DISTANCE)
        if r_ob_norm < self.MIN_DISTANCE:
            r_ob_norm = float(self.MIN_DISTANCE)

        return -mu * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
