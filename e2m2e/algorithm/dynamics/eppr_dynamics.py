"""Earth-Moon EPPR 系内的时空动力学（``EPPR_Dynamics``）。

状态与 CR3BP 同为无量纲：位置以瞬时地月距离 d(t) 为单位（地球在 (−μ,0,0)、月球
在 (1−μ,0,0)），时间以 t_c 为单位。运动方程把两主星引力与旋转-脉动系全部惯性项
一并写出（帧量由 ``EPPRFrameModel`` 从真实星历供给）：

.. math::

    \\ddot{\\xi} = \\rho^{-3} g
        + W \\times (-2\\xi' + v_s \\xi + \\xi \\times W)
        + \\xi \\times \\dot{W}
        + a_{bc} + r_s \\xi + v_s \\xi'

其中 :math:`\\rho = d/l^*`、:math:`v_s = -2 t_c \\dot d / d`、
:math:`r_s = -t_c^2 \\ddot d / d`、:math:`W = t_c \\omega`、
:math:`\\dot W = t_c^2 \\dot\\omega`、:math:`a_{bc} = -(t_c^2/d) R^\\top a_{bc}^{\\rm J2000}`，
:math:`g` 为两主星在 ξ 单位下的牛顿引力。平面退化（W=(0,0,ω)）时逐项化为
``crates/e2m2e-hjb-dynamics`` 的 ``EphemerisPlanar`` 平面漂移项；圆形星历
（d≡lstar、ω≡1/t_c、脉冲项为零）时精确退化为 CR3BP。

时变系统无 Jacobi 积分、无可解析变分矩阵：``compute_jacobi_constant`` 与
``with_stm=True`` 均抛 ``NotImplementedError``（``BCR4BP_Dynamics`` 先例）。
传播走 ``Dynamics`` 基类 scipy 路径，不接 Rust 后端。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .dynamics import Dynamics
from .eppr_system import EPPRSystem


class EPPR_Dynamics(Dynamics):
    """EPPR 系内的时空运动方程（无量纲时间 τ）。

    Attributes:
        system: ``EPPRSystem``，提供 μ、特征尺度与帧量模型。
    """

    system: EPPRSystem

    def __init__(self, system: EPPRSystem) -> None:
        """初始化 EPPR 动力学。

        Args:
            system: ``EPPRSystem`` 对象。
        """
        super().__init__(system)

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回 EPPR 运动方程函数；EPPR 不提供 STM 版本。"""
        if with_stm:
            raise NotImplementedError("EPPR 为时变系统，不提供状态转移矩阵（无可解析变分矩阵）")
        return self.equations_of_motion

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """六维状态向量的运动方程（显式含时间 τ）。

        Args:
            t: 无量纲时间 τ（帧量随 τ 变化，不能忽略）。
            state: 状态向量 ``[x, y, z, vx, vy, vz]``（无量纲，EPPR 系）。

        Returns:
            状态导数 ``[vx, vy, vz, ax, ay, az]``。
        """
        system = self.system
        t_c = system.characteristic_time
        lstar = system.characteristic_length
        frame_state = system.frame_model.frame_state(system.et0 + t * t_c)

        rho = frame_state.distance / lstar
        g_scale = rho**-3.0
        v_scale = -2.0 * (t_c * frame_state.distance_rate) / frame_state.distance
        r_scale = -(t_c * t_c * frame_state.distance_accel) / frame_state.distance
        omega = t_c * frame_state.angular_velocity
        omega_dot = t_c * t_c * frame_state.angular_acceleration
        barycenter_accel = -(frame_state.rotation.T @ frame_state.barycenter_accel) * (
            t_c * t_c / frame_state.distance
        )

        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:], dtype=float)
        mu = float(system.mu)

        r1 = position - np.array([-mu, 0.0, 0.0])
        r2 = position - np.array([1.0 - mu, 0.0, 0.0])
        r1_norm = max(float(np.linalg.norm(r1)), self.MIN_DISTANCE)
        r2_norm = max(float(np.linalg.norm(r2)), self.MIN_DISTANCE)
        gravity = -(1.0 - mu) * r1 / r1_norm**3 - mu * r2 / r2_norm**3

        acceleration = (
            g_scale * gravity
            + np.cross(omega, -2.0 * velocity + v_scale * position + np.cross(position, omega))
            + np.cross(position, omega_dot)
            + barycenter_accel
            + r_scale * position
            + v_scale * velocity
        )
        return np.concatenate([velocity, acceleration])

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """EPPR 是时变系统，无 Jacobi 积分。"""
        raise NotImplementedError("EPPR 是时变系统，无 Jacobi 积分")

    def _handle_jacobi(self, states: np.ndarray, out: dict[str, Any]) -> dict[str, Any]:
        """EPPR 无 Jacobi 积分，``with_jacobi=True`` 时报错。"""
        raise NotImplementedError("EPPR 是时变系统，无 Jacobi 积分")

    def __str__(self):
        return f"EPPR_Dynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        return (
            f"EPPR_Dynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )
