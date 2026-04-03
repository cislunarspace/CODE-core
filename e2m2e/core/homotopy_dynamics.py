"""
同伦星历动力学模块

通过摄动天体逐步引入的同伦参数 λ，将动力学模型从基础模型（如 E+M）
平滑过渡到完整星历模型（E+M+S），实现同伦法轨道修正。

同伦方程物理含义：
    λ=0: 仅 base_bodies 的引力（接近 CRTBP 的星历等效）
    λ=1: 所有天体的完整引力（完整星历模型）

加速度公式：
    a(r, t, λ) = Σ_{b ∈ base} a_b(r, t) + λ · Σ_{p ∈ perturbation} a_p(r, t)

其中 a_b 和 a_p 均在 J2000 惯性系下计算，量纲一致 (km/s²)。
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .ephemeris_dynamics import EphemerisDynamics
from .ephemeris_system import EphemerisSystem


class HomotopyEphemerisDynamics(EphemerisDynamics):
    """基于摄动天体逐步引入的同伦星历动力学

    继承 EphemerisDynamics，将 system.bodies 分为 base_bodies 和 perturbation_bodies
    两组。base_bodies 的引力始终以满值参与计算，perturbation_bodies 的引力乘以
    同伦参数 λ。

    Args:
        system: EphemerisSystem 对象
        base_bodies: 基础天体列表（如 ["EARTH", "MOON"]），始终满引力
        perturbation_bodies: 摄动天体列表（如 ["SUN"]），引力乘以 λ。
            若为 None，自动取 system.bodies 中不在 base_bodies 的天体。
        homotopy_param: 同伦参数 λ ∈ [0, 1]
    """

    def __init__(
        self,
        system: EphemerisSystem,
        base_bodies: List[str],
        perturbation_bodies: Optional[List[str]] = None,
        homotopy_param: float = 0.0,
    ) -> None:
        super().__init__(system)
        self.base_bodies = list(base_bodies)
        if perturbation_bodies is not None:
            self.perturbation_bodies = list(perturbation_bodies)
        else:
            self.perturbation_bodies = [
                b for b in system.bodies if b not in self.base_bodies
            ]
        self.lam = float(homotopy_param)

    def _compute_acceleration(
        self, r_sc: npt.NDArray, et: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """计算加速度及其位置 Jacobian

        Returns:
            (acc, dacc_dr): 加速度 3-vector 和 3x3 Jacobian
        """
        acc = np.zeros(3)
        dacc_dr = np.zeros((3, 3))

        for body in self.base_bodies:
            gm = self.system.spice.get_gm(body)
            if body == self.system.origin:
                r_norm = np.linalg.norm(r_sc)
                acc -= gm * r_sc / r_norm**3
                dacc_dr -= gm * (
                    np.eye(3) / r_norm**3 - 3.0 * np.outer(r_sc, r_sc) / r_norm**5
                )
            else:
                r_ob = self.system.spice.get_body_position(
                    body, et, self.system.frame, self.system.origin
                )
                r_bsc = r_sc - r_ob
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
                dacc_dr -= gm * (
                    np.eye(3) / r_bsc_norm**3
                    - 3.0 * np.outer(r_bsc, r_bsc) / r_bsc_norm**5
                )

        for body in self.perturbation_bodies:
            gm = self.system.spice.get_gm(body)
            r_ob = self.system.spice.get_body_position(
                body, et, self.system.frame, self.system.origin
            )
            r_bsc = r_sc - r_ob
            r_bsc_norm = np.linalg.norm(r_bsc)
            r_ob_norm = np.linalg.norm(r_ob)
            acc -= self.lam * gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
            dacc_dr -= self.lam * gm * (
                np.eye(3) / r_bsc_norm**3
                - 3.0 * np.outer(r_bsc, r_bsc) / r_bsc_norm**5
            )

        return acc, dacc_dr

    def equations_of_motion(
        self, et: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        r_sc = state[:3]
        v_sc = state[3:]
        acc, _ = self._compute_acceleration(r_sc, et)
        return np.concatenate([v_sc, acc])

    def equations_with_stm(
        self, et: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        state = augmented_state[:6]
        r_sc = state[:3]
        stm = augmented_state[6:].reshape((6, 6))

        acc, dacc_dr = self._compute_acceleration(r_sc, et)

        state_deriv = np.concatenate([state[3:], acc])

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = dacc_dr

        stm_dot = A @ stm

        return np.concatenate([state_deriv, stm_dot.flatten()])
