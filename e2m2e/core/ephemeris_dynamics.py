from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .dynamics import Dynamics
from .ephemeris_system import EphemerisSystem


class EphemerisDynamics(Dynamics):
    STM_DIMENSION = 42

    def __init__(self, system: EphemerisSystem) -> None:
        self.system = system
        self.integrator = "DOP853"
        self.rtol = 1e-12
        self.atol = 1e-12
        self.max_step = 60.0
        self.last_trajectory = None
        self.last_stm = None
        self.cross_section_tolerance = 1e-8
        self.last_crossing = None
        self.jacobi_history = []
        self.jacobi_error = 0.0
        self.initialized = True

    def equations_of_motion(
        self, et: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        r_sc = state[:3]
        v_sc = state[3:]

        acc = np.zeros(3)
        for body in self.system.bodies:
            gm = self.system.spice.get_gm(body)
            if body == self.system.origin:
                r_norm = np.linalg.norm(r_sc)
                acc -= gm * r_sc / r_norm**3
            else:
                r_ob = self.system.spice.get_body_position(
                    body, et, self.system.frame, self.system.origin
                )
                r_bsc = r_sc - r_ob
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)

        return np.concatenate([v_sc, acc])

    def equations_with_stm(
        self, et: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        state = augmented_state[:6]
        r_sc = state[:3]

        stm = augmented_state[6:].reshape((6, 6))

        acc = np.zeros(3)
        dacc_dr = np.zeros((3, 3))

        for body in self.system.bodies:
            gm = self.system.spice.get_gm(body)
            if body == self.system.origin:
                r_norm = np.linalg.norm(r_sc)
                acc -= gm * r_sc / r_norm**3
                dacc_dr -= gm * (np.eye(3) / r_norm**3 - 3.0 * np.outer(r_sc, r_sc) / r_norm**5)
            else:
                r_ob = self.system.spice.get_body_position(
                    body, et, self.system.frame, self.system.origin
                )
                r_bsc = r_sc - r_ob
                r_bsc_norm = np.linalg.norm(r_bsc)
                r_ob_norm = np.linalg.norm(r_ob)
                acc -= gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)
                dacc_dr -= gm * (
                    np.eye(3) / r_bsc_norm**3 - 3.0 * np.outer(r_bsc, r_bsc) / r_bsc_norm**5
                )

        state_deriv = np.concatenate([state[3:], acc])

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = dacc_dr

        stm_dot = A @ stm

        return np.concatenate([state_deriv, stm_dot.flatten()])

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        span_duration = abs(t_span[1] - t_span[0])
        if span_duration > 0:
            max_step = min(self.max_step, span_duration / 10.0)
        else:
            max_step = self.max_step

        if with_stm:
            initial_stm = np.eye(6).flatten()
            augmented_state = np.concatenate([np.asarray(initial_state, dtype=float), initial_stm])

            result = solve_ivp(
                self.equations_with_stm,
                t_span,
                augmented_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=max_step,
            )

            states = result.y[:6, :]
            n_times = states.shape[1]
            stm_flat = result.y[6:, :]
            stm_matrices = np.zeros((6, 6, n_times))
            for k in range(n_times):
                stm_matrices[:, :, k] = stm_flat[:, k].reshape(6, 6)
            self.last_trajectory = (result.t, states.T)
            self.last_stm = stm_matrices

            return {
                "time": result.t,
                "states": states,
                "stm": stm_matrices,
            }
        else:
            result = solve_ivp(
                self.equations_of_motion,
                t_span,
                np.asarray(initial_state, dtype=float),
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=max_step,
            )

            states = result.y

            self.last_trajectory = (result.t, states.T)

            return {
                "time": result.t,
                "states": states,
            }
