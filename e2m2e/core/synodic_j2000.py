from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .system import CR3BP_System

_TU_SECONDS_DEFAULT = 4.34811305 * 86400


class SynodicJ2000Transformation:
    def __init__(self, cr3bp_system: CR3BP_System, spice) -> None:
        self.cr3bp_system = cr3bp_system
        self.spice = spice

    def _get_time_unit(self) -> float:
        if (
            hasattr(self.cr3bp_system, "characteristic_time")
            and self.cr3bp_system.characteristic_time is not None
        ):
            return self.cr3bp_system.characteristic_time
        return _TU_SECONDS_DEFAULT

    def _build_rotation_matrix(self, r_m: npt.NDArray, v_m: npt.NDArray) -> npt.NDArray:
        e1 = r_m / np.linalg.norm(r_m)
        h = np.cross(r_m, v_m)
        e3 = h / np.linalg.norm(h)
        e2 = np.cross(e3, e1)
        return np.column_stack([e1, e2, e3])

    def _get_moon_frame(self, et: float):
        moon_state = self.spice.get_body_state("MOON", et, "J2000", "EARTH")
        r_m = moon_state[:3]
        v_m = moon_state[3:]
        l_c = np.linalg.norm(r_m)
        R = self._build_rotation_matrix(r_m, v_m)
        omega = np.cross(r_m, v_m) / np.dot(r_m, r_m)
        return r_m, v_m, l_c, R, omega

    def synodic_to_j2000(self, state_syn: npt.ArrayLike, t_syn: float, et0: float) -> npt.NDArray:
        state_syn = np.asarray(state_syn, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()

        et = et0 + t_syn * t_c
        r_m, v_m, l_c, R, omega = self._get_moon_frame(et)

        r_dim = state_syn[:3] * l_c
        r_from_earth = r_dim - np.array([-mu, 0.0, 0.0]) * l_c
        r_j2000 = R @ r_from_earth

        v_dim = state_syn[3:] * l_c / t_c
        v_j2000 = R @ v_dim + np.cross(omega, r_j2000)

        return np.concatenate([r_j2000, v_j2000])

    def j2000_to_synodic(self, state_j2000: npt.ArrayLike, t_syn: float, et0: float) -> npt.NDArray:
        state_j2000 = np.asarray(state_j2000, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()

        et = et0 + t_syn * t_c
        r_m, v_m, l_c, R, omega = self._get_moon_frame(et)

        r_from_earth = R.T @ state_j2000[:3]
        r_dim = r_from_earth + np.array([-mu, 0.0, 0.0]) * l_c
        r_syn = r_dim / l_c

        v_from_earth = R.T @ (state_j2000[3:] - np.cross(omega, state_j2000[:3]))
        v_syn = v_from_earth * t_c / l_c

        return np.concatenate([r_syn, v_syn])

    def batch_synodic_to_j2000(
        self,
        states_syn: npt.ArrayLike,
        t_syn_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray:
        states_syn = np.asarray(states_syn, dtype=float)
        t_syn_arr = np.asarray(t_syn_arr, dtype=float)
        n = len(t_syn_arr)
        results = np.empty((n, 6))
        for i in range(n):
            results[i] = self.synodic_to_j2000(states_syn[i], t_syn_arr[i], et0)
        return results

    def batch_j2000_to_synodic(
        self,
        states_j2000: npt.ArrayLike,
        t_syn_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray:
        states_j2000 = np.asarray(states_j2000, dtype=float)
        t_syn_arr = np.asarray(t_syn_arr, dtype=float)
        n = len(t_syn_arr)
        results = np.empty((n, 6))
        for i in range(n):
            results[i] = self.j2000_to_synodic(states_j2000[i], t_syn_arr[i], et0)
        return results
