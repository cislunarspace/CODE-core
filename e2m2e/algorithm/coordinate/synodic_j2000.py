"""基于 ``CoordinateSystem`` 的 synodic ↔ J2000 转换器。

synodic 坐标系 = (SynodicAxes, CelestialBodyOrigin("EARTH"))
J2000 坐标系   = (ICRSAxes,      CelestialBodyOrigin("EARTH"))

二者原点相同 (地心) 因此 ``CoordinateSystem.transform_state`` 仅旋转轴向；
无量纲 synodic 状态在调用前后通过 ``l_c`` / ``t_c`` 量纲化与无量纲化。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..dynamics.cr3bp_system import CR3BP_System
from .coordinate_system import CoordinateSystem
from .standard_axes import ICRSAxes
from .standard_origins import CelestialBodyOrigin
from .synodic_axes import SynodicAxes

_TU_SECONDS_DEFAULT = 4.34811305 * 86400.0


class SynodicJ2000System:
    """基于 ``CoordinateSystem`` 的 synodic ↔ J2000 转换器。"""

    def __init__(self, cr3bp_system: CR3BP_System, spice) -> None:
        self.cr3bp_system = cr3bp_system
        self.spice = spice
        self.synodic_axes = SynodicAxes(spice)
        earth_origin = CelestialBodyOrigin("EARTH", spice)
        self.synodic_cs = CoordinateSystem(self.synodic_axes, earth_origin)
        self.j2000_cs = CoordinateSystem(ICRSAxes(), earth_origin)

    def _get_time_unit(self) -> float:
        if (
            hasattr(self.cr3bp_system, "characteristic_time")
            and self.cr3bp_system.characteristic_time is not None
        ):
            return self.cr3bp_system.characteristic_time
        return _TU_SECONDS_DEFAULT

    @staticmethod
    def _bary_to_earth_offset(mu: float) -> npt.NDArray[np.floating]:
        return np.array([mu, 0.0, 0.0])

    def synodic_to_j2000(
        self, state_syn: npt.ArrayLike, t_syn: float, et0: float
    ) -> npt.NDArray[np.floating]:
        state_syn = np.asarray(state_syn, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()
        et = et0 + t_syn * t_c
        l_c = self.synodic_axes.characteristic_length(et)
        # 把无量纲"质心系"位置先平移到"地心 + moon-earth 轴"位置描述
        offset = self._bary_to_earth_offset(mu)
        position_in = (state_syn[:3] + offset) * l_c
        velocity_in = state_syn[3:] * l_c / t_c
        state_in = np.concatenate([position_in, velocity_in])
        return self.j2000_cs.transform_state(state_in, self.synodic_cs, self.j2000_cs, et)

    def j2000_to_synodic(
        self, state_j2000: npt.ArrayLike, t_syn: float, et0: float
    ) -> npt.NDArray[np.floating]:
        state_j2000 = np.asarray(state_j2000, dtype=float)
        mu = self.cr3bp_system.mu
        t_c = self._get_time_unit()
        et = et0 + t_syn * t_c
        l_c = self.synodic_axes.characteristic_length(et)
        state_earth_syn = self.synodic_cs.transform_state(
            state_j2000, self.j2000_cs, self.synodic_cs, et
        )
        offset = self._bary_to_earth_offset(mu)
        r_syn = state_earth_syn[:3] / l_c - offset
        v_syn = state_earth_syn[3:] * t_c / l_c
        return np.concatenate([r_syn, v_syn])

    def batch_synodic_to_j2000(
        self,
        states_syn: npt.ArrayLike,
        t_syn_arr: npt.ArrayLike,
        et0: float,
    ) -> npt.NDArray[np.floating]:
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
    ) -> npt.NDArray[np.floating]:
        states_j2000 = np.asarray(states_j2000, dtype=float)
        t_syn_arr = np.asarray(t_syn_arr, dtype=float)
        n = len(t_syn_arr)
        results = np.empty((n, 6))
        for i in range(n):
            results[i] = self.j2000_to_synodic(states_j2000[i], t_syn_arr[i], et0)
        return results

