"""EPPR ↔ J2000 双向转换器测试。

覆盖：往返闭合（多个无量纲时刻）、主星锚点（月球 → (1−μ,0,0)、地球 →
(−μ,0,0)）、批量变体逐行的主星锚点与往返闭合。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.coordinate import EPPRJ2000System
from e2m2e.data.constants import SECONDS_PER_DAY, Datum

pytestmark = pytest.mark.data

JD_TDB_AT_ET0 = 2451545.0
MU = Datum.DE421.mu


@pytest.mark.spice
@requires_spice
class TestEPPRJ2000System:
    ET0_JD = 2459000.0

    @property
    def et0(self) -> float:
        return (self.ET0_JD - JD_TDB_AT_ET0) * SECONDS_PER_DAY

    @pytest.fixture
    def converter(self, spice_manager, earth_moon_system):
        return EPPRJ2000System(cr3bp_system=earth_moon_system, spice=spice_manager)

    def test_round_trip(self, converter):
        states = [
            np.array([380000.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([350000.0, 12000.0, 8000.0, 0.2, 0.9, -0.1]),
        ]
        for t_nd in (0.0, 0.3, -0.7, 1.5):
            for state in states:
                eppr = converter.j2000_to_eppr(state, t_nd, self.et0)
                back = converter.eppr_to_j2000(eppr, t_nd, self.et0)
                np.testing.assert_allclose(back, state, rtol=1e-9, atol=1e-9)

    def test_primaries_are_fixed(self, converter, spice_manager):
        et = self.et0
        moon_relative = spice_manager.get_body_state("MOON", et, "J2000", "EARTH")
        moon_eppr = converter.j2000_to_eppr(moon_relative, 0.0, et)
        np.testing.assert_allclose(moon_eppr[:3], [1.0 - MU, 0.0, 0.0], atol=1e-10)

        earth_eppr = converter.j2000_to_eppr(np.zeros(6), 0.0, et)
        np.testing.assert_allclose(earth_eppr[:3], [-MU, 0.0, 0.0], atol=1e-10)

    def test_batch_preserves_primary_anchor(self, converter, spice_manager, earth_moon_system):
        """批量路径逐行给出物理正确的 EPPR 结果（月球锚点）且逐行往返闭合。"""
        times = np.array([0.0, 0.3, -0.7])
        characteristic_time = earth_moon_system.characteristic_time
        moon_states = np.array(
            [
                spice_manager.get_body_state(
                    "MOON", self.et0 + float(t) * characteristic_time, "J2000", "EARTH"
                )
                for t in times
            ]
        )

        eppr = converter.batch_j2000_to_eppr(moon_states, times, self.et0)
        np.testing.assert_allclose(
            eppr[:, :3], np.tile([1.0 - MU, 0.0, 0.0], (len(times), 1)), atol=1e-10
        )

        back = converter.batch_eppr_to_j2000(eppr, times, self.et0)
        np.testing.assert_allclose(back, moon_states, rtol=1e-9, atol=1e-9)
