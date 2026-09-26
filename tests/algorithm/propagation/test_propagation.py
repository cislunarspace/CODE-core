"""algorithm/propagation 模块测试。"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.propagation import _extract_bodies, propagate_orbit
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.integrator


class TestExtractBodies:
    def test_single_body(self):
        cfg = {
            "version": 1,
            "forces": [
                {
                    "name": "g",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH"},
                }
            ],
        }
        assert _extract_bodies(cfg) == ["EARTH"]

    def test_unique_and_uppercase(self):
        cfg = {
            "version": 1,
            "forces": [
                {
                    "name": "g1",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "earth"},
                },
                {
                    "name": "g2",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH"},
                },
                {
                    "name": "g3",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "MOON"},
                },
            ],
        }
        assert _extract_bodies(cfg) == ["EARTH", "MOON"]

    def test_empty_forces(self):
        assert _extract_bodies({"version": 1, "forces": []}) == []

    def test_missing_body(self):
        assert _extract_bodies({"version": 1, "forces": [{"params": {"mu": 1.0}}]}) == []


@pytest.mark.spice
@requires_spice
class TestPropagateOrbit:
    def test_default_three_body(self, spice_manager, reference_epoch):
        # 用默认三体配置（_DEFAULT_FORCE_CONFIG）：地球点质量 + 月球/太阳
        # ThirdBodyGravity。若第三方天体被误配成朝向地心的
        # PointMassGravity，太阳 mu 主导会使任何合理初值步长坍缩，
        # 本测试即对该失效模式的回归保护。
        initial_state = np.array(
            [
                7000.0,  # km
                0.0,
                0.0,
                0.0,  # km/s
                7.7,
                0.0,
            ]
        )
        result = propagate_orbit(
            initial_state=initial_state,
            epoch=reference_epoch,
            duration=86400.0,
            output_step=3600.0,
        )
        assert len(result.ephemeris) == 25  # 0..24h inclusive
        assert result.ephemeris.position_km.shape == (25, 3)
        assert result.ephemeris.velocity_mps.shape == (25, 3)
        assert result.status is ConvergenceState.CONVERGED
        assert result.cause is FailureCause.NONE
        assert result.message == "任务完成"
        assert not np.allclose(result.ephemeris.position_km[0], result.ephemeris.position_km[-1])

    def test_epoch_tuple(self, spice_manager):
        # 非奇异初值（地球点质量即可，聚焦 epoch 元组解析）
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.7, 0.0])
        force_config = {
            "version": 1,
            "forces": [
                {
                    "name": "earth",
                    "type": "PointMassGravity",
                    "params": {"body": "EARTH", "mu": 398600.435507},
                },
            ],
        }
        result = propagate_orbit(
            initial_state=initial_state,
            epoch=[2025, 6, 21, 11, 0, 6.0],
            duration=3600.0,
            force_config=force_config,
            output_step=3600.0,
        )
        assert len(result.ephemeris) == 2

    def test_invalid_initial_state_shape(self):
        with pytest.raises(ValueError, match="initial_state"):
            propagate_orbit(
                initial_state=np.zeros(5),
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
            )

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration"):
            propagate_orbit(
                initial_state=np.zeros(6),
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_custom_force_config(self, spice_manager, reference_epoch):
        force_config = {
            "version": 1,
            "forces": [
                {
                    "name": "earth",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH", "mu": 398600.435507},
                },
                {
                    "name": "moon",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "MOON", "mu": 4902.800118},
                },
            ],
        }
        result = propagate_orbit(
            initial_state=np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
            epoch=reference_epoch,
            duration=86400.0,
            force_config=force_config,
            output_step=3600.0,
        )
        assert len(result.ephemeris) == 25

    def test_backward_roundtrip(self, spice_manager, reference_epoch):
        # 反向 6h 得到 prev 态，再用 prev 态正向 6h 应闭合回初值。
        s0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.7, 0.0])
        bwd = propagate_orbit(
            initial_state=s0,
            epoch=reference_epoch,
            duration=21600.0,
            direction="backward",
            output_step=3600.0,
        )
        s_prev = np.concatenate(
            [bwd.ephemeris.position_km[-1], bwd.ephemeris.velocity_mps[-1] / 1000.0]
        )
        et0 = spice_manager.utc_to_et(reference_epoch)
        utc_prev = spice_manager.et_to_utc(et0 - 21600.0)
        fwd = propagate_orbit(
            initial_state=s_prev,
            epoch=utc_prev,
            duration=21600.0,
            output_step=3600.0,
        )
        assert np.linalg.norm(fwd.ephemeris.position_km[-1] - s0[:3]) < 1e-2
        assert np.linalg.norm(fwd.ephemeris.velocity_mps[-1] / 1000.0 - s0[3:]) < 1e-6

    def test_backward_time_axis(self, spice_manager, reference_epoch):
        s0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.7, 0.0])
        result = propagate_orbit(
            initial_state=s0,
            epoch=reference_epoch,
            duration=21600.0,
            direction="backward",
            output_step=3600.0,
        )
        assert len(result.ephemeris) == 7
        jd = result.ephemeris.times_jd_tdb
        assert np.all(np.diff(jd) < 0)
        et0 = spice_manager.utc_to_et(reference_epoch)
        assert jd[0] == pytest.approx(2451545.0 + et0 / 86400.0)
        assert jd[-1] == pytest.approx(2451545.0 + (et0 - 21600.0) / 86400.0)

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            propagate_orbit(
                initial_state=np.zeros(6),
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
                direction="sideways",
            )
