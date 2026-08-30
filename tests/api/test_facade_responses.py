"""Facade 公开响应的结果翻译与 JSON 序列化测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.api.facade import Facade
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.maneuver import ManeuverTable
from e2m2e.data.types.sk_statistic import SKStatistic
from e2m2e.data.types.trajectory import EphemerisTable

pytestmark = pytest.mark.interface


def _make_ephemeris(n: int = 3, with_jd: bool = False) -> EphemerisTable:
    return EphemerisTable(
        year=np.full(n, 2024, dtype=int),
        month=np.full(n, 1, dtype=int),
        day=np.full(n, 1, dtype=int),
        hour=np.arange(n, dtype=int),
        minute=np.zeros(n, dtype=int),
        second=np.zeros(n, dtype=float),
        position_km=np.arange(n * 3, dtype=float).reshape(n, 3),
        velocity_mps=np.full((n, 3), 1000.0),
        synodic_position=np.full((n, 3), 0.5),
        times_jd_tdb=np.linspace(2460310.0, 2460311.0, n) if with_jd else None,
    )


def _make_design_result(*, with_system: bool = True):
    system = SimpleNamespace(mu=Datum.DE421.mu) if with_system else None
    return SimpleNamespace(
        orbit_type="DRO",
        epoch_utc="2024-01-01T00:00:00.000",
        duration_day=365.25,
        output_step_sec=3600.0,
        initial_state=np.zeros(6),
        ephemeris=_make_ephemeris(),
        cr3bp_orbit=SimpleNamespace(
            states=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]]),
            times=np.array([0.0, 1.234]),
            system=system,
        ),
        cr3bp_jacobi=3.16,
        correction=SimpleNamespace(iterations=4),
        correction_method="two_level",
        force_config={"sun_body": 1},
        status="converged",
        cause="none",
        message="任务完成",
        drift_e=None,
        drift_aop_deg=None,
        drift_rp_km=None,
        secular_aop_rate_deg_per_year=None,
    )


def _make_control_result(*, controlled: bool):
    return SimpleNamespace(
        num_failed=1,
        status=ConvergenceState.CONVERGED if controlled else ConvergenceState.FAILED,
        cause=FailureCause.NONE if controlled else FailureCause.UNKNOWN,
        message="任务完成" if controlled else "全部蒙特卡洛样本失败",
        sk_statistic=SKStatistic(rows=np.zeros((2, 3)), num_failed=1),
        maneuvers=ManeuverTable(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
        controlled_ephemeris=_make_ephemeris(n=2) if controlled else None,
    )


class TestTransferResponse:
    def test_serializes_nested_numpy_details(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        def fake_transfer(*args, **kwargs):
            return SimpleNamespace(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="任务完成",
                transfer_type="HMN",
                delta_v=3.1,
                trajectory=np.array([[1.0] * 6]),
                trajectory_times=np.array([0.0]),
                state_frame="synodic_barycentric_km",
                details={
                    "array": np.array([1.0, 2.0]),
                    "nested": {"tuple": (np.array([3.0]), 4.0)},
                },
            )

        monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer)
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )

        assert response.trajectory == [[1.0] * 6]
        assert response.trajectory_times == [0.0]
        assert response.state_frame == "synodic_barycentric_km"
        assert response.details == {"array": [1.0, 2.0], "nested": {"tuple": [[3.0], 4.0]}}

    def test_serializes_dataclass_details(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        def fake_transfer(*args, **kwargs):
            return SimpleNamespace(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="任务完成",
                transfer_type="HMN",
                delta_v=3.1,
                trajectory=None,
                trajectory_times=None,
                state_frame="synodic_barycentric_km",
                details=ManeuverTable(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
            )

        monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer)
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.details["mjd_tdb"] == [60000.0]


class TestDesignResponse:
    def test_translates_geometry_and_ephemeris(self, monkeypatch):
        import e2m2e.algorithm.design as design

        monkeypatch.setattr(design, "design_orbit", lambda *args, **kwargs: _make_design_result())
        response = Facade().design_orbit(orbit_type="DRO")

        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert response.initial_state == [0.0] * 6
        assert response.states[1] == [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
        assert response.times == [0.0, 1.234]
        assert response.mu == pytest.approx(Datum.DE421.mu)
        assert response.ephemeris is not None
        assert response.ephemeris["position_km"] == [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
        assert response.ephemeris["velocity_mps"] == [[1000.0, 1000.0, 1000.0]] * 3
        assert response.ephemeris["times_jd_tdb"] is None
        assert set(response.ephemeris) == {
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "position_km",
            "velocity_mps",
            "synodic_position",
            "times_jd_tdb",
        }

    def test_translates_ephemeris_jd_when_populated(self, monkeypatch):
        import e2m2e.algorithm.design as design

        result = _make_design_result()
        result.ephemeris = _make_ephemeris(with_jd=True)
        monkeypatch.setattr(design, "design_orbit", lambda *args, **kwargs: result)
        response = Facade().design_orbit(orbit_type="DRO")

        assert response.ephemeris is not None
        assert response.ephemeris["times_jd_tdb"] == pytest.approx(
            [2460310.0, 2460310.5, 2460311.0]
        )

    def test_allows_missing_system_mu(self, monkeypatch):
        import e2m2e.algorithm.design as design

        monkeypatch.setattr(
            design, "design_orbit", lambda *args, **kwargs: _make_design_result(with_system=False)
        )
        assert Facade().design_orbit(orbit_type="DRO").mu is None

    def test_translates_correction_method(self, monkeypatch):
        import e2m2e.algorithm.design as design

        result = _make_design_result()
        result.correction_method = "segmented"
        monkeypatch.setattr(design, "design_orbit", lambda *args, **kwargs: result)
        response = Facade().design_orbit(orbit_type="DRO")

        assert response.correction_method == "segmented"


class TestControlResponse:
    @pytest.mark.parametrize(
        ("controlled", "status", "cause"),
        [
            (True, ConvergenceState.CONVERGED, FailureCause.NONE),
            (False, ConvergenceState.FAILED, FailureCause.UNKNOWN),
        ],
    )
    def test_translates_control_result(self, monkeypatch, controlled, status, cause):
        import e2m2e.algorithm.station_keeping as station_keeping

        monkeypatch.setattr(
            station_keeping,
            "control_orbit",
            lambda *args, **kwargs: _make_control_result(controlled=controlled),
        )
        response = Facade().control_orbit(input_ephemeris="x", mu=Datum.DE421.mu)

        assert response.status is status
        assert response.cause is cause
        assert response.num_failed == 1
        assert response.mu == pytest.approx(Datum.DE421.mu)
        if controlled:
            assert response.controlled_ephemeris is not None
            assert response.controlled_ephemeris["synodic_position"] == [[0.5, 0.5, 0.5]] * 2
            assert response.controlled_ephemeris["times_jd_tdb"] is None
        else:
            assert response.controlled_ephemeris is None

        assert response.sk_statistic == {"rows": [[0.0] * 3] * 2, "num_failed": 1}
        assert response.maneuvers == {"mjd_tdb": [60000.0], "delta_v_mps": [1.0]}
