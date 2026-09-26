"""Facade 公开响应的结果翻译与 JSON 序列化测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.api.facade import Facade
from e2m2e.data.constants import Datum
from e2m2e.data.constants.bodies import MOON
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
    @staticmethod
    def _fake_transfer_result(**overrides):
        """带机动事件的算法层结果替身（SimpleNamespace，字段随契约演进）。"""
        from e2m2e.algorithm.transfer import ManeuverEvent

        base = dict(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="任务完成",
            transfer_type="LGA",
            delta_v=3.1,
            trajectory=np.array([[1.0] * 6]),
            trajectory_times=np.array([0.0]),
            trajectory_gcrs_km=np.array([[2.0] * 6]),
            state_frame="synodic_barycentric_km",
            maneuver_events=(
                ManeuverEvent(kind="departure", t_sec=0.0, dv_km_s=3.1),
                ManeuverEvent(kind="perilune", t_sec=100.0, dv_km_s=0.0),
                ManeuverEvent(kind="arrival", t_sec=200.0, dv_km_s=0.4),
            ),
            details={
                "array": np.array([1.0, 2.0]),
                "nested": {"tuple": (np.array([3.0]), 4.0)},
            },
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_serializes_nested_numpy_details(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result()
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )

        assert response.trajectory == [[1.0] * 6]
        assert response.trajectory_times == [0.0]
        assert response.state_frame == "synodic_barycentric_km"
        assert response.details == {"array": [1.0, 2.0], "nested": {"tuple": [[3.0], 4.0]}}

    def test_maps_gcrs_segment_to_response(self, monkeypatch):
        """惯性段 ndarray → 响应 list（#584）；信封序列化无 ndarray 残留。"""
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result(
            trajectory_gcrs_km=np.array([[7.0] * 6, [8.0] * 6]),
        )
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="LGA",
            tli_epoch="2025-06-21T11:00:00",
            target_ephemeris=[[1.0] * 6],
        )

        assert response.trajectory_gcrs_km == [[7.0] * 6, [8.0] * 6]
        # MCP/sidecar 信封透传：model_dump(mode="json") 直接可序列化
        dumped = response.model_dump(mode="json")
        assert dumped["trajectory_gcrs_km"] == [[7.0] * 6, [8.0] * 6]
        assert dumped["state_frame"] == "synodic_barycentric_km"

    def test_gcrs_segment_absent_maps_to_none(self, monkeypatch):
        """算法层结果无惯性段（low_thrust/零结果）时响应字段为 None。"""
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result(trajectory_gcrs_km=None)
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="LGA",
            tli_epoch="2025-06-21T11:00:00",
            target_ephemeris=[[1.0] * 6],
        )

        assert response.trajectory_gcrs_km is None

    def test_maps_maneuver_events_to_response(self, monkeypatch):
        """算法层 ManeuverEvent 元组 → 响应 maneuver_events（#575 契约）。"""
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result()
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="LGA",
            tli_epoch="2025-06-21T11:00:00",
            target_ephemeris=[[1.0] * 6],
        )

        assert [(e.kind, e.t_sec, e.dv_km_s) for e in response.maneuver_events] == [
            ("departure", 0.0, 3.1),
            ("perilune", 100.0, 0.0),
            ("arrival", 200.0, 0.4),
        ]
        assert all(e.note is None for e in response.maneuver_events)
        # JSON 序列化（MCP 信封）携带事件列表
        dumped = response.model_dump(mode="json")
        assert dumped["maneuver_events"][1] == {
            "kind": "perilune",
            "t_sec": 100.0,
            "dv_km_s": 0.0,
            "note": None,
        }

    def test_maneuver_events_default_empty(self, monkeypatch):
        """算法层未填事件时（旧结果对象），响应为空列表而非缺字段。"""
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result(maneuver_events=())
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.maneuver_events == []

    def test_serializes_dataclass_details(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result(
            details=ManeuverTable(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
        )
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.details["mjd_tdb"] == [60000.0]

    def test_maps_pcn_bplane_and_asymptote(self, monkeypatch):
        """PCN 出口字段逐字段映射（#635）；perilune_alt = r_p − R_moon。"""
        import e2m2e.algorithm.transfer as transfer

        r_moon = MOON.require_mean_radius_km()
        fake_result = self._fake_transfer_result(
            transfer_type="PCN",
            bplane=SimpleNamespace(
                v_inf_km_s=1.79,
                c3_km2_s2=3.2,
                rha_deg=54.9,
                dha_deg=12.2,
                bdot_r_km=0.0,
                bdot_t_km=-3087.4,
                b_mag_km=3087.4,
                theta_deg=180.0,
                perilune_radius_km=r_moon + 182.0,
            ),
            departure_asymptote=SimpleNamespace(rha_deg=32.0, dha_deg=0.0, c3_km2_s2=1.0),
        )
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="PCN",
            tli_epoch=0.0,
            departure_asymptote={"rha_deg": 32.0, "dha_deg": 0.0, "c3_km2_s2": 1.0},
        )

        assert response.bplane is not None
        assert response.bplane.v_inf_km_s == 1.79
        assert response.bplane.perilune_alt_km == pytest.approx(182.0)
        assert response.departure_asymptote is not None
        assert response.departure_asymptote.rha_deg == 32.0

    def test_pcn_fields_absent_maps_to_none(self, monkeypatch):
        """算法层结果无 PCN 字段（HMN/LGA）时响应两字段为 None。"""
        import e2m2e.algorithm.transfer as transfer

        fake_result = self._fake_transfer_result()
        monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: fake_result)
        response = Facade().transfer_design(
            transfer_type="LGA",
            tli_epoch="2025-06-21T11:00:00",
            target_ephemeris=[[1.0] * 6],
        )

        assert response.bplane is None
        assert response.departure_asymptote is None


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


class TestPcnFacadeValidation:
    """PCN 目标参数 XOR 校验（#635）：facade 边界映射 INVALID_PARAMS。"""

    def test_missing_target_rejected(self):
        from e2m2e.api.models import OrbitError

        with pytest.raises(OrbitError) as exc:
            Facade().transfer_design(transfer_type="PCN", tli_epoch=0.0, parking_alt_km=200.0)
        assert exc.value.code == "INVALID_PARAMS"

    def test_both_targets_rejected(self):
        from e2m2e.api.models import OrbitError

        with pytest.raises(OrbitError) as exc:
            Facade().transfer_design(
                transfer_type="PCN",
                tli_epoch=0.0,
                parking_alt_km=200.0,
                bplane_target={"perilune_alt_km": 200.0, "bdot_t_km": 0.0},
                departure_asymptote={"rha_deg": 32.0, "dha_deg": 0.0, "c3_km2_s2": 1.0},
            )
        assert exc.value.code == "INVALID_PARAMS"

    def test_non_pcn_type_with_target_rejected(self):
        from e2m2e.api.models import OrbitError

        with pytest.raises(OrbitError) as exc:
            Facade().transfer_design(
                transfer_type="HMN",
                tli_epoch=0.0,
                target_orbit_radius_km=384400.0,
                bplane_target={"perilune_alt_km": 200.0, "bdot_t_km": 0.0},
            )
        assert exc.value.code == "INVALID_PARAMS"

    def test_bad_tof_range_rejected_as_invalid_params(self):
        """tof_range 形状/序非法 → INVALID_PARAMS（#698），不再是 TRANSFER_FAILED。"""
        from e2m2e.api.models import OrbitError

        for bad in ([1.5], [2.0, 1.5], [1.5, float("inf")]):
            with pytest.raises(OrbitError) as exc:
                Facade().transfer_design(
                    transfer_type="PCN",
                    tli_epoch=0.0,
                    tof_range=bad,
                    departure_asymptote={"rha_deg": 32.0, "dha_deg": 0.0, "c3_km2_s2": 1.0},
                )
            assert exc.value.code == "INVALID_PARAMS"


class TestPcnFailureResponse:
    """PCN 非交会解的真实调用出口（#698）：回显几何 + 清洗非有限 details。"""

    def test_far_flyby_echoes_geometry_and_sanitizes_details(self):
        asym = {"rha_deg": 20.0, "dha_deg": 0.0, "c3_km2_s2": 1.0}
        with pytest.warns(UserWarning, match="未收敛"):
            response = Facade().transfer_design(
                transfer_type="PCN",
                tli_epoch=0.0,
                tof_range=[1.5, 2.0],
                departure_asymptote=asym,
            )
        assert response.status is ConvergenceState.INFEASIBLE
        assert response.cause is FailureCause.CONSTRAINT_VIOLATION
        # 失败解仍回显实际使用的渐近线与达成的 B-plane（字段描述承诺）
        assert response.departure_asymptote is not None
        assert response.departure_asymptote.rha_deg == 20.0
        assert response.bplane is not None
        # 未达成的 LOI 脉冲不写成 inf：细节出口统一清洗为 None
        assert response.details["dv_loi_km_s"] is None
        dumped_details = response.model_dump(mode="json")["details"]
        assert "Infinity" not in repr(dumped_details) and "NaN" not in repr(dumped_details)
