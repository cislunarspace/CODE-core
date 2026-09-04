"""Facade 公共行为、错误翻译与工具元数据测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from api.conftest import control_orbit_business_parameters
from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.api.config import Config
from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.api.models import (
    OrbitError,
    TransferDesignResponse,
)
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import OrbitFamily

pytestmark = pytest.mark.interface


def _fake_control_result():
    return SimpleNamespace(
        num_failed=0,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        sk_statistic=SimpleNamespace(rows=np.zeros((2, 3)), num_failed=0),
        maneuvers=SimpleNamespace(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
        controlled_ephemeris=None,
    )


class TestFacadeValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda facade: facade.design_orbit(orbit_type="DRO", duration=0.0),
            lambda facade: facade.control_orbit(control_mode=9),
            lambda facade: facade.transfer_design(transfer_type="HMN"),
            lambda facade: facade.orbit_propagation(),
            lambda facade: facade.spacetime_transform(),
            lambda facade: facade.catalog.orbit_family_generation(),
        ],
    )
    def test_invalid_input_is_translated_to_orbit_error(self, call):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            call(Facade())


class TestFacadeDelegation:
    def test_control_upper_bounds_are_translated_to_orbit_error(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_controls=10001)
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_monte_carlo=1001)

    def test_design_preserves_algorithm_status(self, monkeypatch):
        from e2m2e.algorithm.design import DesignNotConvergedError

        def fail_design(*args, **kwargs):
            raise DesignNotConvergedError(
                "修正未收敛",
                status=ConvergenceState.MAX_ITERATIONS,
                cause=FailureCause.MAX_ITERATIONS_REACHED,
            )

        monkeypatch.setattr("e2m2e.algorithm.design.design_orbit", fail_design)
        with pytest.raises(OrbitError) as exc_info:
            Facade().design_orbit(orbit_type="DRO")
        assert exc_info.value.status is ConvergenceState.MAX_ITERATIONS
        assert exc_info.value.cause is FailureCause.MAX_ITERATIONS_REACHED

    def test_low_thrust_passes_engine_and_solver_params_to_algorithm(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        captured: dict[str, Any] = {}

        def fake_transfer(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="任务完成",
                transfer_type="low_thrust",
                delta_v=1.0,
                trajectory=None,
                trajectory_times=None,
                state_frame="force_model_state",
                maneuver_events=(),
                details={},
            )

        monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer)
        Facade().transfer_design(
            transfer_type="low_thrust",
            tli_epoch="2025-06-21T11:00:00",
            engine_config={"t_max": 0.5, "isp": 3000.0},
            initial_mass=1000.0,
            n_segments=20,
            solver_method="collocation",
            duration_days=45.0,
            target_oe=[7000.0, 0.1, 0.2],
            departure_state=[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            target_state=[8000.0, 0.0, 0.0, 0.0, 7.0, 0.0],
        )

        engine = captured["engine_config"]
        assert isinstance(engine, transfer.EngineConfig)
        assert engine.t_max == 0.5
        assert engine.isp == 3000.0
        assert captured["initial_mass"] == 1000.0
        assert captured["n_segments"] == 20
        assert captured["solver_method"] == "collocation"
        assert captured["duration_days"] == 45.0
        assert captured["target_oe"] == (7000.0, 0.1, 0.2)
        assert np.allclose(captured["departure_state"], [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        assert np.allclose(captured["target_state"], [8000.0, 0.0, 0.0, 0.0, 7.0, 0.0])

    def test_low_thrust_missing_engine_config_is_invalid_params(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().transfer_design(
                transfer_type="low_thrust",
                tli_epoch="2025-06-21T11:00:00",
                initial_mass=1000.0,
            )

    def test_control_passes_model_values_and_config_to_algorithm(self, monkeypatch):
        import e2m2e.algorithm.station_keeping as station_keeping

        captured: dict[str, Any] = {}
        business = set(control_orbit_business_parameters())

        def fake_control(input_ephemeris, **kwargs):
            captured.update(kwargs)
            return _fake_control_result()

        monkeypatch.setattr(station_keeping, "control_orbit", fake_control)
        facade = Facade(config=Config(kernel_dir="configured-kernels"))
        facade.control_orbit(
            input_ephemeris="x",
            control_interval=45.0,
            feedback_arc=20.0,
            position_accuracy=123.0,
            thrust_mean=8.0,
            tight_tolerance_km=0.5,
        )

        assert set(captured) == business | {"kernel_dir"}
        assert captured["kernel_dir"] == "configured-kernels"
        assert captured["control_interval"] == 45.0
        assert captured["feedback_arc"] == 20.0
        assert captured["position_accuracy"] == 123.0
        assert captured["thrust_mean"] == 8.0
        assert captured["tight_tolerance_km"] == 0.5
        assert captured["num_controls"] == 120


class TestFacadeCallChains:
    def test_unknown_design_type_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().design_orbit(orbit_type="NOPE")

    def test_hmn_returns_converged_response(self):
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert response.message == "霍曼转移完成"
        # ADR 0040：收敛轨迹与会合系时刻随响应下发
        assert response.trajectory is not None
        assert len(response.trajectory[0]) == 6
        assert response.trajectory_times is not None
        assert len(response.trajectory_times) == len(response.trajectory)
        # ADR 0040 增补：数据系标签随响应下发
        assert response.state_frame == "synodic_barycentric_km"
        # #584：惯性几何段随响应下发，与会合几何同行对齐
        assert response.trajectory_gcrs_km is not None
        assert len(response.trajectory_gcrs_km) == len(response.trajectory)
        assert len(response.trajectory_gcrs_km[0]) == 6

    def test_unknown_transfer_type_is_not_implemented_error(self):
        with pytest.raises(OrbitError, match="NOT_IMPLEMENTED"):
            Facade().transfer_design(
                transfer_type="UNKNOWN_TYPE",
                tli_epoch="2025-06-21T11:00:00",
                target_orbit_radius_km=42164.0,
            )

    def test_unknown_spacetime_transform_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spacetime_transform(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type="unknown",
                et0_jd=2459000.0,
            )

    def test_spacetime_transform_rejects_mismatched_batches(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().spacetime_transform(
                states=[[1.0] * 6, [2.0] * 6],
                times=[0.0],
                transform_type="j2000_to_synodic",
                et0_jd=2459000.0,
            )

    def test_halo_family_delegates_to_algorithm(self):
        family = Facade().catalog.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )
        assert family.family_type == "halo"
        assert family.status is ConvergenceState.CONVERGED
        assert family.cause is FailureCause.NONE
        assert len(family) >= 1
        assert all(orbit.parameters.get("libration_point") == 1 for orbit in family)

    @pytest.mark.parametrize(
        ("orbit_type", "entry_name", "params", "expected_args"),
        [
            (
                "NRHO",
                "design_nrho_family",
                {"libration_point": 1, "north_south": 1, "perilune_height_max_km": 30000.0},
                (1, 1, 30000.0),
            ),
            (
                "AXIAL",
                "design_axial_family",
                {"libration_point": 1, "max_amplitude_km": -5000.0},
                (1, -5000.0),
            ),
            (
                "LISSAJOUS",
                "design_lissajous_family",
                {
                    "libration_point": 3,
                    "amplitude_in_km": 8000.0,
                    "amplitude_out_km": 9000.0,
                    "phase_in": 0.1,
                    "phase_out": 0.6,
                },
                (3, 8000.0, 9000.0, 0.1, 0.6),
            ),
            (
                "SPO",
                "design_spo_family",
                {"libration_point": 4, "min_amplitude_km": 5000.0, "max_amplitude_km": 20000.0},
                (4, 5000.0, 20000.0),
            ),
            (
                "LPO",
                "design_lpo_family",
                {"libration_point": 5, "min_amplitude_km": 5000.0, "max_amplitude_km": 30000.0},
                (5, 5000.0, 30000.0),
            ),
            (
                "HORSESHOE",
                "design_horseshoe_family",
                {"libration_point": 4, "min_amplitude_km": 50000.0, "max_amplitude_km": 100000.0},
                (4, 50000.0, 100000.0),
            ),
            (
                "DRO",
                "design_dro_family",
                {"min_amplitude_km": 5000.0, "max_amplitude_km": 20000.0},
                (5000.0, 20000.0),
            ),
        ],
    )
    def test_non_halo_family_dispatches_to_algorithm(
        self, monkeypatch, orbit_type, entry_name, params, expected_args
    ):
        import e2m2e.algorithm.family as family_module

        sentinel = OrbitFamily(family_type=orbit_type.lower())
        calls = []

        def fake_entry(*args, **kwargs):
            calls.append((args, kwargs))
            return FamilyGenerationResult(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="测试完成",
                family=sentinel,
                requested_members=2,
                generated_members=0,
            )

        monkeypatch.setattr(family_module, entry_name, fake_entry)
        result = Facade().catalog.orbit_family_generation(
            orbit_type=orbit_type, n_orbits=2, **params
        )

        expected_kwargs = {"n_orbits": 2}
        if orbit_type == "NRHO":
            expected_kwargs["continuation_direction"] = "toward-moon"
        elif orbit_type == "AXIAL":
            expected_kwargs["continuation_direction"] = "increase-amplitude"
        elif orbit_type == "LISSAJOUS":
            expected_kwargs["sampling_mode"] = "linear-amplitudes"
        elif orbit_type in ("SPO", "LPO", "HORSESHOE"):
            expected_kwargs.update(
                continuation_direction="decrease-x0",
                match_tolerance_km=50.0 if orbit_type == "HORSESHOE" else 20.0,
            )
        assert isinstance(result, OrbitFamily)
        assert result.family_type == sentinel.family_type
        assert result.status is ConvergenceState.CONVERGED
        assert result.generated_members == 0
        assert calls == [(expected_args, expected_kwargs)]

    def test_soft_failure_preserves_partial_family_and_status(self, monkeypatch):
        import e2m2e.algorithm.family as family_module

        partial = OrbitFamily(family_type="spo")
        soft_failure = FamilyGenerationResult(
            status=ConvergenceState.STAGNATED,
            cause=FailureCause.STAGNATION_DETECTED,
            message="PAL 步长降至下限",
            family=partial,
            requested_members=5,
            generated_members=0,
        )
        monkeypatch.setattr(
            family_module, "design_spo_family", lambda *args, **kwargs: soft_failure
        )

        result = Facade().catalog.orbit_family_generation(
            orbit_type="SPO",
            libration_point=4,
            min_amplitude_km=5000.0,
            max_amplitude_km=20000.0,
            n_orbits=5,
        )

        assert isinstance(result, OrbitFamily)
        assert result.family_type == partial.family_type
        assert result.orbits == partial.orbits
        assert result.status is ConvergenceState.STAGNATED
        assert result.cause is FailureCause.STAGNATION_DETECTED

    @pytest.mark.parametrize(
        "params",
        [{"orbit_type": "NOPE"}, {"orbit_type": "HALO", "libration_point": 4}],
    )
    def test_invalid_family_input_is_translated_to_orbit_error(self, params):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.orbit_family_generation(**params)

    def test_invalid_family_libration_point_type_is_translated_to_orbit_error(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.orbit_family_generation(orbit_type="HALO", libration_point="bad")


class TestFacadeToolInventory:
    """接口类分家后的工具清单（ADR 0043）：Facade 组合根扫多个暴露类。"""

    def test_inventory_counts_eighteen_implemented_tools(self):
        inventory = tool_inventory(Facade())
        assert len(inventory) == 18
        assert all(tool.status == "implemented" for tool in inventory)

    def test_each_class_keeps_its_domain(self):
        from e2m2e.api.catalog import Catalog
        from e2m2e.api.spatiography import Spatiography

        facade = Facade()
        # Facade 只留任务级五方法（ADR 0043 决策 1）
        assert set(mcp_tools(facade)) == {
            "design_orbit",
            "control_orbit",
            "transfer_design",
            "orbit_propagation",
            "spacetime_transform",
        }
        assert set(mcp_tools(Catalog())) == {
            "catalog_query",
            "catalog_get",
            "catalog_delete",
            "catalog_tag",
            "catalog_terminology",
            "catalog_export",
            "catalog_sweep",
            "orbit_family_generation",
        }
        assert set(mcp_tools(Spatiography())) == {
            "spatiography_scales",
            "spatiography_classify",
            "spatiography_boundaries",
            "spatiography_resonance_atlas",
            "spatiography_dynamical_map",
        }
        # 组合根把两类作为实例暴露给进程内调用方（ADR 0043 决策 2/3）
        assert isinstance(facade.catalog, Catalog)
        assert isinstance(facade.spatiography, Spatiography)


class TestFacadeTransferTopN:
    """top-N 可行解契约（#583，ADR 0040 增补）：opt-in 翻译与默认零变化。"""

    @staticmethod
    def _fake_result_with_candidates(monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        captured: dict[str, Any] = {}

        def fake_transfer(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="任务完成",
                transfer_type="low_thrust",
                delta_v=1.0,
                trajectory=np.arange(6, dtype=float).reshape(1, 6),
                trajectory_times=np.array([0.0]),
                state_frame="force_model_state",
                maneuver_events=(),
                candidates=(
                    transfer.TransferCandidate(
                        delta_v_km_s=1.0,
                        tli_epoch=2460800.5,
                        tof_sec=86400.0,
                        trajectory=np.arange(6, dtype=float).reshape(1, 6),
                        trajectory_times=np.array([0.0]),
                        state_frame="force_model_state",
                        selected=True,
                        refined=True,
                    ),
                    transfer.TransferCandidate(
                        delta_v_km_s=1.4,
                        tli_epoch=2460800.5,
                        tof_sec=90000.0,
                        trajectory=None,
                        trajectory_times=None,
                        state_frame="force_model_state",
                        selected=False,
                        refined=False,
                    ),
                ),
                details={},
            )

        monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer)
        return captured

    def test_top_n_passed_through_and_candidates_translated(self, monkeypatch):
        captured = self._fake_result_with_candidates(monkeypatch)
        response = Facade().transfer_design(
            transfer_type="low_thrust",
            tli_epoch="2025-06-21T11:00:00",
            engine_config={"t_max": 0.5, "isp": 3000.0},
            initial_mass=1000.0,
            top_n=5,
        )
        # opt-in 参数透传算法层
        assert captured["top_n"] == 5
        # 候选翻译为响应模型，ndarray → list，标记原样
        assert response.candidates is not None
        assert len(response.candidates) == 2
        selected, unrefined = response.candidates
        assert selected.selected is True and selected.refined is True
        assert selected.delta_v_km_s == pytest.approx(1.0)
        assert selected.trajectory == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
        assert unrefined.selected is False and unrefined.refined is False
        assert unrefined.trajectory is None and unrefined.trajectory_times is None

    def test_default_response_carries_no_candidates(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        captured: dict[str, Any] = {}

        def fake_transfer(*args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="任务完成",
                transfer_type="low_thrust",
                delta_v=1.0,
                trajectory=None,
                trajectory_times=None,
                state_frame="force_model_state",
                maneuver_events=(),
                details={},
            )

        monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer)
        response = Facade().transfer_design(
            transfer_type="low_thrust",
            tli_epoch="2025-06-21T11:00:00",
            engine_config={"t_max": 0.5, "isp": 3000.0},
            initial_mass=1000.0,
        )
        # 未开启时：不向算法层传 top_n（保持 None），响应无候选字段语义
        assert captured["top_n"] is None
        assert response.candidates is None

    @pytest.mark.parametrize("bad_top_n", [0, -3])
    def test_non_positive_top_n_is_invalid_params(self, bad_top_n):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().transfer_design(
                transfer_type="HMN",
                tli_epoch="2025-06-21T11:00:00",
                target_orbit_radius_km=42164.0,
                top_n=bad_top_n,
            )

    def test_candidates_survive_json_round_trip(self, monkeypatch):
        self._fake_result_with_candidates(monkeypatch)
        response = Facade().transfer_design(
            transfer_type="low_thrust",
            tli_epoch="2025-06-21T11:00:00",
            engine_config={"t_max": 0.5, "isp": 3000.0},
            initial_mass=1000.0,
            top_n=5,
        )
        # MCP/JSON 序列化下新字段完整可读（模型往返逐字段相等）
        revived = TransferDesignResponse.model_validate_json(response.model_dump_json())
        assert revived.candidates == response.candidates
