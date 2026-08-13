"""api 层 Facade 门面测试。"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from e2m2e.api import NumericRange
from e2m2e.api.facade import Facade, mcp_tools
from e2m2e.api.models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    OrbitError,
    PropagationRequest,
    SpacetimeTransformRequest,
    TransferDesignRequest,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.trajectory import EphemerisTable

pytestmark = pytest.mark.interface

_CONTROL_RUNTIME_PARAMS = {"spice", "kernel_dir", "n_workers", "seed"}
_CONDITIONAL_RANGE_CASES = [
    ("DRO", "amplitude", 1737.0, 110000.0, True),
    ("DPO", "amplitude", 1737.0, 110000.0, True),
    ("HALO", "amplitude", -73000.0, 73000.0, True),
    ("NRHO", "perilune_height", 100.0, 10000.0, True),
    ("L4", "amplitude_out", 0.0, 76000.0, False),
    ("L5", "amplitude_out", 0.0, 76000.0, False),
    ("AXIAL", "amplitude", -60000.0, 60000.0, True),
    ("L4_SPO", "amplitude", 1737.0, 200000.0, True),
    ("L5_SPO", "amplitude", 1737.0, 200000.0, True),
    ("L4_LPO", "amplitude", 1000.0, 200000.0, True),
    ("L5_LPO", "amplitude", 1000.0, 200000.0, True),
    ("L4_HORSESHOE", "amplitude", 50000.0, 200000.0, True),
    ("L5_HORSESHOE", "amplitude", 50000.0, 200000.0, True),
]


def _control_business_params() -> dict[str, inspect.Parameter]:
    """返回 ControlOrbitRequest 应覆盖的算法层业务参数。"""
    from e2m2e.algorithm.station_keeping import control_orbit

    return {
        name: param
        for name, param in inspect.signature(control_orbit).parameters.items()
        if name != "input_ephemeris" and name not in _CONTROL_RUNTIME_PARAMS
    }


class TestDesignOrbitRequest:
    def test_defaults(self):
        req = DesignOrbitRequest(orbit_type="DRO")
        assert req.duration == 31557600.0  # 1 年 = 365.25 × 86400 秒
        assert req.output_step == 3600.0
        assert req.correction_method == "two_level"

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            DesignOrbitRequest(orbit_type="DRO", duration=0.0)

    def test_elfo_defaults(self):
        req = DesignOrbitRequest(orbit_type="ELFO", semi_major_axis=3000.0)
        assert req.duration == 5184000.0  # 60 天
        assert req.inclination == 75.0
        assert req.arg_of_pericenter == 270.0
        assert req.perilune_height == 200.0

    def test_perilune_height_bounds(self):
        with pytest.raises(ValidationError):
            DesignOrbitRequest(orbit_type="NRHO", perilune_height=50.0)

    def test_dead_correction_velocity_tolerance_rejected(self):
        """correction_velocity_tolerance 无消费方（#410），删除后按 extra=forbid 拒绝。"""
        with pytest.raises(ValidationError, match="correction_velocity_tolerance"):
            DesignOrbitRequest(orbit_type="DRO", correction_velocity_tolerance=0.5)
        with pytest.raises(ValidationError, match="correction_velocity_tolerance"):
            DesignOrbitRequest(orbit_type="NRHO", correction_velocity_tolerance=0.5)

    def test_global_amplitude_out_bound_survives_lissajous_l3_extension(self):
        with pytest.raises(ValidationError, match="amplitude_out"):
            DesignOrbitRequest(orbit_type="DRO", amplitude_out=80000.0)
        with pytest.raises(ValidationError, match="amplitude_out"):
            DesignOrbitRequest(orbit_type="ELFO", semi_major_axis=3000.0, amplitude_out=80000.0)

    def test_valid_ranges_exposes_conditional_amplitude_limits(self):
        dro = DesignOrbitRequest.valid_ranges("dro")
        halo = DesignOrbitRequest.valid_ranges("HALO")
        axial = DesignOrbitRequest.valid_ranges("AXIAL")

        assert dro["amplitude"].minimum == 1737.0
        assert dro["amplitude"].maximum == 110000.0
        assert dro["amplitude"].minimum_inclusive
        assert dro["amplitude"].maximum_inclusive
        assert halo["amplitude"].minimum == -73000.0
        assert halo["amplitude"].maximum == 73000.0
        assert axial["amplitude"].minimum == -60000.0
        assert axial["amplitude"].maximum == 60000.0

    @pytest.mark.parametrize(
        ("orbit_type", "field", "minimum", "maximum", "minimum_inclusive"),
        _CONDITIONAL_RANGE_CASES,
    )
    def test_valid_ranges_exposes_all_conditional_numeric_limits(
        self, orbit_type, field, minimum, maximum, minimum_inclusive
    ):
        numeric_range = DesignOrbitRequest.valid_ranges(orbit_type)[field]

        assert numeric_range.minimum == minimum
        assert numeric_range.maximum == maximum
        assert numeric_range.minimum_inclusive is minimum_inclusive
        assert numeric_range.maximum_inclusive

    @pytest.mark.parametrize(
        ("orbit_type", "field", "minimum", "maximum", "minimum_inclusive"),
        _CONDITIONAL_RANGE_CASES,
    )
    def test_conditional_range_boundaries_match_request_validation(
        self, orbit_type, field, minimum, maximum, minimum_inclusive
    ):
        accepted = DesignOrbitRequest(orbit_type=orbit_type, **{field: maximum})
        assert getattr(accepted, field) == maximum

        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(orbit_type=orbit_type, **{field: maximum + 1.0})

        if minimum_inclusive:
            accepted = DesignOrbitRequest(orbit_type=orbit_type, **{field: minimum})
            assert getattr(accepted, field) == minimum
            with pytest.raises(ValidationError, match=field):
                DesignOrbitRequest(orbit_type=orbit_type, **{field: minimum - 1.0})
        else:
            with pytest.raises(ValidationError, match=field):
                DesignOrbitRequest(orbit_type=orbit_type, **{field: minimum})
            accepted = DesignOrbitRequest(orbit_type=orbit_type, **{field: minimum + 1.0})
            assert getattr(accepted, field) == minimum + 1.0

    def test_valid_ranges_exposes_lissajous_limit_by_collinear_point(self):
        default = DesignOrbitRequest.valid_ranges("LISSAJOUS")
        l1 = DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=1)
        l3 = DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=3)

        assert default["amplitude_out"].maximum == 7600.0
        assert l1["amplitude_in"].minimum == 0.0
        assert not l1["amplitude_in"].minimum_inclusive
        assert l1["amplitude_out"].maximum == 7600.0
        assert l3["amplitude_in"].maximum == 100000.0
        assert l3["amplitude_out"].maximum == 100000.0
        assert isinstance(l3["amplitude_out"], NumericRange)

        with pytest.raises(ValueError, match="collinear_point"):
            DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=4)
        with pytest.raises(ValueError, match="字符串"):
            DesignOrbitRequest.valid_ranges(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["amplitude_in", "amplitude_out"])
    def test_lissajous_boundaries_match_ranges(self, field):
        l1 = DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 7600.0})
        l3 = DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=3, **{field: 80000.0})
        assert getattr(l1, field) == 7600.0
        assert getattr(l3, field) == 80000.0

        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 0.0})
        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 80000.0})


class TestControlOrbitRequest:
    def test_control_mode_bounds(self):
        """mode 7 超出范围；mode 4 在 API 层允许（算法层校验 engine_layout）。"""
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", control_mode=7)
        # mode 4 在 API 层不报错（engine_layout 校验在算法层）
        req = ControlOrbitRequest(input_ephemeris="x", control_mode=4)
        assert req.control_mode == 4

    def test_num_controls_upper_bound(self):
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", num_controls=10001)
        assert ControlOrbitRequest(input_ephemeris="x", num_controls=10000).num_controls == 10000

    def test_num_monte_carlo_upper_bound(self):
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", num_monte_carlo=1001)
        assert (
            ControlOrbitRequest(input_ephemeris="x", num_monte_carlo=1000).num_monte_carlo == 1000
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dyb", [0.0] * 8),
            ("real_dyb", [0.0] * 10),
            ("srp_offset_m", [0.0] * 2),
            ("srp_torque", [0.0] * 4),
        ],
    )
    def test_vector_lengths(self, field, value):
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", **{field: value})

    @pytest.mark.parametrize(
        "perturbation",
        [{"unknown": 1}, {"sun_body": 2}, {"solar_radiation": 3}],
    )
    def test_perturbation_switches(self, perturbation):
        with pytest.raises(ValidationError):
            ControlOrbitRequest(input_ephemeris="x", perturbation=perturbation)


class TestControlOrbitAlignment:
    """模型与算法层签名对齐契约（#408）：字段集、默认值、schema 约束不漂移。"""

    def test_request_covers_algorithm_business_params(self):
        """模型字段集 ⊇ 算法层业务参数集（#408 缺口①的防漂移契约）。"""
        business = set(_control_business_params())
        model = set(ControlOrbitRequest.model_fields)
        missing = business - model
        assert not missing, f"算法层参数未进模型：{sorted(missing)}"

    def test_defaults_match_algorithm_signature(self):
        """模型默认值逐项等于算法层签名默认值（#408 缺口①）。"""
        req = ControlOrbitRequest(input_ephemeris="x")
        for name, param in _control_business_params().items():
            assert getattr(req, name) == param.default, f"{name} 默认值不一致"

    def test_schema_exposes_range_metadata(self):
        """JSON schema 带 range 元数据，GUI 据此生成范围提示（#408 缺口②）。"""
        props = ControlOrbitRequest.model_json_schema()["properties"]
        assert props["num_controls"]["maximum"] == 10000
        assert props["num_monte_carlo"]["maximum"] == 1000
        assert props["control_interval"]["exclusiveMinimum"] == 0
        assert props["position_accuracy"]["exclusiveMinimum"] == 0
        assert props["earth_degree"]["minimum"] == 2
        assert props["special_damping_factor"]["maximum"] == 1.0


class TestTransferDesignRequest:
    def test_defaults(self):
        req = TransferDesignRequest(transfer_type="HMN", tli_epoch="2025-06-21T11:00:00")
        assert req.parking_alt_km == 200.0
        assert req.incl_deg == 28.5
        assert req.flight_path_deg == 0.0

    def test_invalid_transfer_type_type(self):
        with pytest.raises(ValidationError):
            TransferDesignRequest(transfer_type=123, tli_epoch="2025-06-21T11:00:00")


class TestPropagationRequest:
    def test_defaults(self):
        req = PropagationRequest(
            initial_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            epoch="2025-06-21T11:00:00",
            duration=3600.0,
        )
        assert req.output_step == 3600.0
        assert req.force_config is None

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            PropagationRequest(
                initial_state=[0.0] * 6,
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_invalid_state_shape(self):
        # Pydantic 在 API 边界强制长度为 6
        with pytest.raises(ValidationError):
            PropagationRequest(
                initial_state=[0.0] * 5,
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
            )


class TestSpacetimeTransformRequest:
    def test_defaults(self):
        req = SpacetimeTransformRequest(
            states=[[1.0] * 6],
            times=[0.0],
            transform_type="j2000_to_synodic",
            et0_jd=2459000.0,
        )
        assert req.ephemeris_path is None

    def test_unknown_transform_type_type(self):
        with pytest.raises(ValidationError):
            SpacetimeTransformRequest(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type=123,
                et0_jd=2459000.0,
            )


class TestFacade:
    def test_construct(self):
        facade = Facade()
        assert facade._config is not None

    def test_design_orbit_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.design_orbit(orbit_type="DRO", duration=0.0)

    def test_design_orbit_unknown_type(self):
        facade = Facade()
        with pytest.raises(OrbitError):
            facade.design_orbit(orbit_type="NOPE")

    def test_design_failure_preserves_algorithm_status(self, monkeypatch):
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

    def test_control_orbit_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(control_mode=9)

    def test_control_orbit_upper_bounds(self):
        """拍板的上界在模型层拦截（#408）：10001/1001 拒绝，10000/1000 放行。"""
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_controls=10001)
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.control_orbit(input_ephemeris="x", num_monte_carlo=1001)

    def test_control_orbit_passthrough_to_algorithm(self, monkeypatch):
        """facade 把模型全部算法参数透传给算法层（#408 缺口①的防漂移契约）。"""
        import e2m2e.algorithm.station_keeping as sk

        # 先取真实签名，再 patch（patch 后签名变为 fake 的 **kwargs）
        business = set(_control_business_params())

        captured: dict[str, Any] = {}

        def fake_control(input_ephemeris, **kwargs):
            captured.update(kwargs)
            return _fake_control_result()

        monkeypatch.setattr(sk, "control_orbit", fake_control)
        Facade().control_orbit(
            input_ephemeris="x",
            control_interval=45.0,
            feedback_arc=20.0,
            position_accuracy=123.0,
            thrust_mean=8.0,
            tight_tolerance_km=0.5,
        )
        # 键集：算法层业务参数 + Facade 注入的 kernel_dir，一个不少
        assert set(captured) == business | {"kernel_dir"}
        # 非默认值确实到达（值透传，非仅键透传）
        assert captured["control_interval"] == 45.0
        assert captured["feedback_arc"] == 20.0
        assert captured["position_accuracy"] == 123.0
        assert captured["thrust_mean"] == 8.0
        assert captured["tight_tolerance_km"] == 0.5
        # 未显式传的字段带模型默认值
        assert captured["num_controls"] == 120

    def test_transfer_design_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.transfer_design(transfer_type="HMN")

    def test_orbit_propagation_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.orbit_propagation(
                initial_state=[0.0] * 6,
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_spacetime_transform_invalid_params(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.spacetime_transform(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type="unknown",
                et0_jd=2459000.0,
            )

    def test_orbit_family_generation_unknown(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.orbit_family_generation(orbit_type="NOPE")


class TestMcpTools:
    def test_derives_exposed_methods(self):
        facade = Facade()
        names = mcp_tools(facade)
        assert "design_orbit" in names
        assert "control_orbit" in names
        assert "transfer_design" in names
        assert "orbit_propagation" in names
        assert "spacetime_transform" in names
        assert "orbit_family_generation" in names


class TestFacadeCallChain:
    """无需 SPICE 的轻量调用链：仅验证错误码与序列化路径。"""

    def test_transfer_design_hmn_call_chain(self):
        response = Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch="2025-06-21T11:00:00",
            target_orbit_radius_km=42164.0,
        )
        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert response.message == "霍曼转移完成"

    def test_transfer_design_invalid_type_call_chain(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="NOT_IMPLEMENTED"):
            facade.transfer_design(
                transfer_type="UNKNOWN_TYPE",
                tli_epoch="2025-06-21T11:00:00",
                target_orbit_radius_km=42164.0,
            )

    def test_spacetime_transform_mismatched_lengths(self):
        facade = Facade()
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade.spacetime_transform(
                states=[[1.0] * 6, [2.0] * 6],
                times=[0.0],
                transform_type="j2000_to_synodic",
                et0_jd=2459000.0,
            )

    def test_details_to_dict_with_ndarray(self):
        from e2m2e.api.facade import _details_to_dict

        class DummyDataclass:
            pass

        details = {
            "array": np.array([1.0, 2.0]),
            "nested": {"tuple": (np.array([3.0]), 4.0)},
        }
        result = _details_to_dict(details)
        assert result["array"] == [1.0, 2.0]
        assert result["nested"]["tuple"] == [[3.0], 4.0]


# ---------------------------------------------------------------------------
# 几何字段补齐（#312）：Facade Response 须带 mu/states/times/ephemeris，
# 让下游（transfer-orbit-design）可退回 Facade、移除 algorithm 层直调。
# 这些测试不依赖 SPICE：直接验证序列化助手 + 纯翻译函数 + Pydantic 模型。
# ---------------------------------------------------------------------------


def _fake_control_result():
    """构造鸭子类型 ControlOrbitResult（不调算法层、不需 SPICE）。"""
    from types import SimpleNamespace

    return SimpleNamespace(
        num_failed=0,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        sk_statistic=SimpleNamespace(rows=np.zeros((2, 3)), num_failed=0),
        maneuvers=SimpleNamespace(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
        controlled_ephemeris=None,
    )


def _make_ephemeris(n: int = 3, with_jd: bool = False) -> EphemerisTable:
    """构造一个小型 EphemerisTable（无 SPICE 依赖），供翻译测试用。"""
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


class TestEphemerisToDict:
    def test_serializes_ndarrays_to_lists(self):
        from e2m2e.api.facade import _ephemeris_to_dict

        d = _ephemeris_to_dict(_make_ephemeris(n=2))
        assert d is not None
        # ndarray → list
        assert d["position_km"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        assert d["velocity_mps"] == [[1000.0, 1000.0, 1000.0]] * 2
        assert d["synodic_position"] == [[0.5, 0.5, 0.5]] * 2
        assert d["year"] == [2024, 2024]
        assert d["hour"] == [0, 1]
        # 全字段在列（EphemerisTable 数据列）
        assert set(d) == {
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

    def test_times_jd_tdb_none_and_populated(self):
        from e2m2e.api.facade import _ephemeris_to_dict

        # design 链路不填 times_jd_tdb → None
        d_none = _ephemeris_to_dict(_make_ephemeris(with_jd=False))
        assert d_none["times_jd_tdb"] is None
        # 传播链路填了 times_jd_tdb → list
        d_jd = _ephemeris_to_dict(_make_ephemeris(with_jd=True))
        assert d_jd["times_jd_tdb"] == pytest.approx([2460310.0, 2460310.5, 2460311.0])

    def test_none_input_returns_none(self):
        from e2m2e.api.facade import _ephemeris_to_dict

        assert _ephemeris_to_dict(None) is None


class TestDesignResultToResponse:
    def _mock_result(self, *, with_system: bool = True):
        """构造一个鸭子类型的 OrbitDesignResult（不调算法层、不需 SPICE）。"""
        from types import SimpleNamespace

        system = SimpleNamespace(mu=Datum.DE421.mu) if with_system else None
        cr3bp_orbit = SimpleNamespace(
            states=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]]),
            times=np.array([0.0, 1.234]),
            system=system,
        )
        correction = SimpleNamespace(
            status="converged", cause="none", message="任务完成", iterations=4
        )
        return SimpleNamespace(
            orbit_type="DRO",
            epoch_utc="2024-01-01T00:00:00.000",
            duration_day=365.25,
            output_step_sec=3600.0,
            initial_state=np.zeros(6),
            ephemeris=_make_ephemeris(n=3),
            cr3bp_orbit=cr3bp_orbit,
            cr3bp_jacobi=3.16,
            correction=correction,
            force_config={"sun_body": 1},
            status="converged",
            cause="none",
            message="任务完成",
            drift_e=None,
            drift_aop_deg=None,
            drift_rp_km=None,
            secular_aop_rate_deg_per_year=None,
        )

    def test_extracts_geometry_fields(self):
        from e2m2e.api.facade import _design_result_to_response

        resp = _design_result_to_response(self._mock_result())
        # 摘要字段保留
        assert resp.orbit_type == "DRO"
        assert resp.cr3bp_jacobi == pytest.approx(3.16)
        assert resp.status.value == "converged"
        assert resp.correction_iterations == 4
        assert resp.initial_state == [0.0] * 6
        # 新增几何字段
        assert resp.mu == pytest.approx(Datum.DE421.mu)
        assert resp.states == [[0.0] * 6, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]]
        assert resp.times == [0.0, 1.234]
        assert resp.ephemeris is not None
        assert resp.ephemeris["position_km"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]

    def test_mu_none_when_system_missing(self):
        """system 未绑定时 mu 防御性回退为 None（同下游 getattr 约定）。"""
        from e2m2e.api.facade import _design_result_to_response

        resp = _design_result_to_response(self._mock_result(with_system=False))
        assert resp.mu is None
        # 其余几何字段仍正常
        assert resp.states == [[0.0] * 6, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]]

    def test_orbit_design_result_field_contract(self):
        """OrbitDesignResult 字段契约：initial_state 6 维、cr3bp_orbit 多点、jacobi float。

        从 design/scenarios/test_lissajous.py & test_triangular.py 的
        initial_state_shape / cr3bp_orbit_present 断言下沉（ADR 0021）：
        字段契约归 interface 类，用 mock result 零管线验证，不靠真传播。
        """
        from e2m2e.api.facade import _design_result_to_response

        result = self._mock_result()
        # algorithm 层 OrbitDesignResult 字段契约
        assert result.initial_state.shape == (6,)
        assert result.cr3bp_orbit is not None
        assert isinstance(result.cr3bp_jacobi, float)
        assert len(result.cr3bp_orbit.states) > 1
        # 翻译到 api 层 Response 后对应字段
        resp = _design_result_to_response(result)
        assert len(resp.initial_state) == 6
        assert isinstance(resp.cr3bp_jacobi, float)
        assert len(resp.states) > 1
        assert isinstance(resp.mu, float)


class TestControlResultToResponse:
    def _mock_result(self, *, controlled):
        from types import SimpleNamespace

        from e2m2e.data.types.maneuver import ManeuverTable
        from e2m2e.data.types.sk_statistic import SKStatistic

        return SimpleNamespace(
            num_failed=1,
            status="converged" if controlled else "failed",
            cause="none" if controlled else "unknown",
            message="任务完成" if controlled else "全部蒙特卡洛样本失败",
            sk_statistic=SKStatistic(rows=np.zeros((2, 3)), num_failed=1),
            maneuvers=ManeuverTable(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0])),
            controlled_ephemeris=_make_ephemeris(n=2) if controlled else None,
        )

    def test_with_controlled_ephemeris_and_mu_echo(self):
        from e2m2e.api.facade import _control_result_to_response

        resp = _control_result_to_response(self._mock_result(controlled=True), mu=Datum.DE421.mu)
        assert resp.status.value == "converged"
        assert resp.cause.value == "none"
        assert resp.num_failed == 1
        assert resp.controlled_ephemeris is not None
        assert resp.controlled_ephemeris["synodic_position"] == [[0.5, 0.5, 0.5]] * 2
        # mu 由请求透传（算法层不产 mu）
        assert resp.mu == pytest.approx(Datum.DE421.mu)

    def test_all_failed_no_controlled_ephemeris(self):
        from e2m2e.api.facade import _control_result_to_response

        resp = _control_result_to_response(self._mock_result(controlled=False), mu=None)
        assert resp.status.value == "failed"
        assert resp.cause.value == "unknown"
        assert resp.controlled_ephemeris is None
        assert resp.mu is None


class TestGeometryModelFields:
    """Pydantic 模型字段验收（#312）。"""

    def test_design_response_carries_geometry(self):
        resp = DesignOrbitResponse(
            status="converged",
            cause="none",
            message="任务完成",
            orbit_type="DRO",
            epoch_utc="2024-01-01T00:00:00.000",
            duration_day=365.25,
            initial_state=[0.0] * 6,
            cr3bp_jacobi=3.16,
            correction_iterations=4,
            force_config={"sun_body": 1},
            mu=Datum.DE421.mu,
            states=[[0.0] * 6],
            times=[0.0],
            ephemeris={"position_km": [[0.0, 0.0, 0.0]]},
        )
        # 序列化往返：ndarray-free dict/list 可 JSON 化
        dumped = resp.model_dump()
        assert dumped["mu"] == pytest.approx(Datum.DE421.mu)
        assert dumped["states"] == [[0.0] * 6]
        assert dumped["ephemeris"]["position_km"] == [[0.0, 0.0, 0.0]]

    def test_design_response_mu_optional(self):
        """mu 防御性可空（system 未绑定时）。"""
        resp = DesignOrbitResponse(
            status="converged",
            cause="none",
            message="任务完成",
            orbit_type="DRO",
            epoch_utc="2024-01-01T00:00:00.000",
            duration_day=365.25,
            initial_state=[0.0] * 6,
            cr3bp_jacobi=3.16,
            correction_iterations=4,
            force_config={},
            mu=None,
            states=[],
            times=[],
            ephemeris={},
        )
        assert resp.mu is None

    def test_control_response_backward_compatible(self):
        """新增字段带默认值：旧路径（仅摘要）构造不报错。"""
        resp = ControlOrbitResponse(
            status="converged",
            cause="none",
            message="任务完成",
            num_failed=0,
            sk_statistic={"rows": [[0.0]], "num_failed": 0},
            maneuvers={"mjd_tdb": [60000.0], "delta_v_mps": [1.0]},
        )
        assert resp.controlled_ephemeris is None
        assert resp.mu is None

    @pytest.mark.parametrize(
        ("status", "cause"),
        [
            (ConvergenceState.ITERATING, FailureCause.NONE),
            (ConvergenceState.CONVERGED, FailureCause.UNKNOWN),
        ],
    )
    def test_response_rejects_invalid_status_cause(self, status, cause):
        with pytest.raises(ValidationError):
            ControlOrbitResponse(
                status=status,
                cause=cause,
                message="非法状态",
                num_failed=0,
                sk_statistic={"rows": [], "num_failed": 0},
                maneuvers={"mjd_tdb": [], "delta_v_mps": []},
            )

    def test_control_request_accepts_mu(self):
        """mu 透传字段（画地月/L 点标注用）。"""
        req = ControlOrbitRequest(input_ephemeris="x", mu=Datum.DE421.mu)
        assert req.mu == pytest.approx(Datum.DE421.mu)
        # 缺省 None
        req_default = ControlOrbitRequest(input_ephemeris="x")
        assert req_default.mu is None


class TestWsbTransferDetailsContract:
    """WsbTransferDetails 字段类型契约（从 transfer/test_wsb.py 下沉）。

    字段类型契约不靠真 WSB 搜索验证，直接构造 dataclass 断言（ADR 0021：
    字段契约归 interface 类，零管线）。
    """

    def test_field_types(self):
        from e2m2e.algorithm.transfer import WsbSearchParams, WsbTransferDetails

        details = WsbTransferDetails(
            tli_epoch="2025-01-01T00:00:00",
            tof_sec=1e7,
            perilune_alt_km=100.0,
            perilune_vel_km_s=2.5,
            perilune_state=np.zeros(6),
            h2_kepler=-0.5,
            dv_departure_km_s=3.1,
            dv_arrival_km_s=0.8,
            n_candidates_searched=100,
            n_candidates_feasible=5,
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="收敛",
            search_params=WsbSearchParams(),
        )
        assert isinstance(details.tof_sec, float)
        assert isinstance(details.perilune_alt_km, float)
        assert isinstance(details.perilune_vel_km_s, float)
        assert isinstance(details.dv_departure_km_s, float)
        assert isinstance(details.dv_arrival_km_s, float)
        assert isinstance(details.h2_kepler, float)
        assert isinstance(details.n_candidates_searched, int)
        assert isinstance(details.n_candidates_feasible, int)
        assert details.status is ConvergenceState.CONVERGED
        assert details.cause is FailureCause.NONE
        assert details.message == "收敛"
        assert isinstance(details.search_params, WsbSearchParams)
