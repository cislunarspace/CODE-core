"""api/models.py 的公开输入、输出与状态契约测试。"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from api.conftest import control_orbit_business_parameters
from e2m2e.api.models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    FamilyGenerationRequest,
    FamilyGenerationResponse,
    NumericRange,
    PropagationRequest,
    SpacetimeTransformRequest,
    TransferDesignRequest,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import OrbitFamily

pytestmark = pytest.mark.interface


class TestDesignOrbitRequest:
    def test_defaults_and_elfo_defaults(self):
        dro = DesignOrbitRequest(orbit_type="DRO")
        assert dro.duration == 31557600.0
        assert dro.output_step == 3600.0
        assert dro.correction_method == "two_level"

        elfo = DesignOrbitRequest(orbit_type="ELFO", semi_major_axis=3000.0)
        assert elfo.duration == 5184000.0
        assert elfo.inclination == 75.0
        assert elfo.arg_of_pericenter == 270.0
        assert elfo.perilune_height == 200.0

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"duration": 0.0}, "duration"),
            ({"orbit_type": "NRHO", "perilune_height": 50.0}, "perilune_height"),
            (
                {"orbit_type": "DRO", "correction_velocity_tolerance": 0.5},
                "correction_velocity_tolerance",
            ),
            (
                {"orbit_type": "NRHO", "correction_velocity_tolerance": 0.5},
                "correction_velocity_tolerance",
            ),
            ({"orbit_type": "DRO", "amplitude_out": 80000.0}, "amplitude_out"),
            (
                {"orbit_type": "ELFO", "semi_major_axis": 3000.0, "amplitude_out": 80000.0},
                "amplitude_out",
            ),
        ],
    )
    def test_rejects_invalid_input(self, kwargs, field):
        kwargs.setdefault("orbit_type", "DRO")
        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(**kwargs)

    @pytest.mark.parametrize(
        ("orbit_type", "field", "minimum", "maximum", "minimum_inclusive"),
        [
            ("DRO", "amplitude", 1737.0, 110000.0, True),
            ("DPO", "amplitude", 1737.0, 110000.0, True),
            ("HALO", "amplitude", -73000.0, 73000.0, True),
            ("NRHO", "perilune_height", 100.0, 10000.0, True),
            ("L4", "amplitude_out", 0.0, 76000.0, False),
            ("L5", "amplitude_out", 0.0, 76000.0, False),
            ("AXIAL", "amplitude", -60000.0, 60000.0, True),
            ("L4_SPO", "amplitude", 1737.0, 200000.0, True),
            ("L5_SPO", "amplitude", 1737.0, 200000.0, True),
            ("L4_LPO", "amplitude", 1000.0, 110000.0, True),
            ("L5_LPO", "amplitude", 1000.0, 110000.0, True),
            ("L4_HORSESHOE", "amplitude", 50000.0, 110000.0, True),
            ("L5_HORSESHOE", "amplitude", 50000.0, 110000.0, True),
        ],
    )
    def test_public_ranges_match_validation(
        self, orbit_type, field, minimum, maximum, minimum_inclusive
    ):
        numeric_range = DesignOrbitRequest.valid_ranges(orbit_type)[field]
        assert isinstance(numeric_range, NumericRange)
        assert numeric_range.minimum == minimum
        assert numeric_range.maximum == maximum
        assert numeric_range.minimum_inclusive is minimum_inclusive
        assert numeric_range.maximum_inclusive

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

    @pytest.mark.parametrize("orbit_type", ["HALO", "NRHO", "DPO"])
    def test_unstable_family_defaults_to_segmented_silently(self, orbit_type):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            request = DesignOrbitRequest(orbit_type=orbit_type)
        assert request.correction_method == "segmented"

    @pytest.mark.parametrize("orbit_type", ["DRO", "LISSAJOUS", "L4", "AXIAL"])
    def test_stable_families_default_to_two_level(self, orbit_type):
        assert DesignOrbitRequest(orbit_type=orbit_type).correction_method == "two_level"

    @pytest.mark.parametrize("orbit_type", ["HALO", "NRHO", "DPO"])
    @pytest.mark.parametrize("method", ["two_level", "standard", "rust"])
    def test_unstable_family_conflicting_method_warns_and_rewrites(self, orbit_type, method):
        with pytest.warns(UserWarning, match=orbit_type):
            request = DesignOrbitRequest(orbit_type=orbit_type, correction_method=method)
        assert request.correction_method == "segmented"

    def test_unstable_family_explicit_segmented_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            request = DesignOrbitRequest(orbit_type="HALO", correction_method="segmented")
        assert request.correction_method == "segmented"

    def test_stable_family_keeps_any_explicit_method(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for method in ("segmented", "two_level", "standard", "rust"):
                request = DesignOrbitRequest(orbit_type="DRO", correction_method=method)
                assert request.correction_method == method

    def test_elfo_skips_correction_method_dispatch(self):
        # ELFO 不经星历修正，不参与规范化：值保持默认原样
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            request = DesignOrbitRequest(orbit_type="ELFO", semi_major_axis=3000.0)
        assert request.correction_method == "two_level"

    @pytest.mark.parametrize(
        ("orbit_type", "amplitude"),
        [
            ("L4_LPO", 50000.0),
            ("L5_LPO", 50000.0),
            ("L4_HORSESHOE", 100000.0),
            ("L5_HORSESHOE", 100000.0),
        ],
    )
    def test_lpo_and_horseshoe_defaults(self, orbit_type, amplitude):
        assert DesignOrbitRequest(orbit_type=orbit_type).amplitude == amplitude

    def test_lissajous_ranges_depend_on_collinear_point(self):
        default = DesignOrbitRequest.valid_ranges("LISSAJOUS")
        l1 = DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=1)
        l3 = DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=3)
        assert default["amplitude_out"].maximum == 7600.0
        assert l1["amplitude_out"].maximum == 7600.0
        assert not l1["amplitude_in"].minimum_inclusive
        assert l3["amplitude_in"].maximum == 100000.0
        assert l3["amplitude_out"].maximum == 100000.0
        with pytest.raises(ValueError, match="collinear_point"):
            DesignOrbitRequest.valid_ranges("LISSAJOUS", collinear_point=4)
        with pytest.raises(ValueError, match="字符串"):
            DesignOrbitRequest.valid_ranges(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["amplitude_in", "amplitude_out"])
    def test_lissajous_boundaries_match_public_ranges(self, field):
        l1 = DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 7600.0})
        l3 = DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=3, **{field: 80000.0})
        assert getattr(l1, field) == 7600.0
        assert getattr(l3, field) == 80000.0
        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 0.0})
        with pytest.raises(ValidationError, match=field):
            DesignOrbitRequest(orbit_type="LISSAJOUS", collinear_point=1, **{field: 80000.0})


class TestControlOrbitRequest:
    def test_algorithm_business_signature_is_present(self):
        business = control_orbit_business_parameters()
        assert set(business) <= set(ControlOrbitRequest.model_fields)

        request = ControlOrbitRequest(input_ephemeris="x")
        for name, parameter in business.items():
            assert getattr(request, name) == parameter.default

    def test_defaults_and_schema_ranges(self):
        request = ControlOrbitRequest(input_ephemeris="x")
        assert request.num_controls == 120
        assert request.num_monte_carlo == 5
        properties = ControlOrbitRequest.model_json_schema()["properties"]
        assert properties["num_controls"]["maximum"] == 10000
        assert properties["num_monte_carlo"]["maximum"] == 1000
        assert properties["control_interval"]["exclusiveMinimum"] == 0
        assert properties["position_accuracy"]["exclusiveMinimum"] == 0
        assert properties["earth_degree"]["minimum"] == 2
        assert properties["special_damping_factor"]["maximum"] == 1.0

    @pytest.mark.parametrize(
        ("field", "value", "accepted"),
        [
            ("control_mode", 7, 6),
            ("num_controls", 10001, 10000),
            ("num_monte_carlo", 1001, 1000),
        ],
    )
    def test_rejects_upper_bounds(self, field, value, accepted):
        with pytest.raises(ValidationError, match=field):
            ControlOrbitRequest(input_ephemeris="x", **{field: value})
        assert (
            getattr(ControlOrbitRequest(input_ephemeris="x", **{field: accepted}), field)
            == accepted
        )

    def test_control_mode_four_is_allowed_at_api_boundary(self):
        # mode 4 在 API 层放行，engine_layout 校验留算法层
        request = ControlOrbitRequest(input_ephemeris="x", control_mode=4)
        assert request.control_mode == 4

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dyb", [0.0] * 8),
            ("real_dyb", [0.0] * 10),
            ("srp_offset_m", [0.0] * 2),
            ("srp_torque", [0.0] * 4),
        ],
    )
    def test_rejects_invalid_vectors(self, field, value):
        with pytest.raises(ValidationError, match=field):
            ControlOrbitRequest(input_ephemeris="x", **{field: value})

    @pytest.mark.parametrize(
        "perturbation",
        [{"unknown": 1}, {"sun_body": 2}, {"solar_radiation": 3}],
    )
    def test_rejects_invalid_perturbations(self, perturbation):
        with pytest.raises(ValidationError, match="摄动"):
            ControlOrbitRequest(input_ephemeris="x", perturbation=perturbation)


class TestOtherRequests:
    def test_transfer_defaults(self):
        request = TransferDesignRequest(transfer_type="HMN", tli_epoch="2025-06-21T11:00:00")
        assert request.parking_alt_km == 200.0
        assert request.incl_deg == 28.5
        assert request.flight_path_deg == 0.0
        with pytest.raises(ValidationError, match="transfer_type"):
            TransferDesignRequest(transfer_type=123, tli_epoch="2025-06-21T11:00:00")

    def test_propagation_defaults_and_state_shape(self):
        request = PropagationRequest(
            initial_state=[0.0] * 6,
            epoch="2025-06-21T11:00:00",
            duration=3600.0,
        )
        assert request.output_step == 3600.0
        assert request.force_config is None
        with pytest.raises(ValidationError, match="initial_state"):
            PropagationRequest(
                initial_state=[0.0] * 5,
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
            )
        with pytest.raises(ValidationError, match="duration"):
            PropagationRequest(
                initial_state=[0.0] * 6,
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_spacetime_transform_defaults(self):
        request = SpacetimeTransformRequest(
            states=[[1.0] * 6],
            times=[0.0],
            transform_type="j2000_to_synodic",
            et0_jd=2459000.0,
        )
        assert request.ephemeris_path is None
        with pytest.raises(ValidationError, match="transform_type"):
            SpacetimeTransformRequest(
                states=[[1.0] * 6],
                times=[0.0],
                transform_type=123,
                et0_jd=2459000.0,
            )


class TestFamilyGenerationRequest:
    def test_halo_defaults(self):
        request = FamilyGenerationRequest(orbit_type="HALO")
        assert request.libration_point == 2
        assert request.max_amplitude_km == 30000.0
        assert request.n_orbits == 50

    def test_halo_defaults_and_point_dependent_limit(self):
        request = FamilyGenerationRequest(orbit_type="HALO", libration_point=1)
        assert request.max_amplitude_km == 25000.0
        assert request.libration_point == 1
        assert FamilyGenerationRequest.valid_ranges("HALO", libration_point=1)[
            "max_amplitude_km"
        ].maximum == pytest.approx(26908.0)
        assert FamilyGenerationRequest.valid_ranges("HALO")[
            "max_amplitude_km"
        ].maximum == pytest.approx(57660.0)
        assert FamilyGenerationRequest.valid_ranges("SPO")["libration_point"].minimum == 4

    def test_accepts_signed_amplitude_and_triangular_defaults(self):
        halo = FamilyGenerationRequest(
            orbit_type="HALO", libration_point=1, max_amplitude_km=-20000.0, n_orbits=5
        )
        assert halo.max_amplitude_km == -20000.0
        assert halo.n_orbits == 5
        assert FamilyGenerationRequest(orbit_type="LPO").libration_point == 4
        assert FamilyGenerationRequest(orbit_type="HORSESHOE").libration_point == 4

    @pytest.mark.parametrize(
        ("orbit_type", "libration_point"),
        [("HALO", 4), ("SPO", 1), ("DRO", 2)],
    )
    def test_rejects_inapplicable_family(self, orbit_type, libration_point):
        kwargs = {"orbit_type": orbit_type}
        if libration_point is not None:
            kwargs["libration_point"] = libration_point
        with pytest.raises(ValidationError, match="orbit_type|libration_point"):
            FamilyGenerationRequest(**kwargs)

    def test_rejects_zero_amplitude_and_nonpositive_count(self):
        with pytest.raises(ValidationError, match="max_amplitude_km"):
            FamilyGenerationRequest(orbit_type="HALO", max_amplitude_km=0.0)
        with pytest.raises(ValidationError, match="n_orbits"):
            FamilyGenerationRequest(orbit_type="HALO", n_orbits=0)
        with pytest.raises(ValidationError, match="libration_point"):
            FamilyGenerationRequest(orbit_type="HALO", libration_point=0)
        with pytest.raises(ValidationError, match="libration_point"):
            FamilyGenerationRequest(orbit_type="HALO", libration_point=6)
        with pytest.raises(ValidationError, match="max_amplitude_km"):
            FamilyGenerationRequest(orbit_type="HALO", libration_point=2, max_amplitude_km=60000.0)
        with pytest.raises(ValidationError, match="max_amplitude_km"):
            FamilyGenerationRequest(orbit_type="HALO", libration_point=1, max_amplitude_km=30000.0)
        with pytest.raises(ValueError, match="orbit_type"):
            FamilyGenerationRequest.valid_ranges("NOPE")

    def test_schema_exposes_family_specific_fields(self):
        schema = FamilyGenerationRequest.model_json_schema()
        properties = schema["properties"]
        for field in (
            "max_amplitude_km",
            "min_amplitude_km",
            "north_south",
            "perilune_height_max_km",
            "amplitude_in_km",
            "amplitude_out_km",
            "phase_in",
            "phase_out",
            "continuation_direction",
            "sampling_mode",
            "match_tolerance_km",
        ):
            assert field in properties
        assert "第一版仅" not in properties["orbit_type"]["description"]

    def test_non_halo_defaults(self):
        nrho = FamilyGenerationRequest(orbit_type="NRHO")
        assert nrho.libration_point == 2
        assert nrho.north_south == 2
        assert nrho.perilune_height_max_km == 20000.0
        assert nrho.continuation_direction == "toward-moon"
        assert nrho.sampling_mode == "halo-segment"

        axial = FamilyGenerationRequest(orbit_type="AXIAL")
        assert axial.max_amplitude_km == 10000.0
        assert axial.continuation_direction == "increase-amplitude"
        assert axial.sampling_mode == "fixed-vz0"

        lissajous = FamilyGenerationRequest(orbit_type="LISSAJOUS")
        assert lissajous.amplitude_in_km == 2500.0
        assert lissajous.amplitude_out_km == 7500.0
        assert lissajous.phase_in == 0.01
        assert lissajous.phase_out == 0.55
        assert lissajous.sampling_mode == "linear-amplitudes"

        spo = FamilyGenerationRequest(orbit_type="SPO")
        assert (spo.min_amplitude_km, spo.max_amplitude_km) == (2000.0, 60000.0)
        assert spo.continuation_direction == "decrease-x0"
        assert spo.sampling_mode == "full-period-pal"
        assert spo.match_tolerance_km == 20.0
        lpo = FamilyGenerationRequest(orbit_type="LPO")
        assert (lpo.min_amplitude_km, lpo.max_amplitude_km) == (2000.0, 110000.0)
        horseshoe = FamilyGenerationRequest(orbit_type="HORSESHOE")
        assert (horseshoe.min_amplitude_km, horseshoe.max_amplitude_km) == (50000.0, 110000.0)
        assert horseshoe.match_tolerance_km == 50.0

        dro = FamilyGenerationRequest(orbit_type="DRO")
        assert dro.libration_point is None
        assert (dro.min_amplitude_km, dro.max_amplitude_km) == (2000.0, 60000.0)
        assert dro.sampling_mode == "natural-x0"

    def test_valid_ranges_are_family_specific(self):
        nrho = FamilyGenerationRequest.valid_ranges("NRHO")
        assert nrho["perilune_height_max_km"].format_interval() == "[1000.0, 40000.0]"
        assert nrho["north_south"].contains(1)

        axial = FamilyGenerationRequest.valid_ranges("AXIAL")
        assert axial["max_amplitude_km"].format_interval() == "[-60000.0, 60000.0]"
        assert not axial["max_amplitude_km"].contains(0.0)

        l12 = FamilyGenerationRequest.valid_ranges("LISSAJOUS", libration_point=2)
        assert l12["amplitude_in_km"].maximum == 7600.0
        assert l12["phase_in"].format_interval() == "[0.0, 1.0]"
        l3 = FamilyGenerationRequest.valid_ranges("LISSAJOUS", libration_point=3)
        assert l3["amplitude_in_km"].maximum == 100000.0

        horseshoe = FamilyGenerationRequest.valid_ranges("HORSESHOE")
        assert horseshoe["min_amplitude_km"].minimum == 50000.0
        assert horseshoe["match_tolerance_km"].minimum_inclusive is False

        # DRO 是月心族：无平动点范围，振幅包络与单轨 DRO 一致
        dro = FamilyGenerationRequest.valid_ranges("DRO")
        assert "libration_point" not in dro
        assert dro["min_amplitude_km"].format_interval() == "[1737.0, 110000.0]"
        with pytest.raises(ValueError, match="不绑定平动点"):
            FamilyGenerationRequest.valid_ranges("DRO", libration_point=2)
        assert FamilyGenerationRequest.valid_options("DRO")["sampling_mode"] == ("natural-x0",)

        options = FamilyGenerationRequest.valid_options("LPO")
        assert options["continuation_direction"] == ("decrease-x0", "increase-x0")
        assert options["sampling_mode"] == ("full-period-pal",)

    def test_rejects_invalid_family_specific_fields(self):
        with pytest.raises(ValidationError, match="perilune_height_max_km"):
            FamilyGenerationRequest(orbit_type="NRHO", perilune_height_max_km=500.0)
        with pytest.raises(ValidationError, match="max_amplitude_km 不能为 0"):
            FamilyGenerationRequest(orbit_type="AXIAL", max_amplitude_km=0.0)
        with pytest.raises(ValidationError, match="max_amplitude_km"):
            FamilyGenerationRequest(orbit_type="AXIAL", max_amplitude_km=70000.0)
        with pytest.raises(ValidationError, match="amplitude_in_km"):
            FamilyGenerationRequest(
                orbit_type="LISSAJOUS", libration_point=2, amplitude_in_km=8000.0
            )
        with pytest.raises(ValidationError, match="min_amplitude_km"):
            FamilyGenerationRequest(orbit_type="HORSESHOE", min_amplitude_km=30000.0)
        with pytest.raises(ValidationError, match="min_amplitude_km 必须小于"):
            FamilyGenerationRequest(
                orbit_type="LPO", min_amplitude_km=60000.0, max_amplitude_km=30000.0
            )
        with pytest.raises(ValidationError, match="不适用字段"):
            FamilyGenerationRequest(orbit_type="NRHO", max_amplitude_km=5000.0)
        with pytest.raises(ValidationError, match="sampling_mode"):
            FamilyGenerationRequest(orbit_type="LISSAJOUS", sampling_mode="grid")
        with pytest.raises(ValidationError, match="continuation_direction"):
            FamilyGenerationRequest(orbit_type="SPO", continuation_direction="sideways")
        with pytest.raises(ValidationError, match="min_amplitude_km"):
            FamilyGenerationRequest(orbit_type="SPO", min_amplitude_km=1000.0)
        with pytest.raises(ValidationError, match="libration_point"):
            FamilyGenerationRequest(orbit_type="LPO", libration_point=2)
        with pytest.raises(ValidationError, match="match_tolerance_km"):
            FamilyGenerationRequest(orbit_type="HORSESHOE", match_tolerance_km=0.0)
        with pytest.raises(ValidationError, match="不绑定平动点"):
            FamilyGenerationRequest(orbit_type="DRO", libration_point=1)
        with pytest.raises(ValidationError, match="不适用字段"):
            FamilyGenerationRequest(orbit_type="DRO", north_south=1)
        with pytest.raises(ValidationError, match="不适用字段"):
            FamilyGenerationRequest(orbit_type="DRO", continuation_direction="decrease-x0")
        with pytest.raises(ValidationError, match="min_amplitude_km"):
            FamilyGenerationRequest(orbit_type="DRO", min_amplitude_km=1700.0)
        with pytest.raises(ValidationError, match="max_amplitude_km"):
            FamilyGenerationRequest(orbit_type="DRO", max_amplitude_km=120000.0)
        with pytest.raises(ValidationError, match="min_amplitude_km 必须小于"):
            FamilyGenerationRequest(
                orbit_type="DRO", min_amplitude_km=30000.0, max_amplitude_km=20000.0
            )
        with pytest.raises(ValidationError, match="sampling_mode"):
            FamilyGenerationRequest(orbit_type="DRO", sampling_mode="grid")


class TestResponses:
    def test_design_response_serializes_public_geometry(self):
        response = DesignOrbitResponse(
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
            mu=Datum.DE421.mu,
            states=[[0.0] * 6],
            times=[0.0],
            ephemeris={"position_km": [[0.0, 0.0, 0.0]]},
        )
        dumped = response.model_dump()
        assert dumped["mu"] == pytest.approx(Datum.DE421.mu)
        assert dumped["states"] == [[0.0] * 6]

        optional = response.model_copy(update={"mu": None, "states": [], "times": []})
        assert optional.mu is None
        assert optional.states == []

    def test_control_response_allows_missing_optional_geometry(self):
        response = ControlOrbitResponse(
            status="converged",
            cause="none",
            message="任务完成",
            num_failed=0,
            sk_statistic={"rows": [], "num_failed": 0},
            maneuvers={"mjd_tdb": [], "delta_v_mps": []},
        )
        assert response.controlled_ephemeris is None
        assert response.mu is None
        request = ControlOrbitRequest(input_ephemeris="x", mu=Datum.DE421.mu)
        assert request.mu == pytest.approx(Datum.DE421.mu)
        assert ControlOrbitRequest(input_ephemeris="x").mu is None

    @pytest.mark.parametrize(
        ("status", "cause"),
        [
            (ConvergenceState.ITERATING, FailureCause.NONE),
            (ConvergenceState.CONVERGED, FailureCause.UNKNOWN),
        ],
    )
    def test_rejects_invalid_status_triplet(self, status, cause):
        with pytest.raises(ValidationError):
            ControlOrbitResponse(
                status=status,
                cause=cause,
                message="非法状态",
                num_failed=0,
                sk_statistic={"rows": [], "num_failed": 0},
                maneuvers={"mjd_tdb": [], "delta_v_mps": []},
            )


class TestFamilyGenerationResponse:
    def test_is_pydantic_response_and_orbit_family(self):
        response = FamilyGenerationResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="轨道族生成完成",
            orbits=[],
            family_type="halo",
            metadata={"periodicity": "periodic"},
            requested_members=2,
            generated_members=0,
        )

        assert isinstance(response, OrbitFamily)
        assert response.periodicity == "periodic"
        assert response.model_dump()["status"] is ConvergenceState.CONVERGED

    def test_rejects_member_count_mismatch(self):
        with pytest.raises(ValidationError):
            FamilyGenerationResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="轨道族生成完成",
                orbits=[],
                family_type="halo",
                metadata={},
                requested_members=2,
                generated_members=1,
            )
