"""api/models.py 的公开输入、输出与状态契约测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.conftest import control_orbit_business_parameters
from e2m2e.api.models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    FamilyGenerationRequest,
    NumericRange,
    PropagationRequest,
    SpacetimeTransformRequest,
    TransferDesignRequest,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause

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
            ("L4_LPO", "amplitude", 1000.0, 200000.0, True),
            ("L5_LPO", "amplitude", 1000.0, 200000.0, True),
            ("L4_HORSESHOE", "amplitude", 50000.0, 200000.0, True),
            ("L5_HORSESHOE", "amplitude", 50000.0, 200000.0, True),
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
        [("HALO", 4), ("SPO", 1), ("DRO", None)],
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
