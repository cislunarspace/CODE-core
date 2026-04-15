"""
Requirement: Transfer Optimization Interface for e2m2e

This test case defines the requirements for the Transfer class in e2m2e.transfer.
The Transfer class provides an interface for DRO-RO transfer trajectory optimization
using NLP methods (Cui et al. 2025).

=== Usage Example (Expected API) ===

    from e2m2e.transfer import Transfer, TransferConfig
    from e2m2e.core import CR3BP_System, CR3BP_Dynamics

    # Setup system and dynamics
    system = CR3BP_System(mu=0.01215, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system=system)
    dynamics.integrator = "DOP853"
    dynamics.rtol = 1e-12
    dynamics.atol = 1e-12

    # Load orbits
    dro_orbit = load_orbit_from_json("dro_31.json")
    ro_orbit = load_orbit_from_json("ro_31.json")
    ro_orbit.period = ro_json["properties"]["period"]

    # Create Transfer instance
    transfer = Transfer(dynamics)

    # Set orbits
    transfer.set_orbit(start=dro_orbit, end=ro_orbit)

    # Optimize - auto sample departure state
    result = transfer.optimize(
        initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
        alpha_range=(0.5, 2.5),
        use_relaxed_velocity=True,
        velocity_angle_tol=0.05,
    )

    # Or with manual departure_state
    result = transfer.optimize(
        initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
        alpha_range=(0.5, 2.5),
        departure_state=np.array([x, y, z, vx, vy, vz]),
        t_ins_range=(t0, t0 + ro_orbit.period),
        use_relaxed_velocity=True,
        velocity_angle_tol=0.05,
    )

=== Transfer Instance Fields ===

    transfer.dynamics          # CR3BP_Dynamics instance (微分求解)
    transfer.departure_orbit   # Orbit: DRO orbit
    transfer.arrival_orbit     # Orbit: RO orbit
    transfer.config            # TransferConfig: optimization configuration
    transfer.result            # TransferOptimizationResult: latest optimization result

=== TransferConfig Fields ===

    alpha_min: float = 0.5
    alpha_max: float = 2.5
    earth_radius: float = 200.0 / DU
    moon_radius: float = 100.0 / DU
    use_relaxed_velocity: bool = True
    velocity_angle_tol: float = 0.05
    use_copt: bool = False
    fallback_to_scipy: bool = True

=== TransferOptimizationResult Fields ===

    success: bool                          # Optimization success flag
    message: str                           # Solver message
    departure_state: np.ndarray            # [6] Departure state (x,y,z,vx,vy,vz)
    departure_alpha: float                 # Tangential velocity ratio at departure
    departure_beta: float                  # Normal velocity ratio at departure
    insertion_state: np.ndarray            # [6] Insertion state on RO
    final_state: np.ndarray                # [6] Final state after insertion
    delta_v1: float                        # Departure impulse magnitude
    delta_v2: float                        # Insertion impulse magnitude
    total_delta_v: float                   # Total delta-v (objective)
    transfer_time: float                   # Transfer duration T
    t_ins: float                           # Insertion time on RO
    transfer_trajectory: np.ndarray        # [n_steps, 6] Full transfer trajectory
    transfer_trajectory_times: np.ndarray  # [n_steps] Time values for trajectory
    constraints_violation: float            # Max constraint violation
"""

import json
from pathlib import Path

import numpy as np
import pytest


class TestTransferCreation:
    """Test Transfer class instantiation and configuration."""

    def test_transfer_creation_with_dynamics(self, dynamics):
        """Transfer should be created with a dynamics instance."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)

        assert transfer.dynamics is not None
        assert transfer.departure_orbit is None
        assert transfer.arrival_orbit is None
        assert transfer.result is None

    def test_transfer_set_orbits(self, dynamics, dro_orbit, ro_orbit):
        """Transfer.set_orbit() should accept start (departure) and end (arrival) orbits."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        assert transfer.departure_orbit is dro_orbit
        assert transfer.arrival_orbit is ro_orbit

    def test_transfer_config_fields(self, dynamics):
        """Transfer.config should expose optimization configuration."""
        from e2m2e.transfer import Transfer, TransferConfig

        transfer = Transfer(dynamics)

        assert hasattr(transfer, "config")
        assert isinstance(transfer.config, TransferConfig)
        assert transfer.config.alpha_min == 0.5
        assert transfer.config.alpha_max == 2.5


class TestTransferOptimization:
    """Test Transfer.optimize() method."""

    def test_optimize_with_auto_departure_sampling(self, dynamics, dro_orbit, ro_orbit):
        """optimize() should work without explicit departure_state (auto-sample)."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "total_delta_v")
        assert hasattr(result, "departure_state")
        assert hasattr(result, "insertion_state")

    def test_optimize_with_manual_departure_state(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """optimize() should accept manual departure_state parameter."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result is not None
        np.testing.assert_array_almost_equal(result.departure_state, dro_departure_state)

    def test_optimize_with_t_ins_range(self, dynamics, dro_orbit, ro_orbit, dro_departure_state):
        """optimize() should accept t_ins_range parameter."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        t0 = ro_orbit.times[0]
        t_ins_range = (t0, t0 + ro_orbit.period)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            t_ins_range=t_ins_range,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result is not None
        assert t_ins_range[0] <= result.t_ins <= t_ins_range[1]

    def test_optimize_default_t_ins_range_full_period(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """When t_ins_range is not specified, default to full RO period."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        t0 = ro_orbit.times[0]
        t_ins_range_default = (t0, t0 + ro_orbit.period)
        assert t_ins_range_default[0] <= result.t_ins <= t_ins_range_default[1]


class TestTransferOptimizationResult:
    """Test TransferOptimizationResult contains all expected fields."""

    def test_result_contains_departure_info(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """Result should contain departure position and velocity information."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result.departure_state is not None
        assert result.departure_state.shape == (6,)
        assert hasattr(result, "departure_alpha")
        assert hasattr(result, "departure_beta")
        assert hasattr(result, "delta_v1")

    def test_result_contains_insertion_info(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """Result should contain arrival/insertion position and velocity change."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result.insertion_state is not None
        assert result.insertion_state.shape == (6,)
        assert hasattr(result, "final_state")
        assert hasattr(result, "delta_v2")
        assert hasattr(result, "t_ins")

    def test_result_contains_transfer_trajectory(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """Result should contain full transfer trajectory."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        assert result.transfer_trajectory is not None
        assert result.transfer_trajectory.ndim == 2
        assert result.transfer_trajectory.shape[1] == 6
        assert hasattr(result, "transfer_trajectory_times")
        assert len(result.transfer_trajectory_times) == len(result.transfer_trajectory)

    def test_result_total_delta_v_equals_sum(
        self, dynamics, dro_orbit, ro_orbit, dro_departure_state
    ):
        """total_delta_v should equal delta_v1 + delta_v2."""
        from e2m2e.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        result = transfer.optimize(
            initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
            alpha_range=(0.5, 2.5),
            departure_state=dro_departure_state,
            use_relaxed_velocity=True,
            velocity_angle_tol=0.05,
        )

        np.testing.assert_almost_equal(result.total_delta_v, result.delta_v1 + result.delta_v2)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def dro_file(project_root):
    return project_root / "output/dro/dro_31_3857117441.json"


@pytest.fixture
def ro_file(project_root):
    return project_root / "output/ro/ro_31_3857122799.json"


@pytest.fixture
def dro_orbit(dro_file):
    pytest.importorskip("_dro_data", reason="DRO data file not available")
    if not dro_file.exists():
        pytest.skip("DRO orbit data file not found")
    from e2m2e.transfer import load_orbit_from_json

    return load_orbit_from_json(str(dro_file))


@pytest.fixture
def ro_orbit(ro_file):
    if not ro_file.exists():
        pytest.skip("RO orbit data file not found")
    from e2m2e.transfer import load_orbit_from_json

    with open(ro_file, encoding="utf-8") as f:
        ro_json = json.load(f)

    orbit = load_orbit_from_json(str(ro_file))
    if "properties" in ro_json and "period" in ro_json["properties"]:
        orbit.period = float(ro_json["properties"]["period"])
    return orbit


@pytest.fixture
def dynamics():
    MU = 1.21506683e-2

    from e2m2e.core import CR3BP_Dynamics, CR3BP_System

    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    dyn = CR3BP_Dynamics(system=system)
    dyn.integrator = "DOP853"
    dyn.rtol = 1e-12
    dyn.atol = 1e-12
    dyn.max_step = 1.0 / (24.0 * 384405.0 / 26970.0 * 2.0 * np.pi / 27.321661)
    return dyn


@pytest.fixture
def dro_departure_state():
    return np.array(
        [
            -0.8748418222113017,
            0.0,
            0.0,
            0.0,
            -0.07906694836916197,
            0.0,
        ]
    )
