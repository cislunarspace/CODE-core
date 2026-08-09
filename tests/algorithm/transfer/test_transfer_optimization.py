"""Transfer 优化接口需求定义与测试。

覆盖 Transfer 创建、轨道设置、optimize 调用与结果字段。
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.orchestration


class TestTransferCreation:
    """Test Transfer class instantiation and configuration."""

    def test_transfer_creation_with_dynamics(self, dynamics):
        """Transfer should be created with a dynamics instance."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)

        assert transfer.dynamics is not None
        assert transfer.departure_orbit is None
        assert transfer.arrival_orbit is None
        assert transfer.result is None

    def test_transfer_set_orbits(self, dynamics, dro_orbit, ro_orbit):
        """Transfer.set_orbit() should accept start (departure) and end (arrival) orbits."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        assert transfer.departure_orbit is dro_orbit
        assert transfer.arrival_orbit is ro_orbit

    def test_transfer_config_fields(self, dynamics):
        """Transfer.config should expose optimization configuration."""
        from e2m2e.algorithm.transfer import Transfer, TransferConfig

        transfer = Transfer(dynamics)

        assert hasattr(transfer, "config")
        assert isinstance(transfer.config, TransferConfig)
        assert transfer.config.nlp_alpha_min == 0.5
        assert transfer.config.nlp_alpha_max == 2.5


class TestTransferOptimization:
    """Test Transfer.optimize() method."""

    def test_optimize_with_auto_departure_sampling(self, dynamics, dro_orbit, ro_orbit):
        """optimize() should work without explicit departure_state (auto-sample)."""
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
        from e2m2e.algorithm.transfer import Transfer

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
    from e2m2e.algorithm.transfer import load_orbit_from_json

    return load_orbit_from_json(str(dro_file))


@pytest.fixture
def ro_orbit(ro_file):
    if not ro_file.exists():
        pytest.skip("RO orbit data file not found")
    from e2m2e.algorithm.transfer import load_orbit_from_json

    with open(ro_file, encoding="utf-8") as f:
        ro_json = json.load(f)

    orbit = load_orbit_from_json(str(ro_file))
    if "properties" in ro_json and "period" in ro_json["properties"]:
        orbit.period = float(ro_json["properties"]["period"])
    return orbit


@pytest.fixture
def dynamics():
    MU = 1.21506683e-2

    from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System

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
