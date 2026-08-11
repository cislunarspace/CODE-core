"""设计任务结果状态三元组契约测试。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.design.design_orbit import OrbitDesignResult
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.trajectory import EphemerisTable

pytestmark = pytest.mark.orchestration


def _ephemeris() -> EphemerisTable:
    return EphemerisTable(
        year=np.array([2024]),
        month=np.array([1]),
        day=np.array([1]),
        hour=np.array([0]),
        minute=np.array([0]),
        second=np.array([0.0]),
        position_km=np.zeros((1, 3)),
        velocity_mps=np.zeros((1, 3)),
        synodic_position=np.zeros((1, 3)),
    )


@pytest.mark.parametrize(
    ("status", "cause"),
    [
        (ConvergenceState.ITERATING, FailureCause.NONE),
        (ConvergenceState.CONVERGED, FailureCause.UNKNOWN),
    ],
)
def test_orbit_design_result_rejects_invalid_status_cause(status, cause):
    with pytest.raises(ValueError):
        OrbitDesignResult(
            orbit_type="DRO",
            epoch_utc="2024-01-01T00:00:00",
            duration_day=1.0,
            output_step_sec=3600.0,
            initial_state=np.zeros(6),
            ephemeris=_ephemeris(),
            cr3bp_orbit=None,
            cr3bp_jacobi=0.0,
            correction=None,
            force_config={},
            status=status,
            cause=cause,
            message="非法状态",
        )
