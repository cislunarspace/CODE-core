"""站保任务结果状态三元组契约测试。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.station_keeping.controller import ControlOrbitResult
from e2m2e.algorithm.station_keeping.monte_carlo import MonteCarloResult
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types import ManeuverTable, SKStatistic

pytestmark = pytest.mark.orchestration


def _result(status: ConvergenceState, cause: FailureCause) -> ControlOrbitResult:
    raw = MonteCarloResult(
        total_delta_v=np.empty(0),
        max_delta_v=np.empty(0),
        failed_mask=np.empty(0, dtype=bool),
        num_failed=0,
    )
    return ControlOrbitResult(
        sk_statistic=SKStatistic(rows=np.empty((0, 2)), num_failed=0),
        num_failed=0,
        maneuvers=ManeuverTable(mjd_tdb=np.empty(0), delta_v_mps=np.empty(0)),
        controlled_ephemeris=None,
        raw=raw,
        status=status,
        cause=cause,
        message="非法状态",
    )


@pytest.mark.parametrize(
    ("status", "cause"),
    [
        (ConvergenceState.ITERATING, FailureCause.NONE),
        (ConvergenceState.CONVERGED, FailureCause.UNKNOWN),
    ],
)
def test_control_orbit_result_rejects_invalid_status_cause(status, cause):
    with pytest.raises(ValueError):
        _result(status, cause)
