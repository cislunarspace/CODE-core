"""Q-law Rust 反馈积分的公开接口回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig
from e2m2e.algorithm.transfer.qlaw import qlaw_guess
from e2m2e.exceptions import PropagationFailure

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

MU = 398600.435507


def _system_forces():
    return SimpleNamespace(origin="EARTH"), [PointMassGravity("EARTH", mu=MU)]


def test_qlaw_rust_backend_returns_solver_initial_guess():
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    y, segments, q_history, final_state = qlaw_guess(
        system,
        forces,
        EngineConfig(t_max=0.5, isp=3000.0),
        np.array([r0, 0.0, 0.0, 0.0, v0, 0.0]),
        1000.0,
        (7020.0, 0.0, 0.0),
        0.0,
        6 * 3600.0,
        3,
        step=600.0,
    )

    assert y.shape == (9,)
    assert len(segments) == 3
    assert q_history.shape == (3,)
    assert final_state.shape == (7,)
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(q_history))
    assert np.all(np.isfinite(final_state))
    for segment in segments:
        assert segment.throttle == 1.0
        assert np.isclose(np.linalg.norm(segment.direction), 1.0)


def test_qlaw_rust_backend_preserves_propagation_failure():
    system, forces = _system_forces()
    init = np.array([1e-6, 0.0, 0.0, 0.0, 7.5, 0.0])

    with pytest.raises(PropagationFailure):
        qlaw_guess(
            system,
            forces,
            EngineConfig(t_max=0.5, isp=3000.0),
            init,
            1000.0,
            (8000.0, 0.0, 0.0),
            0.0,
            5 * 86400.0,
            5,
            step=60.0,
        )
