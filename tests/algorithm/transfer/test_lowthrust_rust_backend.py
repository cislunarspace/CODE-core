"""低推力求解器 Rust 数值内核与 Python 参照路径的等价性测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig, LowThrustCollocation, LowThrustShooting

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

MU = 398600.435507


def _problem(backend: str, tf: float = 1200.0):
    system = SimpleNamespace(origin="EARTH")
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    initial_state = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    forces = [PointMassGravity("EARTH", mu=MU)]
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    kwargs = dict(
        system=system,
        forces=forces,
        engine=engine,
        initial_state=initial_state,
        initial_mass=1000.0,
        target_state=initial_state.copy(),
        t0=0.0,
        tf=tf,
        backend=backend,
    )
    return LowThrustShooting(**kwargs), LowThrustCollocation(**kwargs)


def test_shooting_rust_evaluator_matches_python_reference():
    rust, _ = _problem("rust")
    python, _ = _problem("python")
    controls = np.array([0.4, 1.2, 0.1, 0.7, 1.4, -0.2])

    rust_time, rust_states = rust._propagate_chain(controls)
    python_time, python_states = python._propagate_chain(controls)
    rust_mass, rust_rv, rust_jac = rust._propagate_chain_with_jacobian(controls)
    python_mass, python_rv, python_jac = python._propagate_chain_with_jacobian(controls)

    np.testing.assert_allclose(rust_time, python_time, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rust_states, python_states, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rust_mass, python_mass, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rust_rv, python_rv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rust_jac, python_jac, rtol=1e-12, atol=1e-12)


def test_shooting_rust_evaluator_accepts_subsecond_segment():
    rust, _ = _problem("rust", tf=0.5)
    controls = np.array([0.4, 1.2, 0.1])

    time, states = rust._propagate_chain(controls)

    assert time[-1] == pytest.approx(0.5)
    assert states.shape == (2, 7)
    assert np.all(np.isfinite(states))


def test_collocation_rust_defects_match_python_reference():
    _, rust = _problem("rust")
    _, python = _problem("python")
    z = python._default_z0(2)

    np.testing.assert_allclose(
        rust._defect_constraints(z, n_segments=2),
        python._defect_constraints(z, n_segments=2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_lowthrust_solver_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        _problem("unsupported")
