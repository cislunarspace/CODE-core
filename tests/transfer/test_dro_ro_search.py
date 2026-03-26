"""
DRO-RO 转移搜索模块测试：最小距离、Δv 字段、可行解判定。
"""

import numpy as np
import pytest

from e2m2e.core import Orbit, CR3BP_System, CR3BP_Dynamics
from e2m2e.transfer import DROTransferSearch
from e2m2e.transfer.transfer_base import DEFAULT_MIN_DISTANCE_THRESHOLD_DU


@pytest.fixture
def system():
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system):
    d = CR3BP_Dynamics(system)
    d.integrator = "DOP853"
    d.rtol = d.atol = 1e-9
    d.max_step = 0.05
    return d


@pytest.fixture
def searcher(system, dynamics):
    s = DROTransferSearch(system, dynamics)
    s.alpha_min = 0.5
    s.alpha_max = 2.5
    s.n_alpha = 5
    s.n_departure = 10
    s.max_transfer_time = 0.5
    s.intersection_threshold = 1e-3
    s.min_distance_threshold = DEFAULT_MIN_DISTANCE_THRESHOLD_DU
    s.collision_earth_radius = 5e-4
    s.collision_moon_radius = 3e-4
    s.integration_dt = 0.02
    return s


def _simple_orbit(n: int = 80) -> Orbit:
    t = np.linspace(0, 6.28, n)
    x = 0.9 + 0.08 * np.cos(t)
    y = 0.08 * np.sin(t)
    z = np.zeros_like(t)
    vx = -0.08 * np.sin(t)
    vy = 0.08 * np.cos(t)
    vz = np.zeros_like(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.period = float(t[-1])
    return orbit


class TestComputeMinDistance:
    def test_returns_traj_and_orbit_indices(self, searcher):
        dro = _simple_orbit(60)
        ro = _simple_orbit(50)
        traj = dro.states[:40]
        md, ti, oi = searcher._compute_min_distance(traj, ro)
        assert md >= 0
        assert 0 <= ti < len(traj)
        assert 0 <= oi < len(ro.states)

    def test_compute_min_distance_to_orbit_public_api(self, searcher):
        dro = _simple_orbit(60)
        ro = _simple_orbit(50)
        traj = dro.states[:40]
        md, ti = searcher.compute_min_distance_to_orbit(traj, ro)
        md2, ti2, _ = searcher._compute_min_distance(traj, ro)
        assert md == md2
        assert ti == ti2


class TestIsFeasible:
    def test_local_minimum_alone_not_feasible_if_distance_large(self, searcher):
        """局部极小但距离仍大于阈值时不应判为可行。"""
        r = {
            "collision_found": False,
            "intersection_found": False,
            "min_distance": 0.6,
            "local_minimum_found": True,
            "local_minimum_distance": 0.6,
        }
        assert not searcher._is_feasible(r)

    def test_local_minimum_feasible_when_below_threshold(self, searcher):
        below = DEFAULT_MIN_DISTANCE_THRESHOLD_DU * 0.5
        r = {
            "collision_found": False,
            "intersection_found": False,
            "min_distance": 0.2,
            "local_minimum_found": True,
            "local_minimum_distance": below,
        }
        assert searcher._is_feasible(r)

    def test_intersection_always_feasible_if_no_collision(self, searcher):
        r = {
            "collision_found": False,
            "intersection_found": True,
            "min_distance": 1.0,
            "local_minimum_found": False,
            "local_minimum_distance": float("inf"),
        }
        assert searcher._is_feasible(r)

    def test_collision_infeasible(self, searcher):
        r = {
            "collision_found": True,
            "intersection_found": True,
            "min_distance": 0.0,
        }
        assert not searcher._is_feasible(r)


class TestSearchSingleDepartureDv:
    def test_success_includes_dv_fields(self, searcher, dynamics):
        dro = _simple_orbit(100)
        ro = _simple_orbit(80)
        dep = dro.states[0]
        t0 = float(dro.times[0])
        results = searcher._search_single_departure(dep, t0, ro, verbose=False)
        assert len(results) == 5
        ok = [r for r in results if r.get("success")]
        assert ok
        r0 = ok[0]
        assert r0["dv_departure"] is not None
        assert r0["dv_departure"] >= 0
        assert r0["dv_insertion"] is not None
        assert r0["min_distance_orbit_idx"] is not None
