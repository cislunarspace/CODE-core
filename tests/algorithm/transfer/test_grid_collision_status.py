"""网格搜索碰撞格进失败侧回归测试。

碰撞格必须标记 ``status == COLLISION`` 并进失败侧：不进可行候选、
不回传轨迹。碰撞格若被下游按 success 过滤当作有效解会混入可行集。
当前基线已用类型化 ``TransferCandidateResult``（status/cause 枚举），
本测试锁定该语义，防止回归到"碰撞格当成功"。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import TransferSearch
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


@pytest.fixture
def system() -> CR3BP_System:
    return CR3BP_System(mu=Datum.DE421.mu, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system: CR3BP_System) -> CR3BP_Dynamics:
    d = CR3BP_Dynamics(system)
    d.integrator = "DOP853"
    d.rtol = d.atol = 1e-9
    d.max_step = 0.05
    return d


@pytest.fixture
def searcher(system: CR3BP_System, dynamics: CR3BP_Dynamics) -> TransferSearch:
    s = TransferSearch(dynamics)
    s.alpha_min = 1.0
    s.alpha_max = 1.0
    s.n_alpha = 1
    s.n_departure = 1
    s.max_transfer_time = 1.0
    s.intersection_threshold = 0.05
    s.min_distance_threshold = 0.1
    s.collision_earth_radius = 1e-4
    s.collision_moon_radius = 1e-4
    s.integration_dt = 0.05
    return s


def _arrival_orbit() -> Orbit:
    target = np.array([1.0, 0.0, 0.0])
    return Orbit(
        states=np.tile(np.concatenate([target, [0.0, 0.0, 0.0]]), (8, 1)),
        times=np.linspace(0.0, 1.0, 8),
    )


def _run_single_candidate(
    searcher: TransferSearch,
    monkeypatch: pytest.MonkeyPatch,
    collision: tuple[bool, str | None, int],
):
    """固定轨迹（远离目标）+ 可注入碰撞的确定性单候选评估。"""
    n = 6
    traj = np.zeros((n, 6))
    traj[:, 0] = 5.0  # 远离 target x=1.0（不交 / 不近）
    traj_times = np.linspace(0.0, 1.0, n)

    def _fake_integrate(self, initial_state, t_end, dt):
        return traj, traj_times

    def _check_collision(self, traj_states):
        return collision

    def _no_local_min(self, traj_states, arrival_orbit):
        return False, float("inf"), -1

    monkeypatch.setattr(TransferSearch, "_forward_integrate", _fake_integrate, raising=True)
    monkeypatch.setattr(TransferSearch, "_check_collision", _check_collision, raising=True)
    monkeypatch.setattr(TransferSearch, "_detect_local_minimum", _no_local_min, raising=True)

    departure_state = np.array([1.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    results = searcher._search_single_departure(
        departure_state=departure_state,
        departure_time=0.0,
        arrival_orbit=_arrival_orbit(),
        verbose=False,
    )
    assert len(results) == 1
    return results[0]


class TestCollisionCellEntersFailureSide:
    def test_collision_cell_flagged_collision(self, searcher, monkeypatch):
        """碰撞格 status==COLLISION + cause==BODY_COLLISION + 碰撞字段。"""
        r = _run_single_candidate(searcher, monkeypatch, (True, "earth", 2))

        assert r.status is ConvergenceState.COLLISION
        assert r.cause is FailureCause.BODY_COLLISION
        assert r.collision_found is True
        assert r.collision_body == "earth"
        assert r.collision_idx == 2

    def test_collision_cell_excluded_from_feasible(self, searcher, monkeypatch):
        """碰撞格进失败侧：不可行、不回传轨迹。"""
        r = _run_single_candidate(searcher, monkeypatch, (True, "moon", 3))

        assert searcher._is_feasible(r) is False
        # item ③：非成功候选不回传轨迹（省内存），下游按 status 过滤
        assert r.transfer_trajectory is None
        assert r.transfer_times is None

    def test_collision_cell_not_listed_in_feasible_results(self, searcher, monkeypatch):
        """碰撞格不出现在 get_feasible_results（失败侧，不参与选优）。"""
        r = _run_single_candidate(searcher, monkeypatch, (True, "earth", 1))
        searcher._search_results = [r]

        assert searcher.get_feasible_results() == []


class TestNonCollisionInfeasibleStillFailureSide:
    def test_no_intersection_cell_infeasible(self, searcher, monkeypatch):
        """对照：无交点且不碰撞的格 status==INFEASIBLE（同样不是成功）。"""
        r = _run_single_candidate(searcher, monkeypatch, (False, None, -1))

        assert r.status is ConvergenceState.INFEASIBLE
        assert r.cause is FailureCause.NO_INTERSECTION
        assert searcher._is_feasible(r) is False
