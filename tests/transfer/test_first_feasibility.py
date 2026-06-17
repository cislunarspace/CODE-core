"""首次可行性字段单元测试。

覆盖 _compute_distance_series 完整距离序列与 search 新字段写入。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit
from e2m2e.transfer import DROTransferSearch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def system() -> CR3BP_System:
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system: CR3BP_System) -> CR3BP_Dynamics:
    d = CR3BP_Dynamics(system)
    d.integrator = "DOP853"
    d.rtol = d.atol = 1e-9
    d.max_step = 0.05
    return d


@pytest.fixture
def searcher(system: CR3BP_System, dynamics: CR3BP_Dynamics) -> DROTransferSearch:
    s = DROTransferSearch(dynamics)
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


def _make_orbit_at(point: np.ndarray, n: int = 8) -> Orbit:
    """构造一条"几乎是单点"的目标轨道，便于精确控制距离序列。

    所有采样点都在 ``point`` 附近 ``1e-12`` 抖动内，距离计算等价于
    "每步轨迹点到 point 的欧氏距离"。
    """
    rng = np.random.default_rng(0)
    states = np.zeros((n, 6))
    states[:, :3] = point + rng.standard_normal((n, 3)) * 1e-12
    times = np.linspace(0, 1, n)
    return Orbit(states, times)


def _trajectory_with_distances(d_sequence: np.ndarray, target: np.ndarray) -> np.ndarray:
    """根据期望的距离序列 d_sequence 构造一条 6-state 轨迹，
    使其第 i 步到 ``target`` 的距离恰为 ``d_sequence[i]``。
    """
    n = len(d_sequence)
    traj = np.zeros((n, 6))
    # 在 x 方向偏移 d_sequence[i] 即可，y/z 同 target
    traj[:, 0] = float(target[0]) + d_sequence
    traj[:, 1] = float(target[1])
    traj[:, 2] = float(target[2])
    return traj


# =============================================================================
# _compute_distance_series
# =============================================================================


class TestComputeDistanceSeries:
    def test_returns_full_per_step_series_with_correct_shapes(self, searcher):
        target = np.array([1.0, 0.0, 0.0])
        orbit = _make_orbit_at(target, n=5)
        d_expected = np.array([0.3, 0.2, 0.1, 0.05, 0.4])
        traj = _trajectory_with_distances(d_expected, target)

        d_per_step, orbit_idx_per_step = searcher._compute_distance_series(traj, orbit)

        assert d_per_step.shape == (5,)
        assert orbit_idx_per_step.shape == (5,)
        np.testing.assert_allclose(d_per_step, d_expected, atol=1e-9)
        # 所有 orbit 采样点几乎重合，orbit_idx 取哪个都可以，但必须落在 [0, n_orbit)
        assert (orbit_idx_per_step >= 0).all()
        assert (orbit_idx_per_step < 5).all()


class TestComputeMinDistanceBackwardCompat:
    def test_returns_three_tuple_with_argmin_of_series(self, searcher):
        target = np.array([1.0, 0.0, 0.0])
        orbit = _make_orbit_at(target, n=5)
        d_expected = np.array([0.3, 0.2, 0.1, 0.05, 0.4])
        traj = _trajectory_with_distances(d_expected, target)

        min_dist, step_idx, orbit_idx = searcher._compute_min_distance(traj, orbit)

        assert step_idx == 3  # argmin of [0.3, 0.2, 0.1, 0.05, 0.4]
        assert min_dist == pytest.approx(0.05, abs=1e-9)
        assert isinstance(step_idx, int)
        assert isinstance(orbit_idx, int)


# =============================================================================
# search() 的 4 个新字段（通过 mock _forward_integrate 注入合成轨迹）
# =============================================================================


class TestSearchFirstFeasibilityFields:
    """通过 mock _forward_integrate 注入已知距离序列的轨迹，
    断言 search_single_departure_point 写入的 4 个字段语义正确。"""

    def _run_search_with_distances(
        self,
        searcher: DROTransferSearch,
        d_sequence: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict:
        target = np.array([1.0, 0.0, 0.0])
        orbit = _make_orbit_at(target, n=8)
        traj = _trajectory_with_distances(d_sequence, target)
        n = len(d_sequence)
        traj_times = np.linspace(0.0, 1.0, n)

        # mock 积分器：返回我们构造好的轨迹
        def _fake_integrate(self, initial_state, t_end, dt):
            return traj, traj_times

        # mock 碰撞 / local minimum（避免依赖真实动力学）
        def _no_collision(self, traj_states):
            return False, None, -1

        def _no_local_min(self, traj_states, arrival_orbit):
            return False, float("inf"), -1

        monkeypatch.setattr(DROTransferSearch, "_forward_integrate", _fake_integrate, raising=True)
        monkeypatch.setattr(DROTransferSearch, "_check_collision", _no_collision, raising=True)
        monkeypatch.setattr(DROTransferSearch, "_detect_local_minimum", _no_local_min, raising=True)

        departure_state = np.array([1.5, 0.0, 0.0, 0.0, 0.5, 0.0])
        results = searcher._search_single_departure(
            departure_state=departure_state,
            departure_time=0.0,
            arrival_orbit=orbit,
            verbose=False,
        )
        assert len(results) == 1
        return results[0]

    def test_records_first_intersection_when_crossed(self, searcher, monkeypatch):
        # threshold=0.05；第一次 d<=0.05 的索引应为 2
        d = np.array([0.20, 0.10, 0.04, 0.03, 0.01, 0.5])
        r = self._run_search_with_distances(searcher, d, monkeypatch)

        assert r["first_intersection_idx"] == 2
        # traj_times = linspace(0,1,6) -> index 2 -> 2/5
        assert r["first_intersection_time"] == pytest.approx(2.0 / 5.0)
        # min_distance 选项 E1: 与 d_per_step.min() 自洽
        assert r["min_distance"] == pytest.approx(0.01)
        assert r["min_distance_idx"] == 4

    def test_records_first_min_distance_when_only_close(self, searcher, monkeypatch):
        # 全程不进入 intersection_threshold=0.05，但进入 min_distance_threshold=0.1
        d = np.array([0.30, 0.20, 0.15, 0.08, 0.09, 0.5])
        r = self._run_search_with_distances(searcher, d, monkeypatch)

        assert r["first_intersection_idx"] is None
        assert r["first_intersection_time"] is None
        assert r["first_min_distance_idx"] == 3  # 第一次 d<=0.1
        assert r["first_min_distance_time"] == pytest.approx(3.0 / 5.0)
        assert r["intersection_found"] is False

    def test_records_none_when_never_feasible(self, searcher, monkeypatch):
        # 全程 d > min_distance_threshold(0.1) > intersection_threshold(0.05)
        d = np.array([0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
        r = self._run_search_with_distances(searcher, d, monkeypatch)

        assert r["first_intersection_idx"] is None
        assert r["first_intersection_time"] is None
        assert r["first_min_distance_idx"] is None
        assert r["first_min_distance_time"] is None
        # 不应判定为 success
        assert r["status"] != "success"

    def test_fields_self_consistent_with_min_distance(self, searcher, monkeypatch):
        """E1 自洽性：first_intersection_idx 存在 ⇒ d_per_step[idx]<=ith
        ⇒ min_distance<=ith ⇒ intersection_found=True。"""
        d = np.array([0.20, 0.04, 0.03, 0.02, 0.5])
        r = self._run_search_with_distances(searcher, d, monkeypatch)

        assert r["first_intersection_idx"] is not None
        assert r["min_distance"] <= searcher.intersection_threshold
        assert r["intersection_found"] is True
