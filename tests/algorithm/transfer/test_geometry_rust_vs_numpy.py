"""转移搜索几何核 Rust vs numpy 等价性单测。

对照权威基准 ``e2m2e.algorithm.transfer.search_geometry``（numpy 实现），
逐函数验证 Rust 后端（``e2m2e._integrators`` 暴露的 ``*_py``）数值与索引
约定一致。三档用例：合成已知答案 / 随机数组 / 边界（含 chunked 触发、
单点轨道、全同点、碰撞正例 earth/moon 命中）。

整数索引（argmin/argmax、首个命中步）要求**精确相等**——分叉即算法不一致；
浮点距离用 ``assert_allclose(rtol=1e-9, atol=1e-12)`` 兜底 ULP 差异。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# 扩展未构建时（doc build / 无 spice 构建合法）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.transfer.search_geometry import (
    check_collision as np_check_collision,
)
from e2m2e.algorithm.transfer.search_geometry import (
    compute_distance_series as np_compute_distance_series,
)
from e2m2e.algorithm.transfer.search_geometry import (
    compute_min_distance as np_compute_min_distance,
)
from e2m2e.algorithm.transfer.search_geometry import (
    detect_intersection as np_detect_intersection,
)
from e2m2e.algorithm.transfer.search_geometry import (
    detect_local_minimum as np_detect_local_minimum,
)
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import (
    check_collision_py,
    compute_distance_series_py,
    compute_min_distance_py,
    detect_intersection_py,
    detect_local_minimum_py,
)

pytestmark = pytest.mark.orchestration


# =============================================================================
# 辅助
# =============================================================================


def _orbit_of(states: np.ndarray) -> Orbit:
    """从 (n,6) 状态数组构造 Orbit（times 用 linspace，距离计算不依赖 times）。"""
    states = np.asarray(states, dtype=float)
    n = states.shape[0]
    return Orbit(states, np.linspace(0.0, 1.0, n))


def _flat(arr: np.ndarray) -> list[float]:
    """展平为 Python list[float]，传给 Rust pyfunction。"""
    return np.asarray(arr, dtype=float).reshape(-1).tolist()


# =============================================================================
# compute_distance_series
# =============================================================================


class TestComputeDistanceSeries:
    def test_known_answer_single_point_orbit(self):
        # 轨迹沿 x 轴移动，轨道单点固定原点 → 距离 = |x|。
        traj = np.zeros((5, 6))
        traj[:, 0] = [1.0, 2.0, 0.5, 3.0, 0.1]
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)
        expected = np.array([1.0, 2.0, 0.5, 3.0, 0.1])

        d_np, idx_np = np_compute_distance_series(traj, orbit)
        d_rs, idx_rs = compute_distance_series_py(_flat(traj), _flat(orbit_states))

        assert_allclose(d_np, expected, atol=1e-12)
        d_rs = np.asarray(d_rs)
        assert d_rs.shape == (5,)
        assert_allclose(d_rs, expected, atol=1e-12)
        # 单点轨道，orbit_idx 恒为 0。
        assert_array_equal(np.asarray(idx_rs), np.zeros(5, dtype=np.int64))

    def test_random_array_shapes_dtype_argmin(self):
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((20, 6))
        orbit_states = rng.standard_normal((15, 6))
        orbit = _orbit_of(orbit_states)

        d_np, idx_np = np_compute_distance_series(traj, orbit)
        d_rs, idx_rs = compute_distance_series_py(_flat(traj), _flat(orbit_states))

        d_rs = np.asarray(d_rs)
        idx_rs = np.asarray(idx_rs)
        assert d_rs.shape == (20,)
        assert idx_rs.shape == (20,)
        assert idx_rs.dtype == np.int64
        assert_allclose(d_rs, d_np, rtol=1e-9, atol=1e-12)
        # argmin 精确一致（重复最小值取首个约定）。
        assert_array_equal(idx_rs, idx_np.astype(np.int64))

    def test_chunked_path_matches_numpy(self):
        # n_traj*n_orbit = 10001*1000 = 1.0001e7 > 1e7，触发内部分块。
        rng = np.random.default_rng(7)
        n_traj, n_orbit = 10001, 1000
        traj = rng.standard_normal((n_traj, 6))
        orbit_states = rng.standard_normal((n_orbit, 6))
        orbit = _orbit_of(orbit_states)

        d_np, idx_np = np_compute_distance_series(traj, orbit)  # numpy 内部自动分块
        d_rs, idx_rs = compute_distance_series_py(_flat(traj), _flat(orbit_states))

        assert_allclose(np.asarray(d_rs), d_np, rtol=1e-9, atol=1e-12)
        assert_array_equal(np.asarray(idx_rs), idx_np.astype(np.int64))

    def test_argmin_first_on_ties(self):
        # 轨道两个重合点（原点），轨迹单步距两者相等 → argmin 取首个（0）。
        orbit_states = np.zeros((2, 6))
        traj = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        orbit = _orbit_of(orbit_states)

        _, idx_np = np_compute_distance_series(traj, orbit)
        _, idx_rs = compute_distance_series_py(_flat(traj), _flat(orbit_states))

        assert idx_np[0] == 0  # numpy argmin 取首个
        assert idx_rs[0] == 0  # Rust 同约定


# =============================================================================
# compute_min_distance
# =============================================================================


class TestComputeMinDistance:
    def test_known_answer_argmin_of_series(self):
        # d_per_step = [0.3, 0.2, 0.1, 0.05, 0.4]，全局最小在 idx=3。
        target = np.array([0.0, 0.0, 0.0])
        d_seq = np.array([0.3, 0.2, 0.1, 0.05, 0.4])
        traj = np.zeros((5, 6))
        traj[:, 0] = d_seq  # 沿 x 偏移即得期望距离
        orbit_states = np.tile(np.r_[target, np.zeros(3)], (3, 1))  # 3 点都在 target
        orbit = _orbit_of(orbit_states)

        md_np, si_np, oi_np = np_compute_min_distance(traj, orbit)
        md_rs, si_rs, oi_rs = compute_min_distance_py(_flat(traj), _flat(orbit_states))

        assert si_np == 3
        assert si_rs == 3
        assert md_rs == pytest.approx(0.05, abs=1e-12)
        assert md_rs == pytest.approx(float(md_np), abs=1e-12)
        assert oi_rs == oi_np  # orbit_idx 自洽

    def test_random_array(self):
        rng = np.random.default_rng(123)
        traj = rng.standard_normal((25, 6))
        orbit_states = rng.standard_normal((10, 6))
        orbit = _orbit_of(orbit_states)

        md_np, si_np, oi_np = np_compute_min_distance(traj, orbit)
        md_rs, si_rs, oi_rs = compute_min_distance_py(_flat(traj), _flat(orbit_states))

        assert isinstance(si_rs, int)
        assert isinstance(oi_rs, int)
        assert md_rs == pytest.approx(float(md_np), rel=1e-9, abs=1e-12)
        assert si_rs == si_np  # argmin 精确一致
        assert oi_rs == oi_np

    def test_argmin_first_on_tied_min(self):
        # d_per_step = [0.5, 0.5, 1.0]，两个并列最小 → argmin 取首个（idx=0）。
        traj = np.zeros((3, 6))
        traj[:, 0] = [0.5, 0.5, 1.0]
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)

        _, si_np, _ = np_compute_min_distance(traj, orbit)
        _, si_rs, _ = compute_min_distance_py(_flat(traj), _flat(orbit_states))

        assert si_np == 0
        assert si_rs == 0


# =============================================================================
# detect_intersection
# =============================================================================


class TestDetectIntersection:
    def test_hit_returns_global_closest_point(self):
        # 轨迹最近点距 orbit < threshold，返回该点完整 6 维状态。
        orbit_states = np.zeros((1, 6))
        traj = np.zeros((4, 6))
        traj[:, 0] = [2.0, 1.0, 0.01, 0.5]  # idx=2 最近（dist=0.01 < 0.1）
        traj[2, 3] = 7.0  # 标志速度，验证返回完整 6 维
        traj[2, 4] = 8.0
        traj[2, 5] = 9.0
        orbit = _orbit_of(orbit_states)

        found_np, pt_np, idx_np = np_detect_intersection(traj, orbit, 0.1)
        found_rs, pt_rs, idx_rs = detect_intersection_py(_flat(traj), _flat(orbit_states), 0.1)

        assert found_np and found_rs
        assert idx_rs == 2
        assert idx_rs == idx_np
        # 返回点含速度分量（完整 6 维），Rust 与 numpy 都是 6 维。
        assert pt_rs is not None
        assert_allclose(np.asarray(pt_rs), traj[2], atol=1e-12)
        assert_allclose(np.asarray(pt_rs), np.asarray(pt_np), atol=1e-12)

    def test_miss_when_min_dist_above_threshold(self):
        rng = np.random.default_rng(5)
        traj = 5.0 + rng.standard_normal((10, 6))  # 远离原点
        orbit_states = rng.standard_normal((4, 6)) * 0.1
        orbit = _orbit_of(orbit_states)

        found_np, pt_np, idx_np = np_detect_intersection(traj, orbit, 0.1)
        found_rs, pt_rs, idx_rs = detect_intersection_py(_flat(traj), _flat(orbit_states), 0.1)

        assert found_rs == found_np
        if not found_np:
            assert not found_rs
            assert pt_rs is None
            assert idx_rs == -1

    def test_threshold_strict_less_than(self):
        # min_dist 恰等于 threshold → 严格 < 不命中（与 numpy 一致）。
        traj = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])  # 距原点恰 0.1
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)

        found_np, _, _ = np_detect_intersection(traj, orbit, 0.1)
        found_rs, _, idx_rs = detect_intersection_py(_flat(traj), _flat(orbit_states), 0.1)

        assert found_np == found_rs  # 两者都为 False（严格 <）
        assert idx_rs == -1


# =============================================================================
# detect_local_minimum
# =============================================================================


class TestDetectLocalMinimum:
    def test_finds_strict_valley(self):
        # d_per_step = [2, 0.5, 1]，单谷在 idx=1。
        traj = np.zeros((3, 6))
        traj[:, 0] = [2.0, 0.5, 1.0]
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)

        found_np, dist_np, idx_np = np_detect_local_minimum(traj, orbit)
        found_rs, dist_rs, idx_rs = detect_local_minimum_py(_flat(traj), _flat(orbit_states))

        assert found_np and found_rs
        assert idx_rs == 1
        assert idx_rs == idx_np
        assert dist_rs == pytest.approx(0.5, abs=1e-12)
        assert dist_rs == pytest.approx(float(dist_np), abs=1e-12)

    def test_none_when_monotonic(self):
        # d_per_step 严格递减，无局部极小。
        traj = np.zeros((5, 6))
        traj[:, 0] = [5.0, 4.0, 3.0, 2.0, 1.0]
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)

        found_np, dist_np, idx_np = np_detect_local_minimum(traj, orbit)
        found_rs, dist_rs, idx_rs = detect_local_minimum_py(_flat(traj), _flat(orbit_states))

        assert not found_np
        assert not found_rs
        assert idx_rs == -1
        assert dist_rs == np.inf

    def test_multiple_minima_picks_smallest(self):
        # 两个谷：[3, 0.5, 2, 0.2, 3]，较深的谷在 idx=3（值 0.2）。
        traj = np.zeros((5, 6))
        traj[:, 0] = [3.0, 0.5, 2.0, 0.2, 3.0]
        orbit_states = np.zeros((1, 6))
        orbit = _orbit_of(orbit_states)

        found_np, dist_np, idx_np = np_detect_local_minimum(traj, orbit)
        found_rs, dist_rs, idx_rs = detect_local_minimum_py(_flat(traj), _flat(orbit_states))

        assert found_rs and found_np
        assert idx_rs == idx_np  # 都取值最小的那个极小
        assert idx_rs == 3
        assert dist_rs == pytest.approx(0.2, abs=1e-12)

    def test_random_array(self):
        rng = np.random.default_rng(99)
        traj = rng.standard_normal((30, 6))
        orbit_states = rng.standard_normal((8, 6))
        orbit = _orbit_of(orbit_states)

        found_np, dist_np, idx_np = np_detect_local_minimum(traj, orbit)
        found_rs, dist_rs, idx_rs = detect_local_minimum_py(_flat(traj), _flat(orbit_states))

        assert found_rs == found_np
        assert idx_rs == idx_np  # 整数索引精确一致
        if found_np:
            assert dist_rs == pytest.approx(float(dist_np), rel=1e-9, abs=1e-12)


# =============================================================================
# check_collision —— 现有测试只 patch False，正例零覆盖，必须补
# =============================================================================


class TestCheckCollision:
    # mu=0.1：earth 中心 [-0.1,0,0]，moon 中心 [0.9,0,0]。
    MU = 0.1
    EARTH_R = 1e-4
    MOON_R = 1e-4

    def test_earth_hit_returns_first_index(self):
        # 轨迹 idx=1 进入 earth 邻域（距 earth 1e-5 < 1e-4），其余点远离。
        traj = np.zeros((4, 6))
        traj[1, 0] = -self.MU + 1e-5  # earth 附近
        # 其余点远离两天体（默认 0 → 距 earth 0.1，距 moon 0.9，均不命中）

        hit_np, body_np, idx_np = np_check_collision(traj, self.MU, self.EARTH_R, self.MOON_R)
        hit_rs, body_rs, idx_rs = check_collision_py(
            _flat(traj), self.MU, self.EARTH_R, self.MOON_R
        )

        assert hit_np and hit_rs
        assert body_rs == "earth"
        assert body_rs == body_np
        assert idx_rs == 1
        assert idx_rs == idx_np

    def test_earth_priority_over_moon_when_both_hit(self):
        # 对抗用例：moon 在 idx=0 命中、earth 在 idx=1 命中——索引更早的是 moon，
        # 但 earth 优先级高于 moon（两侧实现都先完整扫 earth 命中即返回）。
        # 钉死"earth 优先于 moon（索引无关）"这一非显然不变量。
        traj = np.zeros((3, 6))
        traj[0, 0] = 1.0 - self.MU + 1e-5  # moon 附近（距 moon 1e-5 < 1e-4）
        traj[1, 0] = -self.MU + 1e-5  # earth 附近（距 earth 1e-5 < 1e-4）
        traj[2, 0] = 10.0  # 远离两天体

        hit_np, body_np, idx_np = np_check_collision(traj, self.MU, self.EARTH_R, self.MOON_R)
        hit_rs, body_rs, idx_rs = check_collision_py(
            _flat(traj), self.MU, self.EARTH_R, self.MOON_R
        )

        assert hit_np and hit_rs
        assert body_rs == "earth"
        assert body_np == "earth"
        assert idx_rs == 1
        assert idx_np == 1

    def test_moon_hit_when_no_earth(self):
        # 轨迹 idx=0 进入 moon 邻域（距 moon 1e-5），不命中 earth。
        traj = np.zeros((3, 6))
        traj[0, 0] = 1.0 - self.MU + 1e-5  # moon 附近
        traj[1, 0] = 10.0
        traj[2, 0] = -10.0

        hit_np, body_np, idx_np = np_check_collision(traj, self.MU, self.EARTH_R, self.MOON_R)
        hit_rs, body_rs, idx_rs = check_collision_py(
            _flat(traj), self.MU, self.EARTH_R, self.MOON_R
        )

        assert hit_np and hit_rs
        assert body_rs == "moon"
        assert body_rs == body_np
        assert idx_rs == 0
        assert idx_rs == idx_np

    def test_no_hit_when_far(self):
        # 全程远离两天体。
        traj = np.zeros((3, 6))
        traj[:, 0] = [10.0, -10.0, 5.0]
        traj[:, 1] = [1.0, -1.0, 2.0]

        hit_np, body_np, idx_np = np_check_collision(traj, self.MU, self.EARTH_R, self.MOON_R)
        hit_rs, body_rs, idx_rs = check_collision_py(
            _flat(traj), self.MU, self.EARTH_R, self.MOON_R
        )

        assert not hit_np
        assert not hit_rs
        assert body_rs is None
        assert idx_rs == -1

    def test_random_array_matches_numpy(self):
        rng = np.random.default_rng(256)
        traj = rng.standard_normal((40, 6))
        # 放宽阈值让部分点命中，覆盖命中分支。
        earth_r = 0.5
        moon_r = 0.4

        hit_np, body_np, idx_np = np_check_collision(traj, self.MU, earth_r, moon_r)
        hit_rs, body_rs, idx_rs = check_collision_py(_flat(traj), self.MU, earth_r, moon_r)

        assert hit_rs == hit_np
        assert body_rs == body_np
        assert idx_rs == idx_np
