"""porkchop 持久化（SQLite 存档）与 Pareto 前沿测试。"""

import numpy as np
import pytest

from e2m2e.transfer.porkchop import ParetoFront, PorkchopData, pareto_front, porkchop
from e2m2e.transfer.terminal import TerminalCondition

MU_EARTH = 398600.4418  # km³/s²


class CircularOrbitTerminal(TerminalCondition):
    """解析圆轨道终端：半径 r、初始相位 phase0（rad），二体开普勒运动。"""

    def __init__(self, radius: float, phase0: float = 0.0, mu: float = MU_EARTH):
        self.radius = float(radius)
        self.phase0 = float(phase0)
        self.mu = float(mu)
        self.n = np.sqrt(self.mu / self.radius**3)  # 平均运动
        self.v = np.sqrt(self.mu / self.radius)

    def _state_at(self, t: float) -> np.ndarray:
        th = self.phase0 + self.n * t
        return np.array(
            [
                self.radius * np.cos(th),
                self.radius * np.sin(th),
                0.0,
                -self.v * np.sin(th),
                self.v * np.cos(th),
                0.0,
            ]
        )

    def get_initial_state(self) -> np.ndarray:
        return self._state_at(0.0)

    def get_arrival_state(self, t_ins: float, dynamics: object) -> tuple[np.ndarray, np.ndarray]:
        state = self._state_at(float(t_ins))
        return state[:3], state[3:6]


@pytest.fixture
def leo_geo():
    """LEO (7000 km) -> GEO (42164 km) 共面圆轨道场景。"""
    dep = CircularOrbitTerminal(7000.0)
    arr = CircularOrbitTerminal(42164.0)
    return dep, arr


def _small_grid(leo_geo) -> PorkchopData:
    """5×7 LEO→GEO 网格（含 NaN 组合）。"""
    dep, arr = leo_geo
    t_dep = np.linspace(0.0, 3600.0, 5)
    tof = np.linspace(900.0, 3600.0, 7)
    return porkchop(dep, arr, t_dep, tof, mu=MU_EARTH, dynamics=None)


class TestSqlitePersist:
    """SQLite 存档：往返等价、多 scan 累积、元数据。"""

    def test_roundtrip_numeric_equal(self, leo_geo, tmp_path):
        """写入再读回，数值完全等价（含 NaN→NULL→NaN）。"""
        data = _small_grid(leo_geo)
        db = tmp_path / "porkchop.db"
        scan_id = data.to_sqlite(db, orbit_pair="LEO->GEO", direction="short")
        loaded = PorkchopData.from_sqlite(db, scan_id)

        assert loaded.t_dep.shape == data.t_dep.shape
        assert loaded.tof.shape == data.tof.shape
        assert loaded.total.shape == data.total.shape
        np.testing.assert_allclose(loaded.t_dep, data.t_dep)
        np.testing.assert_allclose(loaded.tof, data.tof)
        # NaN 位置一致
        np.testing.assert_array_equal(np.isnan(loaded.total), np.isnan(data.total))
        np.testing.assert_array_equal(np.isnan(loaded.dv1), np.isnan(data.dv1))
        # 有效点数值一致
        valid = ~np.isnan(data.total)
        np.testing.assert_allclose(loaded.total[valid], data.total[valid], rtol=1e-12)
        np.testing.assert_allclose(loaded.dv1[valid], data.dv1[valid], rtol=1e-12)
        np.testing.assert_allclose(loaded.dv2[valid], data.dv2[valid], rtol=1e-12)

    def test_multi_scan_accumulate(self, leo_geo, tmp_path):
        """同一文件多次写入，scan_id 自增累积。"""
        data = _small_grid(leo_geo)
        db = tmp_path / "porkchop.db"
        sid1 = data.to_sqlite(db, orbit_pair="LEO->GEO", note="first")
        sid2 = data.to_sqlite(db, orbit_pair="LEO->GEO", note="second")
        assert sid1 != sid2

        loaded1 = PorkchopData.from_sqlite(db, sid1)
        loaded2 = PorkchopData.from_sqlite(db, sid2)
        np.testing.assert_allclose(loaded1.total, loaded2.total)

    def test_missing_scan_id_raises(self, leo_geo, tmp_path):
        """scan_id 不存在时抛 ValueError。"""
        data = _small_grid(leo_geo)
        db = tmp_path / "porkchop.db"
        data.to_sqlite(db, orbit_pair="LEO->GEO")
        with pytest.raises(ValueError, match="不存在"):
            PorkchopData.from_sqlite(db, 999)

    def test_auto_mkdir(self, leo_geo, tmp_path):
        """父目录不存在时自动创建。"""
        data = _small_grid(leo_geo)
        db = tmp_path / "deep" / "nested" / "porkchop.db"
        scan_id = data.to_sqlite(db, orbit_pair="LEO->GEO")
        assert db.exists()
        loaded = PorkchopData.from_sqlite(db, scan_id)
        assert loaded.total.shape == data.total.shape


class TestParetoFront:
    """Pareto 前沿：已知支配关系、字段一致性、无有效点报错。"""

    def test_known_dominated_points(self):
        """构造 4 点网格：2 点在前沿、2 点被支配。"""
        # 目标 (total, tof)，越小越优。网格 2×2：
        #   (t_dep=0, tof=1, total=1.0)  ← 前沿（tof 最小）
        #   (t_dep=0, tof=2, total=0.5)  ← 前沿（total 最小）
        #   (t_dep=1, tof=1, total=2.0)  ← 被 (tof=1, total=1.0) 支配
        #   (t_dep=1, tof=2, total=1.5)  ← 被 (tof=2, total=0.5) 支配
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        # dv1/dv2 拆半，total 是关键
        dv1 = np.array([[0.5, 0.25], [1.0, 0.75]])
        dv2 = np.array([[0.5, 0.25], [1.0, 0.75]])
        total = dv1 + dv2  # [[1.0, 0.5], [2.0, 1.5]]
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=dv1, dv2=dv2, total=total)

        front = pareto_front(data)
        assert isinstance(front, ParetoFront)
        assert front.total.shape[0] == 2
        # 前沿点：total=1.0 (tof=1) 和 total=0.5 (tof=2)
        np.testing.assert_array_equal(np.sort(front.total), [0.5, 1.0])
        np.testing.assert_array_equal(front.rank, [0, 0])

    def test_front_fields_consistent(self, leo_geo):
        """真实网格：前沿点数 < 总有效点数，且字段形状一致。"""
        data = _small_grid(leo_geo)
        front = pareto_front(data)
        n_valid = int(np.sum(~np.isnan(data.total)))
        assert front.total.shape[0] < n_valid
        assert front.t_dep.shape == front.total.shape
        assert front.tof.shape == front.total.shape
        assert front.dv1.shape == front.total.shape
        assert front.dv2.shape == front.total.shape

    def test_front_total_decreasing_with_tof(self, leo_geo):
        """LEO→GEO 经典形态：前沿上 total 随 tof 增加而下降。"""
        data = _small_grid(leo_geo)
        front = pareto_front(data)
        order = np.argsort(front.tof)
        sorted_total = front.total[order]
        # 允许相邻相等（网格分辨率），但整体趋势下降
        assert sorted_total[-1] < sorted_total[0]

    def test_all_nan_raises(self):
        """全 NaN 网格报错。"""
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        nan_grid = np.full((2, 2), np.nan)
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=nan_grid, dv2=nan_grid, total=nan_grid)
        with pytest.raises(ValueError, match="无有效点"):
            pareto_front(data)

    def test_bad_objective_raises(self, leo_geo):
        """不存在的目标字段报错。"""
        data = _small_grid(leo_geo)
        with pytest.raises(ValueError, match="不存在"):
            pareto_front(data, objectives=("nonexistent", "tof"))

    def test_custom_objectives(self):
        """自定义目标字段 (dv1, tof)：dv1 与 tof 此消彼长，前沿 2 点。"""
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        # dv1 随 tof 递减：tof=1 时 dv1=1.0，tof=2 时 dv1=0.5 → 两点互非支配
        dv1 = np.array([[1.0, 0.5], [1.2, 0.7]])
        dv2 = np.array([[0.5, 0.25], [0.6, 0.35]])
        total = dv1 + dv2
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=dv1, dv2=dv2, total=total)
        front = pareto_front(data, objectives=("dv1", "tof"))
        # (tof=1, dv1=1.0) tof 最小；(tof=2, dv1=0.5) dv1 最小 → 都在前沿
        assert front.total.shape[0] == 2
