"""MissionAssessment 与 SolutionDatabase 测试。"""

import numpy as np
import pytest

from e2m2e.transfer.mission_assessment import MissionAssessment
from e2m2e.transfer.porkchop import PorkchopData
from e2m2e.transfer.solution_database import SolutionDatabase


class TestMissionAssessment:
    """多指标加权综合评估。"""

    def test_evaluate_basic(self):
        """基本加权评分。"""
        ma = MissionAssessment(metric_names=["dv", "tof"])
        solutions = np.array([[3.0, 10.0], [4.0, 5.0], [5.0, 8.0]])
        scores = ma.evaluate(solutions, weights={"dv": 0.5, "tof": 0.5})
        assert scores.shape == (3,)
        # 归一化权重：0.5/1.0 = 0.5
        assert scores[0] == pytest.approx(0.5 * 3.0 + 0.5 * 10.0)
        assert scores[1] == pytest.approx(0.5 * 4.0 + 0.5 * 5.0)

    def test_evaluate_auto_names(self):
        """自动指标名 obj_0, obj_1。"""
        ma = MissionAssessment()
        solutions = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores = ma.evaluate(solutions, weights={"obj_0": 1.0, "obj_1": 0.0})
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(3.0)

    def test_evaluate_unknown_metric_raises(self):
        """未知指标名报错。"""
        ma = MissionAssessment(metric_names=["dv"])
        with pytest.raises(ValueError, match="未知指标名"):
            ma.evaluate(np.array([[1.0]]), weights={"bad": 1.0})

    def test_evaluate_empty_raises(self):
        """空解集报错。"""
        ma = MissionAssessment()
        with pytest.raises(ValueError, match="不能为空"):
            ma.evaluate(np.empty((0, 2)), weights={"obj_0": 1.0})

    def test_evaluate_zero_weights_raises(self):
        """全零权重报错。"""
        ma = MissionAssessment()
        with pytest.raises(ValueError, match="全为 0"):
            ma.evaluate(np.array([[1.0, 2.0]]), weights={"obj_0": 0.0, "obj_1": 0.0})

    def test_rank_and_best(self):
        """排序与最优解。"""
        ma = MissionAssessment(metric_names=["dv", "tof"])
        solutions = np.array([[3.0, 10.0], [4.0, 5.0], [5.0, 8.0]])
        order = ma.rank(solutions, weights={"dv": 0.5, "tof": 0.5})
        assert order[0] == 1  # 解 1 总分最低
        idx, score = ma.best(solutions, weights={"dv": 0.5, "tof": 0.5})
        assert idx == 1
        assert score == pytest.approx(0.5 * 4.0 + 0.5 * 5.0)

    def test_from_pareto_front(self):
        """从 ParetoFront 推断指标名。"""
        ma = MissionAssessment.from_pareto_front(
            type("Front", (), {"total": np.array([1.0]), "tof": np.array([2.0])})()
        )
        assert ma.metric_names == ["dv", "tof"]


class TestSolutionDatabase:
    """解数据库多 scan 聚合查询。"""

    def test_add_and_get_scan(self, tmp_path):
        """写入再读回。"""
        db = SolutionDatabase(tmp_path / "test.db")
        t_dep = np.array([0.0, 1.0])
        tof = np.array([10.0, 20.0])
        total = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=total * 0.5, dv2=total * 0.5, total=total)
        scan_id = db.add_scan(data, orbit_pair="LEO->GEO")
        loaded = db.get_scan(scan_id)
        np.testing.assert_allclose(loaded.total, data.total)

    def test_query(self, tmp_path):
        """插值查询。"""
        db = SolutionDatabase(tmp_path / "test.db")
        t_dep = np.array([0.0, 1.0, 2.0])
        tof = np.array([10.0, 20.0, 30.0])
        total = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=total * 0.5, dv2=total * 0.5, total=total)
        scan_id = db.add_scan(data, orbit_pair="LEO->GEO")
        assert db.query(scan_id, 1.0, 20.0) == pytest.approx(5.0)

    def test_pareto_front(self, tmp_path):
        """前沿提取。"""
        db = SolutionDatabase(tmp_path / "test.db")
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        total = np.array([[1.0, 0.5], [2.0, 1.5]])
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=total * 0.5, dv2=total * 0.5, total=total)
        scan_id = db.add_scan(data, orbit_pair="LEO->GEO")
        front = db.pareto_front(scan_id)
        assert front.total.shape[0] == 2

    def test_list_scans(self, tmp_path):
        """列出扫描元数据。"""
        db = SolutionDatabase(tmp_path / "test.db")
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        total = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=total * 0.5, dv2=total * 0.5, total=total)
        db.add_scan(data, orbit_pair="LEO->GEO", note="first")
        db.add_scan(data, orbit_pair="GEO->Mars", note="second")
        scans = db.list_scans()
        assert len(scans) == 2
        assert scans[0]["orbit_pair"] == "LEO->GEO"
        assert scans[1]["note"] == "second"

    def test_filter(self, tmp_path):
        """筛选钩子。"""
        db = SolutionDatabase(tmp_path / "test.db")
        t_dep = np.array([0.0, 1.0])
        tof = np.array([1.0, 2.0])
        total = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = PorkchopData(t_dep=t_dep, tof=tof, dv1=total * 0.5, dv2=total * 0.5, total=total)
        scan_id = db.add_scan(data, orbit_pair="LEO->GEO")
        # 筛选 total < 3.0
        mask = db.filter(scan_id, lambda td, tf, dv: dv < 3.0)
        assert mask.shape == (2, 2)
        assert mask[0, 0] and mask[0, 1]  # total=1, 2
        assert not mask[1, 0] and not mask[1, 1]  # total=3, 4
