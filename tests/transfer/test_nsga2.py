"""NSGA-II 多目标优化器测试。"""

import numpy as np
import pytest

from e2m2e.transfer.nsga2 import NSGA2Result, nsga2

# ----------------------------------------------------------------------
# 模块级目标函数（ProcessPoolExecutor 可 pickle）
# ----------------------------------------------------------------------


def zdt1(x: np.ndarray) -> tuple[np.ndarray, float]:
    """ZDT1 经典双目标测试问题。

    Pareto 前沿：f2 = 1 - sqrt(f1)，f1 ∈ [0, 1]。
    """
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (x.shape[0] - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2]), 0.0


def schaffer(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Schaffer N.1 单变量双目标。

    Pareto 前沿：f2 = (sqrt(f1) - 2)^2，f1 ∈ [0, 4]。
    """
    return np.array([x[0] ** 2, (x[0] - 2.0) ** 2]), 0.0


def constrained_sphere(x: np.ndarray) -> tuple[np.ndarray, float]:
    """带约束球面：min x² + y² 与 (x-1)² + (y-1)²，s.t. x + y >= 1。"""
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
    viol = max(0.0, 1.0 - x[0] - x[1])
    return np.array([f1, f2]), viol


def infeasible(x: np.ndarray) -> tuple[np.ndarray, float]:
    """全不可行问题：约束 x[0] >= 2，但 bounds 上限 1。"""
    return np.array([x[0] ** 2]), max(0.0, 2.0 - x[0])


class TestNSGA2:
    """NSGA-II 核心功能。"""

    def test_zdt1_converges(self):
        """ZDT1：100 代后前沿贴近理论曲线 f2 = 1 - sqrt(f1)。"""
        result = nsga2(
            zdt1,
            bounds=[(0.0, 1.0)] * 10,
            pop_size=100,
            n_gen=100,
            seed=42,
            n_workers=1,
        )
        assert isinstance(result, NSGA2Result)
        assert result.x.shape[0] == 100  # pop_size 精英全在前沿
        assert result.x.shape[1] == 10
        assert result.f.shape[1] == 2
        assert np.all(result.rank == 0)

        # 与理论前沿误差
        theoretical_f2 = 1.0 - np.sqrt(result.f[:, 0])
        errors = np.abs(result.f[:, 1] - theoretical_f2)
        assert errors.mean() < 0.01
        assert errors.max() < 0.05

    def test_schaffer_converges(self):
        """Schaffer N.1：单变量双目标，前沿覆盖 [0, 4]。"""
        result = nsga2(
            schaffer,
            bounds=[(-5.0, 5.0)],
            pop_size=100,
            n_gen=100,
            seed=42,
            n_workers=1,
        )
        # 前沿 x 应在 [0, 2]（Pareto 最优区间）
        assert np.all(result.x[:, 0] >= -0.1)
        assert np.all(result.x[:, 0] <= 2.1)
        # f1 覆盖 [0, 4]
        assert result.f[:, 0].min() < 0.1
        assert result.f[:, 0].max() > 3.9

    def test_constrained_feasible_dominates(self):
        """Deb 可行支配：约束问题前沿点全部可行。"""
        result = nsga2(
            constrained_sphere,
            bounds=[(0.0, 2.0), (0.0, 2.0)],
            pop_size=100,
            n_gen=50,
            seed=42,
            n_workers=1,
        )
        # 所有前沿点满足 x + y >= 1
        violations = np.maximum(0.0, 1.0 - result.x[:, 0] - result.x[:, 1])
        assert violations.max() == pytest.approx(0.0, abs=1e-6)

    def test_infeasible_all(self):
        """全不可行：viol 最小者 rank 0，其余被淘汰。"""
        result = nsga2(
            infeasible,
            bounds=[(0.0, 1.0)],
            pop_size=20,
            n_gen=10,
            seed=42,
            n_workers=1,
        )
        # 全不可行时 viol 最小者 rank 0（x=1 处 viol=1 最小）
        assert result.rank.min() == 0
        # 种群收敛到唯一最优解（viol 最小点），或至少 rank 0 是 viol 最小的
        assert np.all(result.x[:, 0] >= 0.9)  # x 接近 1（viol 最小）

    def test_seed_reproducibility(self):
        """同种子结果一致。"""
        r1 = nsga2(schaffer, bounds=[(-5, 5)], pop_size=50, n_gen=20, seed=42, n_workers=1)
        r2 = nsga2(schaffer, bounds=[(-5, 5)], pop_size=50, n_gen=20, seed=42, n_workers=1)
        np.testing.assert_array_equal(r1.x, r2.x)
        np.testing.assert_array_equal(r1.f, r2.f)

    def test_bounds_validation(self):
        """空 bounds / 下界 >= 上界报错。"""
        with pytest.raises(ValueError, match="bounds 不能为空"):
            nsga2(schaffer, bounds=[], pop_size=10, n_gen=5)
        with pytest.raises(ValueError, match="下界须严格小于上界"):
            nsga2(schaffer, bounds=[(5.0, 5.0)], pop_size=10, n_gen=5)

    def test_pop_size_validation(self):
        """pop_size < 4 报错。"""
        with pytest.raises(ValueError, match="pop_size"):
            nsga2(schaffer, bounds=[(-5, 5)], pop_size=3, n_gen=5)

    def test_non_picklable_fn_raises(self):
        """lambda/闭包不可 pickle 时报清晰错误。"""
        with pytest.raises(ValueError, match="不可 pickle"):
            nsga2(
                lambda x: (np.array([x[0] ** 2]), 0.0),
                bounds=[(-5, 5)],
                pop_size=10,
                n_gen=5,
                n_workers=2,
            )

    def test_n_eval_counting(self):
        """评估次数 = (1 + n_gen) * pop_size。"""
        result = nsga2(schaffer, bounds=[(-5, 5)], pop_size=20, n_gen=10, n_workers=1)
        assert result.n_eval == 20 + 10 * 20

    def test_history_recording(self):
        """history 记录每代种群与前沿规模。"""
        result = nsga2(schaffer, bounds=[(-5, 5)], pop_size=20, n_gen=10, n_workers=1)
        assert len(result.history) == 10
        for h in result.history:
            assert "gen" in h
            assert "front_size" in h
            assert h["pop_size"] == 20


class TestNSGA2Parallel:
    """并行评估一致性。"""

    @pytest.mark.skipif(not hasattr(pytest, "importorskip"), reason="dummy skip for parallel test")
    def test_parallel_consistent_with_serial(self):
        """同种子下串行与并行结果一致（评估顺序不影响）。"""
        # 注意：并行要求 fn 是模块级可导入函数
        r_serial = nsga2(schaffer, bounds=[(-5, 5)], pop_size=50, n_gen=20, seed=42, n_workers=1)
        r_parallel = nsga2(schaffer, bounds=[(-5, 5)], pop_size=50, n_gen=20, seed=42, n_workers=2)
        np.testing.assert_allclose(r_serial.f, r_parallel.f)
