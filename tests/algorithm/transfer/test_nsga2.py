"""NSGA-II 多目标优化器测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.transfer.nsga2 import (
    NSGA2Result,
    _constrained_non_dominated_sort,
    _environmental_selection,
    _polynomial_mutation,
    _rust_variation,
    _sbx_crossover,
    nsga2,
)
from e2m2e.integrators import (
    nsga2_environmental_selection_py,
    nsga2_sort_py,
    nsga2_tournament_selection_py,
)

pytestmark = pytest.mark.orchestration


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
    # ADR 0037 增补（2026-08-30）：最坏实测 ~9.6s，大头是 Windows spawn 导入成本，
    # 判定不可再压；30s 为最坏实测的 ≥2 倍余量。
    @pytest.mark.time_budget(30)
    def test_parallel_consistent_with_serial(self):
        """同种子下串行与并行结果一致（评估顺序不影响）。

        耗时大头是 Windows spawn 子进程的导入成本（ADR 0037 预算内尽量压
        到最小种群/代数，只保留触发并行路径的最小规模）。
        """
        # 注意：并行要求 fn 是模块级可导入函数
        r_serial = nsga2(schaffer, bounds=[(-5, 5)], pop_size=10, n_gen=5, seed=42, n_workers=1)
        r_parallel = nsga2(schaffer, bounds=[(-5, 5)], pop_size=10, n_gen=5, seed=42, n_workers=2)
        np.testing.assert_allclose(r_serial.f, r_parallel.f)


class TestNSGA2RustBackend:
    """Rust 演化算子与 Python 参照路径的等价性。"""

    def test_deterministic_operators_match_python(self):
        """约束排序、拥挤度与精英保留在固定输入下逐项一致。"""
        fit = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
                [0.5, 0.5],
                [1.0, 4.0],
                [5.0, 5.0],
            ]
        )
        viol = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4])

        py_rank, py_crowd = _constrained_non_dominated_sort(fit, viol)
        rust_rank, rust_crowd = nsga2_sort_py(fit.tolist(), viol.tolist())
        np.testing.assert_array_equal(rust_rank, py_rank)
        np.testing.assert_allclose(rust_crowd, py_crowd)

        expected = _environmental_selection(py_rank, py_crowd, 4)
        actual = nsga2_environmental_selection_py(py_rank.tolist(), py_crowd.tolist(), 4)
        np.testing.assert_array_equal(actual, expected)

        nan_viol = np.array([0.0, np.nan, 0.2])
        nan_fit = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        py_rank, py_crowd = _constrained_non_dominated_sort(nan_fit, nan_viol)
        rust_rank, rust_crowd = nsga2_sort_py(nan_fit.tolist(), nan_viol.tolist())
        np.testing.assert_array_equal(rust_rank, py_rank)
        np.testing.assert_allclose(rust_crowd, py_crowd)

        zero_fit = np.array([[-0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        zero_viol = np.zeros(3)
        py_rank, py_crowd = _constrained_non_dominated_sort(zero_fit, zero_viol)
        rust_rank, rust_crowd = nsga2_sort_py(zero_fit.tolist(), zero_viol.tolist())
        np.testing.assert_array_equal(rust_rank, py_rank)
        np.testing.assert_allclose(rust_crowd, py_crowd)

    def test_tournament_and_variation_match_python(self):
        """固定父代和种子下，Rust 锦标赛、SBX 与变异逐项对拍。"""
        rank = [0, 1, 0, 1]
        crowd = [0.1, 0.9, 0.3, 0.2]
        draws = [0, 1, 2, 3, 1, 2, 3, 0]
        assert nsga2_tournament_selection_py(rank, crowd, draws) == [0, 2, 2, 0]

        parents = np.array([[0.1, 0.8], [0.7, 0.2], [0.4, 0.5], [0.6, 0.3]])
        lo = np.zeros(2)
        hi = np.ones(2)
        python_rng = np.random.default_rng(7)
        rust_rng = np.random.default_rng(7)
        expected = _polynomial_mutation(
            _sbx_crossover(parents, lo, hi, 0.9, 20.0, python_rng),
            lo,
            hi,
            0.5,
            20.0,
            python_rng,
        )
        actual = _rust_variation(parents, lo, hi, 0.9, 20.0, 0.5, 20.0, rust_rng)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_full_evolution_matches_python_backend(self):
        """同一随机种子和参数下，Rust 与 Python 演化结果相同。"""
        kwargs = {
            "bounds": [(-5.0, 5.0)],
            "pop_size": 40,
            "n_gen": 15,
            "seed": 42,
            "n_workers": 1,
        }
        python = nsga2(schaffer, backend="python", **kwargs)
        rust = nsga2(schaffer, backend="rust", **kwargs)

        np.testing.assert_allclose(rust.x, python.x, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(rust.f, python.f, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(rust.rank, python.rank)
        np.testing.assert_allclose(rust.crowding, python.crowding)
        assert rust.n_eval == python.n_eval
        assert rust.history == python.history
