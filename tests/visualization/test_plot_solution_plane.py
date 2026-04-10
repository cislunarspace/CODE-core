"""
需求: Solution Plane 可视化（transfer_time vs total_delta_v 散点图）

复现论文 Fig.6：对搜索/优化结果绘制 solution plane（转移时间 T vs 总 Δv 散点图）。
支持按 transfer_type（DIRECT/LGA/EXTERNAL）着色，对应论文中不同的转移路径。

验收标准:
  1. TransferPlotter 新增 plot_solution_plane() 方法
  2. 输入: results（NLPOptimizationResult 列表或等价 dict 列表）,
         color_by（可选，按 transfer_type 着色）,
         ax（可选，复用已有 Axes）,
         show_colorbar（可选）
  3. 输出: matplotlib Axes 对象
  4. x 轴为 transfer_time, y 轴为 total_delta_v (delta_v1 + delta_v2)
  5. 无有效数据时不报错，显示 "no data" 提示文字
  6. 返回的 ax 可被调用方进一步定制（标题、保存等）

参考论文: Cui et al. (2025), Fig. 6a-d
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from e2m2e.core import CR3BP_System
from e2m2e.visualization.transfer import TransferPlotter
from e2m2e.transfer import NLPOptimizationResult, TransferType


@pytest.fixture
def system():
    s = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
    s.compute_libration_points()
    return s


def _make_result(
    transfer_time: float,
    delta_v1: float,
    delta_v2: float,
    transfer_type: TransferType = TransferType.DIRECT,
    success: bool = True,
) -> NLPOptimizationResult:
    return NLPOptimizationResult(
        alpha=1.0,
        transfer_time=transfer_time,
        t_ins=3.0,
        objective_value=delta_v1 + delta_v2,
        delta_v1=delta_v1,
        delta_v2=delta_v2,
        success=success,
        message="",
        transfer_type=transfer_type,
    )


def _make_results(n: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    types = [TransferType.DIRECT, TransferType.LGA, TransferType.EXTERNAL]
    results = []
    for i in range(n):
        tt = rng.uniform(2.0, 30.0)
        dv = rng.uniform(0.01, 0.5)
        dv1 = dv * rng.uniform(0.3, 0.7)
        ttype = types[i % 3]
        results.append(_make_result(tt, dv1, dv - dv1, transfer_type=ttype))
    return results


class TestPlotSolutionPlaneMethod:
    def test_method_exists(self, system):
        viz = TransferPlotter(system)
        assert hasattr(viz, "plot_solution_plane")
        assert callable(viz.plot_solution_plane)


class TestPlotSolutionPlaneBasic:
    def test_returns_axes(self, system):
        viz = TransferPlotter(system)
        results = _make_results(10)
        ax = viz.plot_solution_plane(results)
        assert ax is not None
        plt.close("all")

    def test_axes_has_scatter(self, system):
        viz = TransferPlotter(system)
        results = _make_results(10)
        ax = viz.plot_solution_plane(results)
        collections = ax.collections
        assert len(collections) >= 1, "应有至少一个 scatter collection"
        plt.close("all")

    def test_x_axis_label(self, system):
        viz = TransferPlotter(system)
        results = _make_results(10)
        ax = viz.plot_solution_plane(results)
        xlabel = ax.get_xlabel().lower()
        assert "transfer" in xlabel or "time" in xlabel or "t" in xlabel
        plt.close("all")

    def test_y_axis_label(self, system):
        viz = TransferPlotter(system)
        results = _make_results(10)
        ax = viz.plot_solution_plane(results)
        ylabel = ax.get_ylabel().lower()
        assert "delta" in ylabel or "dv" in ylabel or "Δ" in ylabel
        plt.close("all")


class TestPlotSolutionPlaneEmptyData:
    def test_empty_list_no_error(self, system):
        viz = TransferPlotter(system)
        ax = viz.plot_solution_plane([])
        assert ax is not None
        plt.close("all")

    def test_all_failed_no_error(self, system):
        viz = TransferPlotter(system)
        results = [_make_result(5.0, 0.1, 0.1, success=False)]
        ax = viz.plot_solution_plane(results)
        assert ax is not None
        plt.close("all")


class TestPlotSolutionPlaneColoredByType:
    def test_color_by_transfer_type(self, system):
        viz = TransferPlotter(system)
        results = _make_results(30)
        ax = viz.plot_solution_plane(results, color_by="transfer_type")
        assert ax is not None
        legend = ax.get_legend()
        assert legend is not None, "应显示图例"
        plt.close("all")

    def test_color_by_with_single_type(self, system):
        viz = TransferPlotter(system)
        results = [_make_result(5.0, 0.1, 0.1, transfer_type=TransferType.DIRECT)] * 5
        ax = viz.plot_solution_plane(results, color_by="transfer_type")
        assert ax is not None
        plt.close("all")


class TestPlotSolutionPlaneExternalAxes:
    def test_accepts_external_ax(self, system):
        viz = TransferPlotter(system)
        results = _make_results(10)
        fig, ax = plt.subplots()
        returned_ax = viz.plot_solution_plane(results, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_multiple_plots_on_same_ax(self, system):
        viz = TransferPlotter(system)
        results1 = _make_results(10, seed=1)
        results2 = _make_results(10, seed=2)
        fig, ax = plt.subplots()
        viz.plot_solution_plane(results1, ax=ax)
        viz.plot_solution_plane(results2, ax=ax)
        assert len(ax.collections) >= 2
        plt.close("all")


class TestPlotSolutionPlaneDictInput:
    def test_accepts_dict_list(self, system):
        viz = TransferPlotter(system)
        results = [
            {"transfer_time": 5.0, "delta_v1": 0.1, "delta_v2": 0.05,
             "objective_value": 0.15, "transfer_type": "direct", "success": True},
            {"transfer_time": 15.0, "delta_v1": 0.2, "delta_v2": 0.1,
             "objective_value": 0.3, "transfer_type": "lga", "success": True},
        ]
        ax = viz.plot_solution_plane(results)
        assert ax is not None
        plt.close("all")
