"""Solution Plane 可视化测试（transfer_time vs total_delta_v 散点图）。

复现论文 Fig.6，支持按 transfer_type 着色。
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.transfer import TransferOptimizationResult
from e2m2e.mbse.data.enums import TransferType
from e2m2e.tools.viz.transfer import TransferPlotter


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
) -> TransferOptimizationResult:
    return TransferOptimizationResult(
        transfer_time=transfer_time,
        delta_v1=delta_v1,
        delta_v2=delta_v2,
        total_delta_v=delta_v1 + delta_v2,
        transfer_type=transfer_type,
        success=success,
        message="",
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
            {
                "transfer_time": 5.0,
                "delta_v1": 0.1,
                "delta_v2": 0.05,
                "objective_value": 0.15,
                "transfer_type": "direct",
                "success": True,
            },
            {
                "transfer_time": 15.0,
                "delta_v1": 0.2,
                "delta_v2": 0.1,
                "objective_value": 0.3,
                "transfer_type": "lga",
                "success": True,
            },
        ]
        ax = viz.plot_solution_plane(results)
        assert ax is not None
        plt.close("all")
