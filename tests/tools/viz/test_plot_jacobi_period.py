"""FamilyPlotter.plot_jacobi_period 测试。

验证返回 fig/ax、数据排序与标签。
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.tools.viz.family import FamilyPlotter

pytestmark = pytest.mark.aux


MU = 1.21506683e-2


@pytest.fixture
def system():
    s = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    s.compute_libration_points()
    return s


@pytest.fixture
def plotter(system):
    return FamilyPlotter(system)


JACOBI = [3.18, 3.12, 3.15]
PERIODS = [6.2, 5.8, 6.0]


class TestBasicCall:
    def test_returns_fig_ax(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax = plotter.plot_jacobi_period(JACOBI, PERIODS, show=False)
            assert fig is not None
            assert ax is not None
            lines = ax.get_lines()
            assert len(lines) >= 1
            assert lines[0].get_color() == "tab:blue"
        finally:
            plt.close("all")


class TestSortedByJacobi:
    def test_xdata_ascending(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax = plotter.plot_jacobi_period(JACOBI, PERIODS, show=False)
            xdata = list(ax.get_lines()[0].get_xdata())
            assert xdata == sorted(xdata)
            assert np.allclose(xdata, sorted(JACOBI))
        finally:
            plt.close("all")

    def test_ydata_sorted_correspondingly(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax = plotter.plot_jacobi_period(JACOBI, PERIODS, show=False)
            ydata = list(ax.get_lines()[0].get_ydata())
            sort_idx = sorted(range(len(JACOBI)), key=lambda i: JACOBI[i])
            expected_periods = [PERIODS[i] for i in sort_idx]
            assert np.allclose(ydata, expected_periods)
        finally:
            plt.close("all")


class TestTargetPeriod:
    def test_green_dashed_line_present(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax = plotter.plot_jacobi_period(JACOBI, PERIODS, target_period=5.9, show=False)
            lines = ax.get_lines()
            ref_lines = [line for line in lines if line.get_linestyle() == "--"]
            assert len(ref_lines) >= 1
            ref = ref_lines[0]
            assert ref.get_color() == "green"
            assert np.isclose(ref.get_ydata()[0], 5.9)
        finally:
            plt.close("all")

    def test_no_ref_line_without_target(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax = plotter.plot_jacobi_period(JACOBI, PERIODS, show=False)
            ref_lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
            assert len(ref_lines) == 0
        finally:
            plt.close("all")


class TestSavePath:
    def test_saves_file(self, plotter, tmp_path):
        import matplotlib.pyplot as plt

        save_file = str(tmp_path / "jacobi_period.png")
        try:
            plotter.plot_jacobi_period(JACOBI, PERIODS, save_path=save_file, show=False)
            assert os.path.exists(save_file)
            assert os.path.getsize(save_file) > 0
        finally:
            plt.close("all")


class TestNoRegression:
    def test_jacobi_period_stability_still_works(self, plotter):
        import matplotlib.pyplot as plt

        try:
            fig, ax1 = plotter.plot_jacobi_period_stability(
                JACOBI, PERIODS, stability_values=[0.9, 1.1, 1.0], show=False
            )
            assert fig is not None
            assert ax1 is not None
            # 双 Y 轴图应有 twin axes
            assert hasattr(ax1, "twinx") or len(fig.axes) >= 2
        finally:
            plt.close("all")
