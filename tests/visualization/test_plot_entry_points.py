"""OrbitVisualizer / FamilyPlotter / TransferPlotter .plot() 入口点测试。

验证三个可视化器的 plot() 公共入口方法行为正确。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from e2m2e.core import CR3BP_System, Orbit, OrbitFamily
from e2m2e.visualization.base import OrbitVisualizer
from e2m2e.visualization.family import FamilyPlotter
from e2m2e.visualization.transfer import TransferPlotter


@pytest.fixture
def system():
    s = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
    s.compute_libration_points()
    return s


def _make_orbit(system):
    t = np.linspace(0, 2 * np.pi, 50)
    x = 0.9 + 0.1 * np.cos(t)
    y = 0.1 * np.sin(t)
    z = 0.02 * np.sin(2 * t)
    vx = -0.1 * np.sin(t)
    vy = 0.1 * np.cos(t)
    vz = 0.04 * np.cos(2 * t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.system = system
    return orbit


class TestOrbitVisualizerPlotEntryPoint:
    """OrbitVisualizer.plot() 委托到 plot_3d_orbit"""

    def test_plot_returns_3d_axes(self, system):
        viz = OrbitVisualizer(system)
        orbit = _make_orbit(system)
        ax = viz.plot(orbit)
        assert ax is not None
        assert ax.name == "3d"
        plt.close("all")


class TestFamilyPlotterPlotEntryPoint:
    """FamilyPlotter.plot() 委托到 plot_family_2d"""

    def test_plot_returns_fig_ax(self, system):
        viz = FamilyPlotter(system)
        family = OrbitFamily(family_type="test")
        family.add_orbit(_make_orbit(system))
        result = viz.plot(family, jacobi_values=[3.0])
        assert result is not None
        fig, ax = result
        assert fig is not None
        assert ax is not None
        plt.close("all")


class TestTransferPlotterPlotEntryPoint:
    """TransferPlotter.plot() 委托到 plot_solution_plane"""

    def test_plot_returns_axes(self, system):
        viz = TransferPlotter(system)
        results = [
            {
                "transfer_time": 5.0,
                "delta_v1": 0.1,
                "delta_v2": 0.05,
                "objective_value": 0.15,
                "transfer_type": "direct",
                "success": True,
            }
        ]
        ax = viz.plot(results)
        assert ax is not None
        plt.close("all")


class TestVisualizationPackageImport:
    """visualization 包在无 plotting.py 的情况下可正常导入"""

    def test_public_api_imports(self):
        from e2m2e.visualization import (
            FamilyPlotter,
            OrbitVisualizer,
            PlotConfig,
            ProjectionPlane,
            TransferPlotter,
        )

        classes = [
            OrbitVisualizer,
            FamilyPlotter,
            TransferPlotter,
            PlotConfig,
            ProjectionPlane,
        ]
        assert all(cls is not None for cls in classes)

    def test_plotting_module_not_in_public_api(self):
        import e2m2e.visualization

        assert not hasattr(e2m2e.visualization, "configure_academic_fonts")
