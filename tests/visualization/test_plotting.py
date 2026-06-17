"""OrbitVisualizer 与 FamilyPlotter 类测试。

覆盖可视化器创建、3D 轨道族绘图与配置。
"""

import os
import tempfile

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # 使用非交互式后端

from e2m2e.core import CR3BP_System, Orbit, OrbitFamily
from e2m2e.visualization.base import OrbitVisualizer
from e2m2e.visualization.family import FamilyPlotter


class TestOrbitVisualizerCreation:
    """测试可视化器创建"""

    def test_visualizer_creation(self):
        """测试可视化器基本创建"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        system.compute_libration_points()

        viz = OrbitVisualizer(system)

        assert viz.system is system
        assert viz.mu == 0.01215

    def test_visualizer_default_settings(self):
        """测试可视化器默认设置"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")

        viz = OrbitVisualizer(system)

        assert viz.config.figsize_2d == (12, 10)
        assert viz.config.dpi == 100
        assert viz.orbit_linewidth == 1.5
        assert viz.orbit_alpha == 0.8


class TestPlot3DOrbitFamily:
    """测试3D轨道族绘图功能"""

    @pytest.fixture
    def sample_system(self):
        """创建测试用CR3BP系统"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        system.compute_libration_points()
        return system

    @pytest.fixture
    def sample_family(self):
        """创建测试用轨道族"""
        family = OrbitFamily(family_type="test")

        # 创建3条简单的测试轨道
        for _i in range(3):
            # 创建椭圆形状的测试状态
            t = np.linspace(0, 2 * np.pi, 50)
            x = 0.9 + 0.1 * np.cos(t)
            y = 0.1 * np.sin(t)
            z = 0.02 * np.sin(2 * t)
            vx = -0.1 * np.sin(t)
            vy = 0.1 * np.cos(t)
            vz = 0.04 * np.cos(2 * t)

            states = np.column_stack([x, y, z, vx, vy, vz])
            times = t

            orbit = Orbit(states, times)
            orbit.period = 2 * np.pi
            orbit.system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")

            family.add_orbit(orbit)

        return family

    def test_plot_3d_orbit_family_basic(self, sample_system, sample_family):
        """测试3D轨道族基本绘图"""
        viz = FamilyPlotter(sample_system)

        fig, ax = viz.plot_family_3d(
            sample_family,
            jacobi_values=[3.0, 3.1, 3.2],
            center=(0.99, 0.0, 0.0),
            radius=0.40,
            show_colorbar=True,
            show=False,
        )

        assert ax is not None
        assert ax.name == "3d"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_3d_orbit_family_with_save(self, sample_system, sample_family):
        """测试3D轨道族绘图并保存到文件"""
        viz = FamilyPlotter(sample_system)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # 绘图并保存
            fig, ax = viz.plot_family_3d(sample_family, jacobi_values=[3.0, 3.1, 3.2], show=False)
            fig.savefig(temp_path, dpi=100)

            # 验证文件已创建
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            import matplotlib.pyplot as plt

            plt.close("all")

    def test_plot_3d_orbit_family_empty(self, sample_system):
        """测试空轨道族绘图"""
        viz = FamilyPlotter(sample_system)
        empty_family = OrbitFamily(family_type="empty")

        # 空轨道族不应报错
        fig, ax = viz.plot_family_3d(empty_family, jacobi_values=[], show=False)
        # 空族可能返回 None 或空坐标轴
        assert ax is None or ax.name == "3d"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_3d_orbit_family_single_orbit(self, sample_system):
        """测试单轨道绘图"""
        viz = FamilyPlotter(sample_system)

        # 创建单条轨道
        t = np.linspace(0, 2 * np.pi, 50)
        x = 0.9 + 0.1 * np.cos(t)
        y = 0.1 * np.sin(t)
        z = 0.02 * np.sin(2 * t)
        vx = -0.1 * np.sin(t)
        vy = 0.1 * np.cos(t)
        vz = 0.04 * np.cos(2 * t)

        states = np.column_stack([x, y, z, vx, vy, vz])

        family = OrbitFamily(family_type="single")
        orbit = Orbit(states, t)
        orbit.period = 2 * np.pi
        orbit.system = sample_system
        family.add_orbit(orbit)

        fig, ax = viz.plot_family_3d(
            family,
            jacobi_values=[3.0],
            center=(0.99, 0.0, 0.0),
            radius=0.40,
            show=False,
        )

        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_3d_orbit_family_custom_view(self, sample_system, sample_family):
        """测试自定义视角参数"""
        viz = FamilyPlotter(sample_system)

        fig, ax = viz.plot_family_3d(
            sample_family,
            jacobi_values=[3.0, 3.1, 3.2],
            center=(1.0, 0.1, 0.0),
            radius=0.5,
            show_colorbar=False,
            show=False,
        )

        # 验证坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        assert xlim[1] - xlim[0] == pytest.approx(1.0, abs=0.01)
        assert ylim[1] - ylim[0] == pytest.approx(1.0, abs=0.01)
        assert zlim[1] - zlim[0] == pytest.approx(1.0, abs=0.01)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_3d_orbit_family_with_jacobi_none(self, sample_system, sample_family):
        """测试Jacobi为None时的处理"""
        viz = FamilyPlotter(sample_system)

        fig, ax = viz.plot_family_3d(
            sample_family,
            jacobi_values=None,
            center=(0.99, 0.0, 0.0),
            radius=0.40,
            show=False,
        )

        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlot3DOrbitFamilyWithBodies:
    """测试3D轨道族绘图（包含天体）"""

    @pytest.fixture
    def system_with_libration(self):
        """创建带拉格朗日点的系统"""
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        system.compute_libration_points()
        return system

    def test_plot_3d_with_primary_bodies(self, system_with_libration):
        """测试3D图中的主次天体"""
        viz = FamilyPlotter(system_with_libration)

        # 创建简单轨道
        t = np.linspace(0, 2 * np.pi, 50)
        x = 0.9 + 0.1 * np.cos(t)
        y = 0.1 * np.sin(t)
        z = 0.02 * np.sin(2 * t)
        vx = -0.1 * np.sin(t)
        vy = 0.1 * np.cos(t)
        vz = 0.04 * np.cos(2 * t)

        states = np.column_stack([x, y, z, vx, vy, vz])

        family = OrbitFamily(family_type="test")
        orbit = Orbit(states, t)
        orbit.period = 2 * np.pi
        orbit.system = system_with_libration
        family.add_orbit(orbit)

        fig, ax = viz.plot_family_3d(
            family,
            jacobi_values=[3.0],
            center=(0.99, 0.0, 0.0),
            radius=0.40,
            show=False,
        )

        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")
