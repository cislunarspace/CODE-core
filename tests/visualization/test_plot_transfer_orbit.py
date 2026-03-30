"""
需求: 转移轨道 3D 可视化

复现论文 Fig.8-10：在旋转系下绘制 DRO 出发轨道 + 转移轨迹 + RO 到达轨道的 3D 示意图。
支持单条和多条转移轨迹叠加绘制，标注出发/到达点，绘制天体（地球、月球）和拉格朗日点。

验收标准:
  1. OrbitVisualizer 新增 plot_transfer_orbit() 方法
  2. 输入:
     - departure_orbit: Orbit (DRO)
     - arrival_orbit: Orbit (RO)
     - transfer_trajectory: np.ndarray (n, 6) 转移轨迹状态序列
     - departure_state: np.ndarray (6,) 出发点状态（标注用）
     - insertion_state: np.ndarray (6,) 到达点状态（标注用）
     - ax: 可选，复用已有 3D Axes
     - label: 可选，轨迹标签
     - color: 可选，轨迹颜色
  3. 输出: matplotlib Axes (projection='3d')
  4. 图中应包含:
     - DRO 轨道线（出发轨道）
     - RO 轨道线（到达轨道）
     - 转移轨迹线（突出显示）
     - 出发点标记
     - 到达点标记
     - 地球、月球天体标记
     - 拉格朗日点（可选）
  5. 支持在同一 ax 上叠加绘制多条转移轨迹
  6. 坐标轴标签使用无量纲单位

参考论文: Cui et al. (2025), Fig. 8a-d, Fig. 9a-d, Fig. 10a-d
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from e2m2e.core import Orbit, CR3BP_System
from e2m2e.visualization.plotting import OrbitVisualizer


@pytest.fixture
def system():
    s = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
    s.compute_libration_points()
    return s


def _make_dro(n: int = 100) -> Orbit:
    t = np.linspace(0, 2 * np.pi, n)
    x = 0.9 + 0.08 * np.cos(t)
    y = 0.08 * np.sin(t)
    z = np.zeros_like(t)
    vx = -0.08 * np.sin(t)
    vy = 0.08 * np.cos(t)
    vz = np.zeros_like(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.period = float(t[-1])
    return orbit


def _make_ro(n: int = 80) -> Orbit:
    t = np.linspace(0, 2 * np.pi, n)
    x = 1.2 + 0.15 * np.cos(t)
    y = 0.15 * np.sin(t)
    z = 0.02 * np.sin(t)
    vx = -0.15 * np.sin(t)
    vy = 0.15 * np.cos(t)
    vz = 0.02 * np.cos(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.period = float(t[-1])
    return orbit


def _make_transfer(n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)
    x = 0.98 + 0.25 * t
    y = 0.05 * np.sin(np.pi * t)
    z = 0.01 * np.sin(2 * np.pi * t)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    vz = np.zeros_like(t)
    return np.column_stack([x, y, z, vx, vy, vz])


@pytest.fixture
def dro():
    return _make_dro()


@pytest.fixture
def ro():
    return _make_ro()


@pytest.fixture
def transfer_trajectory():
    return _make_transfer()


class TestPlotTransferOrbitMethod:
    def test_method_exists(self, system):
        viz = OrbitVisualizer(system)
        assert hasattr(viz, "plot_transfer_orbit")
        assert callable(viz.plot_transfer_orbit)


class TestPlotTransferOrbitBasic:
    def test_returns_3d_axes(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        dep_state = dro.states[0]
        ins_state = ro.states[0]
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dep_state,
            insertion_state=ins_state,
        )
        assert ax is not None
        assert ax.name == "3d"
        plt.close("all")

    def test_contains_dro_line(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
        )
        lines = ax.get_lines()
        assert len(lines) >= 3, "至少应有 DRO、RO、transfer 三条线"
        plt.close("all")

    def test_contains_scatter_points(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
        )
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(scatter_collections) >= 1, "应有散点标记（出发点/到达点/天体）"
        plt.close("all")


class TestPlotTransferOrbitLabels:
    def test_axis_labels(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
        )
        assert "X" in ax.get_xlabel() or "x" in ax.get_xlabel()
        assert "Y" in ax.get_ylabel() or "y" in ax.get_ylabel()
        plt.close("all")


class TestPlotTransferOrbitMultiple:
    def test_overlay_two_transfers(self, system, dro, ro):
        viz = OrbitVisualizer(system)
        ax = None
        for i in range(2):
            traj = _make_transfer(60 + i * 10)
            ax = viz.plot_transfer_orbit(
                departure_orbit=dro,
                arrival_orbit=ro,
                transfer_trajectory=traj,
                departure_state=dro.states[i],
                insertion_state=ro.states[i],
                ax=ax,
            )
        lines = ax.get_lines()
        transfer_lines = [l for l in lines if l.get_linewidth() >= 1.5]
        assert len(transfer_lines) >= 2, "应有至少 2 条转移轨迹线"
        plt.close("all")


class TestPlotTransferOrbitExternalAxes:
    def test_accepts_external_ax(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned_ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
            ax=ax,
        )
        assert returned_ax is ax
        plt.close("all")


class TestPlotTransferOrbitCustomStyle:
    def test_custom_color(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
            color="magenta",
        )
        lines = ax.get_lines()
        transfer_line = [l for l in lines if l.get_linewidth() >= 1.5]
        assert any(l.get_color() == "magenta" for l in transfer_line)
        plt.close("all")

    def test_custom_label(self, system, dro, ro, transfer_trajectory):
        viz = OrbitVisualizer(system)
        ax = viz.plot_transfer_orbit(
            departure_orbit=dro,
            arrival_orbit=ro,
            transfer_trajectory=transfer_trajectory,
            departure_state=dro.states[0],
            insertion_state=ro.states[0],
            label="Test Transfer",
        )
        legend = ax.get_legend()
        has_label = False
        if legend:
            texts = [t.get_text() for t in legend.get_texts()]
            has_label = "Test Transfer" in texts
        assert has_label
        plt.close("all")
