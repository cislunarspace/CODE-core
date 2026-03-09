"""
轨道可视化模块

提供3D轨道绘制、2D投影、平动点标注、天体绘制、庞加莱截面等可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class ProjectionPlane(Enum):
    """投影平面枚举"""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class OrbitVisualizer:
    """轨道可视化器

    提供CR3BP轨道的各种可视化方法。

    属性：
        system: CR3BP_System对象
        figsize: 图形大小
        dpi: 分辨率
    """

    DEFAULT_FIGURE_SIZE = (12, 8)
    DEFAULT_DPI = 100

    def __init__(self, system):
        """初始化可视化器

        参数：
        - system: CR3BP_System对象（可选）
        这里引入 system 对象的目的是，在画图时从 system 对象中获取mu值。
        """
        self.system = system
        self.mu = system.mu

        # 图形对象
        self.figure = None
        self.axes = None
        self.axes_3d = None

        # 设置
        self.figsize = self.DEFAULT_FIGURE_SIZE
        self.dpi = self.DEFAULT_DPI

        # 绘图样式
        self.orbit_linewidth = 1.5
        self.orbit_alpha = 0.8
        self.color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_index = 0

        # 天体颜色
        self.primary_body_color = "gold"
        self.primary_body_size = 200
        self.secondary_body_color = "silver"
        self.secondary_body_size = 100

        # 平动点设置
        self.libration_point_colors = ["red", "blue", "green", "purple", "orange"]
        self.libration_point_markers = ["o", "s", "^", "D", "*"]
        self.libration_point_sizes = [100, 100, 100, 150, 150]
        self.libration_point_labels = ["L1", "L2", "L3", "L4", "L5"]

    def _get_next_color(self):
        """获取下一个颜色"""
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def plot_3d_orbit(self, orbit, color=None, label=None, ax=None, show_start=True):
        """绘制3D轨道

        参数：
            orbit: Orbit对象或状态数组 (n, 6)
            color: 轨道颜色
            label: 标签
            ax: matplotlib 3D坐标轴
            show_start: 是否标记起点

        返回：
            ax: 3D坐标轴
        """
        if ax is None:
            if self.axes_3d is None:
                self.figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
                self.axes_3d = self.figure.add_subplot(111, projection="3d")
            ax = self.axes_3d

        # 提取轨道数据
        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        if color is None:
            color = self._get_next_color()

        ax.plot(
            x,
            y,
            z,
            color=color,
            label=label,
            linewidth=self.orbit_linewidth,
            alpha=self.orbit_alpha,
        )

        if show_start and len(x) > 0:
            ax.scatter(
                x[0], y[0], z[0], color=color, marker="o", s=50, edgecolors="black", linewidth=1
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return ax

    def plot_2d_projection(
        self, orbit, plane=ProjectionPlane.XY, color=None, label=None, ax=None, show_start=True
    ):
        """绘制2D投影

        参数：
            orbit: Orbit对象或状态数组
            plane: 投影平面
            color: 颜色
            label: 标签
            ax: matplotlib坐标轴
            show_start: 是否标记起点

        返回：
            ax: 坐标轴
        """
        if ax is None:
            if self.axes is None:
                self.figure, self.axes = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            ax = self.axes

        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        if color is None:
            color = self._get_next_color()

        # 根据投影平面选择坐标
        if isinstance(plane, str):
            plane = ProjectionPlane(plane)

        if plane == ProjectionPlane.XY:
            px, py = x, y
            xlabel, ylabel = "X", "Y"
        elif plane == ProjectionPlane.XZ:
            px, py = x, z
            xlabel, ylabel = "X", "Z"
        elif plane == ProjectionPlane.YZ:
            px, py = y, z
            xlabel, ylabel = "Y", "Z"
        else:
            raise ValueError(f"未知投影平面: {plane}")

        ax.plot(
            px, py, color=color, label=label, linewidth=self.orbit_linewidth, alpha=self.orbit_alpha
        )

        if show_start and len(px) > 0:
            ax.scatter(px[0], py[0], color=color, marker="o", s=50, edgecolors="black", linewidth=1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

        return ax

    def plot_libration_points(self, ax=None, show_labels=True, is_3d=False):
        """绘制平动点

        参数：
            ax: 坐标轴
            show_labels: 是否显示标签
            is_3d: 是否为3D坐标轴

        返回：
            ax: 坐标轴
        """
        if self.system is None or not self.system.has_L_points:
            if self.system is not None:
                self.system.compute_libration_points()
            else:
                return ax

        if ax is None:
            if is_3d and self.axes_3d is not None:
                ax = self.axes_3d
            elif self.axes is not None:
                ax = self.axes
            else:
                return ax

        from ..core.system import LibrationPoint

        for i, lp in enumerate(LibrationPoint):
            coord = self.system.L_points[lp]
            color = self.libration_point_colors[i]
            marker = self.libration_point_markers[i]
            size = self.libration_point_sizes[i]
            label_text = self.libration_point_labels[i]

            if is_3d:
                ax.scatter(
                    coord[0], coord[1], coord[2], color=color, marker=marker, s=size, zorder=5
                )
                if show_labels:
                    ax.text(
                        coord[0], coord[1], coord[2] + 0.02, label_text, fontsize=10, ha="center"
                    )
            else:
                ax.scatter(coord[0], coord[1], color=color, marker=marker, s=size, zorder=5)
                if show_labels:
                    ax.annotate(
                        label_text,
                        (coord[0], coord[1]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=10,
                    )

        return ax

    def plot_primary_bodies(self, ax=None, is_3d=False):
        """绘制主天体和次天体

        参数：
            ax: 坐标轴
            is_3d: 是否为3D

        返回：
            ax: 坐标轴
        """
        if self.mu is None:
            return ax

        if ax is None:
            ax = self.axes_3d if is_3d else self.axes
            if ax is None:
                return ax

        # 主天体位于 (-mu, 0, 0)
        # 次天体位于 (1-mu, 0, 0)
        primary_pos = np.array([-self.mu, 0])
        secondary_pos = np.array([1 - self.mu, 0])

        primary_label = self.system.primary_body if self.system else "Primary"
        secondary_label = self.system.secondary_body if self.system else "Secondary"

        if is_3d:
            ax.scatter(
                *[-self.mu, 0, 0],
                color=self.primary_body_color,
                s=self.primary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=primary_label,
            )
            ax.scatter(
                *[1 - self.mu, 0, 0],
                color=self.secondary_body_color,
                s=self.secondary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=secondary_label,
            )
        else:
            ax.scatter(
                *primary_pos,
                color=self.primary_body_color,
                s=self.primary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=primary_label,
            )
            ax.scatter(
                *secondary_pos,
                color=self.secondary_body_color,
                s=self.secondary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=secondary_label,
            )

        return ax

    def plot_orbit_family(
        self, family_result, plane=ProjectionPlane.XY, colormap="viridis", ax=None
    ):
        """绘制轨道族

        参数：
            family_result: Continuation返回的轨道族字典
            plane: 投影平面
            colormap: 颜色映射
            ax: 坐标轴

        返回：
            ax: 坐标轴
        """
        if ax is None:
            self.figure, self.axes = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            ax = self.axes

        orbits = family_result["orbits"]
        n_orbits = len(orbits)
        cmap = plt.cm.get_cmap(colormap)

        for i, orbit in enumerate(orbits):
            color = cmap(i / max(n_orbits - 1, 1))
            self.plot_2d_projection(orbit, plane=plane, color=color, ax=ax, show_start=False)

        # 添加天体和平动点
        self.plot_primary_bodies(ax=ax)
        self.plot_libration_points(ax=ax)

        ax.set_title(f"Orbit Family ({n_orbits} orbits)")
        return ax

    def plot_poincare_section(self, orbits, plane="y", value=0.0, ax=None):
        """绘制庞加莱截面

        参数：
            orbits: 轨道列表
            plane: 截面平面 ('x', 'y', 'z')
            value: 平面位置值
            ax: 坐标轴

        返回：
            ax: 坐标轴
        """
        if ax is None:
            self.figure, self.axes = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            ax = self.axes

        if not isinstance(orbits, list):
            orbits = [orbits]

        plane_map = {"x": 0, "y": 1, "z": 2}
        plane_idx = plane_map.get(plane, 1)

        for orbit in orbits:
            states = self._extract_states(orbit)
            n = len(states)

            # 检测截面穿越
            crossings = []
            plane_vals = states[:, plane_idx]
            for i in range(n - 1):
                if (plane_vals[i] - value) * (plane_vals[i + 1] - value) < 0:
                    # 线性插值找交叉点
                    frac = (value - plane_vals[i]) / (plane_vals[i + 1] - plane_vals[i])
                    crossing_state = states[i] + frac * (states[i + 1] - states[i])
                    crossings.append(crossing_state)

            if crossings:
                crossings = np.array(crossings)
                # 根据截面选择显示的坐标
                if plane == "y":
                    ax.scatter(crossings[:, 0], crossings[:, 3], s=1, alpha=0.5)
                    ax.set_xlabel("x")
                    ax.set_ylabel("vx")
                elif plane == "x":
                    ax.scatter(crossings[:, 1], crossings[:, 4], s=1, alpha=0.5)
                    ax.set_xlabel("y")
                    ax.set_ylabel("vy")
                elif plane == "z":
                    ax.scatter(crossings[:, 0], crossings[:, 3], s=1, alpha=0.5)
                    ax.set_xlabel("x")
                    ax.set_ylabel("vx")

        ax.set_title(f"Poincaré Section ({plane}={value})")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_jacobi_constant(self, orbit, ax=None):
        """绘制Jacobi常数随时间变化

        参数：
            orbit: Orbit对象
            ax: 坐标轴

        返回：
            ax: 坐标轴
        """
        if ax is None:
            self.figure, self.axes = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            ax = self.axes

        if hasattr(orbit, "jacobi_constants") and orbit.jacobi_constants is not None:
            ax.plot(orbit.times, orbit.jacobi_constants, "b-", linewidth=1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Jacobi Constant")
            ax.set_title("Jacobi Constant Conservation")
            ax.grid(True, alpha=0.3)

        return ax

    def plot_stability_diagram(self, family_result, ax=None):
        """绘制稳定性图

        参数：
            family_result: 轨道族结果
            ax: 坐标轴

        返回：
            ax: 坐标轴
        """
        if ax is None:
            self.figure, self.axes = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            ax = self.axes

        periods = family_result.get("periods", [])
        if len(periods) > 0:
            ax.plot(range(len(periods)), periods, "bo-", markersize=3)
            ax.set_xlabel("Orbit Index")
            ax.set_ylabel("Period")
            ax.set_title("Period Evolution")
            ax.grid(True, alpha=0.3)

        return ax

    def create_overview_plot(self, orbit):
        """创建轨道概览图（四子图）

        参数：
            orbit: Orbit对象

        返回：
            figure: matplotlib图形
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)

        # 3D轨道
        ax1 = fig.add_subplot(221, projection="3d")
        self.plot_3d_orbit(orbit, ax=ax1, label="Orbit")
        ax1.set_title("3D Orbit")

        # XY投影
        ax2 = fig.add_subplot(222)
        self.plot_2d_projection(orbit, plane=ProjectionPlane.XY, ax=ax2)
        self.plot_primary_bodies(ax=ax2)
        ax2.set_title("XY Projection")

        # XZ投影
        ax3 = fig.add_subplot(223)
        self.plot_2d_projection(orbit, plane=ProjectionPlane.XZ, ax=ax3)
        ax3.set_title("XZ Projection")

        # YZ投影
        ax4 = fig.add_subplot(224)
        self.plot_2d_projection(orbit, plane=ProjectionPlane.YZ, ax=ax4)
        ax4.set_title("YZ Projection")

        fig.suptitle("Orbit Overview", fontsize=16)
        fig.tight_layout()

        self.figure = fig
        return fig

    def show(self):
        """显示图形"""
        plt.show()

    def save(self, filename, dpi=None):
        """保存图形

        参数：
            filename: 文件名
            dpi: 分辨率
        """
        if self.figure is not None:
            self.figure.savefig(filename, dpi=dpi or self.dpi, bbox_inches="tight", pad_inches=0.1)

    def _extract_states(self, orbit):
        """从Orbit对象或数组中提取状态数据"""
        if hasattr(orbit, "states"):
            states = orbit.states
        else:
            states = np.array(orbit)

        if states.ndim == 1:
            states = states.reshape(1, -1)

        return states

    def __str__(self):
        return f"OrbitVisualizer(system={self.system})"
