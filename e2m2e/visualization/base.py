"""基础可视化模块

提供轨道可视化的核心类 OrbitVisualizer，支持 2D 投影和 3D 轨道绑图。

v4.0 MBSE 重构：参数类型使用 OrbitContainer Protocol，满足 Visualizer Protocol 接口。
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from ..core.system import CR3BP_System, LibrationPoint
from .config import PlotConfig

if TYPE_CHECKING:
    pass


class ProjectionPlane(Enum):
    """轨道投影视图平面枚举。"""

    XY = "xy"  # X-Y 平面（轨道面内投影）
    XZ = "xz"  # X-Z 平面（侧视图）
    YZ = "yz"  # Y-Z 平面（正视图）


class OrbitVisualizer:
    """轨道可视化器，支持 2D 投影和 3D 轨道绑定绘图。

    提供统一接口绘制单条轨道的 2D 投影、3D 轨道、天体标记和平动点标注。
    支持自定义颜色、线条宽度等样式参数。

    Args:
        system: CR3BP 系统对象，用于获取天体位置和平动点坐标。
        config: 绘图配置，未指定时使用默认 PlotConfig。
    """

    def __init__(self, system: CR3BP_System, config: PlotConfig | None = None) -> None:
        self.system = system
        self.mu = system.mu  # 质量参数，用于天体位置计算
        self.config = config or PlotConfig()

        # matplotlib 对象引用，延迟创建以避免不必要开销
        self.figure: Figure | None = None
        self.axes: Axes | None = None
        self.axes_3d: Any | None = None

        # 轨道样式参数
        self.orbit_linewidth = self.config.orbit_linewidth
        self.orbit_alpha = self.config.orbit_alpha
        self.color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_index = 0  # 颜色循环索引

        # 天体标记样式
        self.primary_body_color = self.config.primary_body_color
        self.primary_body_size = self.config.primary_body_size
        self.secondary_body_color = self.config.secondary_body_color
        self.secondary_body_size = self.config.secondary_body_size

        self.primary_body_use_marker = True
        self.secondary_body_use_marker = True
        self.primary_body_marker = "o"
        self.secondary_body_marker = "o"

        # 平动点标记样式（5 个平动点独立配置）
        self.libration_point_colors = list(self.config.lp_colors)
        self.libration_point_markers = list(self.config.lp_markers)
        self.libration_point_sizes = list(self.config.lp_sizes)
        self.libration_point_labels = ["L1", "L2", "L3", "L4", "L5"]
        self.libration_point_fontsize = self.config.lp_label

    def _get_next_color(self) -> str:
        """从颜色循环中获取下一个颜色。"""
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def _extract_states(self, orbit: Any) -> np.ndarray:
        """从轨道对象或数组中提取状态矩阵 (n, 6)。

        接受任何满足 OrbitContainer Protocol 的对象（有 .states 属性）
        或直接的 numpy 数组 / array-like。
        """
        states = orbit.states if hasattr(orbit, "states") else np.array(orbit)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return states

    # ------------------------------------------------------------------
    # Visualizer Protocol compliance
    # ------------------------------------------------------------------

    def plot(self, data: Any, config: object = None, **kwargs) -> Any:
        """Visualizer Protocol 入口方法。

        委托到 plot_3d_orbit，将 data 作为轨道参数传入。
        如果 config 不是 None，则替换当前 config（不修改原始对象）。

        Args:
            data: 待绘制的轨道数据（Orbit、OrbitContainer 或 array-like）。
            config: 可选的 PlotConfig 配置对象。
            **kwargs: 传递给 plot_3d_orbit 的额外参数。

        Returns:
            matplotlib axes 对象。
        """
        if config is not None and isinstance(config, PlotConfig):
            # 仅在本次调用中使用新 config，不修改实例属性
            saved = self.config
            self.config = config
            try:
                return self.plot_3d_orbit(data, **kwargs)
            finally:
                self.config = saved
        return self.plot_3d_orbit(data, **kwargs)

    def plot_3d_orbit(
        self,
        orbit: Any,
        color: str | None = None,
        label: str | None = None,
        ax: Any | None = None,
        show_start: bool = True,
    ) -> Any:
        """绘制 3D 轨道。

        Args:
            orbit: 轨道对象或状态数组。接受任何满足 OrbitContainer Protocol
                的对象（具有 .states 属性），或直接的 array-like。
            color: 线条颜色，未指定时自动从颜色循环取。
            label: 图例标签。
            ax: 目标 axes 对象，未指定时自动创建。
            show_start: 是否标记轨道起始点。

        Returns:
            matplotlib 3D axes 对象。
        """
        if ax is None:
            if self.axes_3d is None:
                self.figure = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
                self.axes_3d = self.figure.add_subplot(111, projection="3d")
            ax = self.axes_3d

        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]  # 提取位置分量

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

        return ax

    def plot_2d_projection(
        self,
        orbit: Any,
        plane: ProjectionPlane | str = ProjectionPlane.XY,
        color: str | None = None,
        label: str | None = None,
        ax: Any | None = None,
        show_start: bool = True,
    ) -> Any:
        """绘制轨道在指定平面上的 2D 投影。

        Args:
            orbit: 轨道对象或状态数组。接受任何满足 OrbitContainer Protocol
                的对象（具有 .states 属性），或直接的 array-like。
            plane: 投影平面（XY/XZ/YZ）。
            color: 线条颜色。
            label: 图例标签。
            ax: 目标 axes 对象。
            show_start: 是否标记轨道起始点。

        Returns:
            matplotlib axes 对象。
        """
        if ax is None:
            if self.axes is None:
                self.figure, self.axes = plt.subplots(
                    1, 1, figsize=self.config.figsize_2d, dpi=self.config.dpi
                )
            ax = self.axes

        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        if color is None:
            color = self._get_next_color()

        if isinstance(plane, str):
            plane = ProjectionPlane(plane)

        # 根据投影平面选择坐标轴
        if plane == ProjectionPlane.XY:
            px, py = x, y
        elif plane == ProjectionPlane.XZ:
            px, py = x, z
        elif plane == ProjectionPlane.YZ:
            px, py = y, z
        else:
            raise ValueError(f"Unknown projection plane: {plane}")

        ax.plot(
            px, py, color=color, label=label, linewidth=self.orbit_linewidth, alpha=self.orbit_alpha
        )

        if show_start and len(px) > 0:
            ax.scatter(px[0], py[0], color=color, marker="o", s=50, edgecolors="black", linewidth=1)

        return ax

    def plot_libration_points(
        self, ax: Any | None = None, show_labels: bool = True, is_3d: bool = False
    ) -> Any:
        """绘制五个平动点标记。

        Args:
            ax: 目标 axes 对象。
            show_labels: 是否显示 L1-L5 标签。
            is_3d: 是否在 3D 坐标系中绘制。

        Returns:
            matplotlib axes 对象。
        """
        if self.system is None or not self.system.has_L_points:
            if self.system is not None:
                self.system.compute_libration_points()
            else:
                return ax

        assert self.system.L_points is not None

        if ax is None:
            if is_3d and self.axes_3d is not None:
                ax = self.axes_3d
            elif self.axes is not None:
                ax = self.axes
            else:
                return ax

        # 遍历五个平动点，逐个绘制标记和标签
        for i, lp in enumerate(LibrationPoint):
            coord = self.system.L_points[lp]
            color = self.libration_point_colors[i]
            marker = self.libration_point_markers[i]
            size = self.libration_point_sizes[i]
            label_text = self.libration_point_labels[i]

            if is_3d:
                ax.scatter(
                    coord[0],
                    coord[1],
                    coord[2],
                    color=color,
                    marker=marker,
                    s=size,
                    zorder=5,  # type: ignore[misc]
                )
                if show_labels:
                    ax.text(
                        coord[0],
                        coord[1],
                        coord[2] + 0.02,
                        label_text,  # type: ignore[arg-type]
                        fontsize=self.libration_point_fontsize,
                        ha="center",
                    )
            else:
                ax.scatter(coord[0], coord[1], color=color, marker=marker, s=size, zorder=5)
                if show_labels:
                    ax.annotate(
                        label_text,
                        (coord[0], coord[1]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=self.libration_point_fontsize,
                    )
        return ax

    def plot_primary_bodies(self, ax: Any | None = None, is_3d: bool = False) -> Any:
        """绘制主天体和次天体标记。

        天体位置：主天体在 (-μ, 0, 0)，次天体在 (1-μ, 0, 0)（旋转系坐标）。

        Args:
            ax: 目标 axes 对象。
            is_3d: 是否在 3D 坐标系中绘制。

        Returns:
            matplotlib axes 对象。
        """
        if self.mu is None:
            return ax
        if ax is None:
            ax = self.axes_3d if is_3d else self.axes
            if ax is None:
                return ax

        # 获取天体名称，回退为默认值
        primary_name = getattr(self.system, "primary_body", None) or "Earth"
        secondary_name = getattr(self.system, "secondary_body", None) or "Moon"

        if is_3d:
            ax.scatter(
                -self.mu,
                0,
                0,
                color=self.primary_body_color,
                s=self.primary_body_size,  # type: ignore[misc]
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=primary_name,
            )
            ax.scatter(
                1 - self.mu,
                0,
                0,
                color=self.secondary_body_color,
                s=self.secondary_body_size,  # type: ignore[misc]
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=secondary_name,
            )
        else:
            primary_pos = np.array([-self.mu, 0])
            secondary_pos = np.array([1 - self.mu, 0])
            ax.scatter(
                *primary_pos,
                color="#2E86AB",
                s=self.primary_body_size,  # type: ignore[misc]
                edgecolors="#1A5276",
                linewidth=1.5,
                zorder=10,
                label=primary_name,
            )
            ax.scatter(
                *secondary_pos,
                color="#95A5A6",
                s=self.secondary_body_size,  # type: ignore[misc]
                edgecolors="#566573",
                linewidth=1.5,
                zorder=10,
                label=secondary_name,
            )
        return ax

    def show(self) -> None:
        """显示绘图窗口。"""
        plt.show()

    def save(self, filename: str, dpi: int | None = None) -> None:
        """保存图像到文件。

        Args:
            filename: 输出文件路径。
            dpi: 输出分辨率，未指定时使用配置中的默认值。
        """
        if self.figure is not None:
            self.figure.savefig(
                filename, dpi=dpi or self.config.dpi, bbox_inches="tight", pad_inches=0.1
            )

    def _sort_points_by_nearest_neighbor(self, x, y):
        """使用最近邻算法排序散点，使绘制的连线不交叉。"""
        points = np.column_stack((x, y))
        n = len(points)
        if n <= 2:
            return x, y
        distances_from_origin = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        start_idx = np.argmax(distances_from_origin)
        visited = np.zeros(n, dtype=bool)
        sorted_indices = np.zeros(n, dtype=int)
        current_idx = start_idx
        for i in range(n):
            visited[current_idx] = True
            sorted_indices[i] = current_idx
            if i == n - 1:
                break
            min_dist = np.inf
            nearest_idx = -1
            for j in range(n):
                if not visited[j]:
                    dist = np.sqrt(
                        (points[current_idx, 0] - points[j, 0]) ** 2
                        + (points[current_idx, 1] - points[j, 1]) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = j
            current_idx = nearest_idx
        return points[sorted_indices, 0], points[sorted_indices, 1]

    def _sort_3d_points_by_nearest_neighbor(self, x, y, z):
        """使用最近邻算法排序 3D 散点，使绘制的连线不交叉。"""
        points = np.column_stack((x, y, z))
        n = len(points)
        if n <= 2:
            return x, y, z
        distances_from_origin = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        start_idx = np.argmax(distances_from_origin)
        visited = np.zeros(n, dtype=bool)
        sorted_indices = np.zeros(n, dtype=int)
        current_idx = start_idx
        for i in range(n):
            visited[current_idx] = True
            sorted_indices[i] = current_idx
            if i == n - 1:
                break
            min_dist = np.inf
            nearest_idx = -1
            for j in range(n):
                if not visited[j]:
                    dist = np.sqrt(
                        (points[current_idx, 0] - points[j, 0]) ** 2
                        + (points[current_idx, 1] - points[j, 1]) ** 2
                        + (points[current_idx, 2] - points[j, 2]) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = j
            current_idx = nearest_idx
        return points[sorted_indices, 0], points[sorted_indices, 1], points[sorted_indices, 2]
