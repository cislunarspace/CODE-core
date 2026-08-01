"""基础可视化模块

提供轨道可视化的核心类 OrbitVisualizer，支持 2D 投影和 3D 轨道绘图。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ...algorithm.dynamics import CR3BP_System, LibrationPoint
from ...data.templates.enums import ProjectionPlane
from . import icons
from .config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class OrbitVisualizer:
    """轨道可视化器，支持 2D 投影和 3D 轨道绑定绘图。

    提供统一接口绘制单条轨道的 2D 投影、3D 轨道、天体标记和平动点标注。
    支持自定义颜色、线条宽度等样式参数。

    Args:
        system: CR3BP 系统对象，用于获取天体位置和平动点坐标。
        config: 绘图配置，未指定时使用默认 PlotConfig。
    """

    def __init__(self, system: CR3BP_System, config: PlotConfig | None = None) -> None:
        """初始化可视化器。

        Args:
            system: CR3BP 系统对象，用于获取天体位置和平动点坐标。
            config: 绘图配置，未指定时使用默认 PlotConfig。
        """
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
        # 在构造时从 matplotlib rcParams 捕获颜色循环，
        # 避免后续调用时 rcParams 已被修改导致颜色不一致
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

        # 天体图标（PNG 图片缓存，懒加载）
        self._primary_body_image: Any | None = None  # 主天体（地球）图片
        self._secondary_body_image: Any | None = None  # 次天体（月球）图片
        self._icon_loaded: bool = False

    def _ensure_icons_loaded(self) -> None:
        """懒加载天体图标 PNG 图片到 ``self._primary_body_image/_secondary_body_image``。

        路径解析委托给 :func:`icons.resolve_icon_dir`，避免硬编码
        ``~/Downloads``。加载失败时静默回退，不影响绘图流程。

        Note:
            PIL Image 会转换为 numpy array 以便 matplotlib OffsetImage 使用。
        """
        if self._icon_loaded:
            return

        icon_dir = icons.resolve_icon_dir(self.config.icon_path)
        primary, secondary = icons.load_body_icons(
            icon_dir,
            self.config.primary_body_icon,
            self.config.secondary_body_icon,
        )
        self._primary_body_image = primary
        self._secondary_body_image = secondary
        self._icon_loaded = True

    def _get_body_icon(self, is_primary: bool, size: int) -> tuple[Any | None, bool]:
        """获取天体图标和是否可用的元组。

        Args:
            is_primary: True 表示主天体（地球），False 表示次天体（月球）
            size: 目标像素大小（用于计算缩放比例）

        Returns:
            (OffsetImage 或 PIL Image, 是否可用) 元组
        """
        self._ensure_icons_loaded()

        image = self._primary_body_image if is_primary else self._secondary_body_image
        if image is None:
            return None, False

        return icons.make_offset_image(image, size), True

    def _plot_body_marker_3d(self, ax: Any, x: float, color: str, size: int, label: str) -> None:
        """在 3D axes 上画一个天体的圆形 marker（图标加载失败时的回退）。"""
        ax.plot(
            [x],
            [0],
            [0],
            marker="o",
            color=color,
            markersize=(size**0.5),
            markeredgecolor="black",
            markeredgewidth=1,
            linestyle="None",
            label=label,
        )

    def _plot_body_marker_2d(
        self, ax: Any, pos: np.ndarray, color: str, edge: str, size: int, label: str
    ) -> None:
        """在 2D axes 上画一个天体的散点 marker（图标加载失败时的回退）。"""
        ax.scatter(
            *pos,
            color=color,
            s=size,  # type: ignore[misc]
            edgecolors=edge,
            linewidth=1.5,
            zorder=10,
            label=label,
        )

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

    def plot(self, data: Any, config: object = None, **kwargs) -> Any:
        """统一绘图入口，委托到 plot_3d_orbit。
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
                x[0],
                y[0],
                z[0],
                color=color,
                marker="o",
                s=50,
                edgecolors="black",
                linewidth=1,
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

        if self.system.L_points is None:
            raise ValueError(
                "Libration points not computed on system. Call compute_libration_points() first."
            )

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
                ax.plot(
                    [coord[0]],
                    [coord[1]],
                    [coord[2]],
                    marker=marker,
                    color=color,
                    markersize=(size**0.5),
                    linestyle="None",
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
        2D 图表优先使用 PNG 图标。3D 图表也优先使用 PNG 图标，通过 Billboard
        技术将图标"贴"在 3D 空间中的固定位置（不随视角旋转），图标加载失败时
        回退到圆形 marker。

        Args:
            ax: 目标 axes 对象。
            is_3d: 是否在 3D 坐标系中绘制。

        Returns:
            matplotlib axes 对象。
        """
        # mu=None 时静默返回而非 raise：允许无系统上下文的纯装饰性绘图，
        # 部分子类方法可能不需要天体标记
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
            # 3D 图表：尝试用 PNG 图标（Billboard：通过 draw_event 回调
            # 把 3D 数据点投影到 2D 屏幕坐标，每次重绘都同步图标位置）。
            primary_icon, primary_ok = self._get_body_icon(
                is_primary=True,
                size=int(self.primary_body_size * self.config.primary_body_icon_scale),
            )
            secondary_icon, secondary_ok = self._get_body_icon(
                is_primary=False,
                size=int(self.secondary_body_size * self.config.secondary_body_icon_scale),
            )

            if primary_ok and secondary_ok:
                icons.add_3d_billboard_icon(ax, primary_icon, (-self.mu, 0.0, 0.0), primary_name)
                icons.add_3d_billboard_icon(
                    ax, secondary_icon, (1 - self.mu, 0.0, 0.0), secondary_name
                )
            else:
                # 图标加载失败，回退到圆形 marker
                self._plot_body_marker_3d(
                    ax, -self.mu, self.primary_body_color, self.primary_body_size, primary_name
                )
                self._plot_body_marker_3d(
                    ax,
                    1 - self.mu,
                    self.secondary_body_color,
                    self.secondary_body_size,
                    secondary_name,
                )
        else:
            # 2D 图表优先使用 PNG 图标
            primary_pos = np.array([-self.mu, 0])
            secondary_pos = np.array([1 - self.mu, 0])

            # 尝试加载并使用图标（应用图标缩放系数）
            primary_icon, primary_ok = self._get_body_icon(
                is_primary=True,
                size=int(self.primary_body_size * self.config.primary_body_icon_scale),
            )
            secondary_icon, secondary_ok = self._get_body_icon(
                is_primary=False,
                size=int(self.secondary_body_size * self.config.secondary_body_icon_scale),
            )

            if primary_ok and secondary_ok:
                # 成功加载图标，使用 AnnotationBbox
                # 确保图标不为 None（类型断言）
                assert primary_icon is not None
                assert secondary_icon is not None

                icons.add_2d_icon(
                    ax, primary_icon, (float(primary_pos[0]), float(primary_pos[1])), primary_name
                )
                icons.add_2d_icon(
                    ax,
                    secondary_icon,
                    (float(secondary_pos[0]), float(secondary_pos[1])),
                    secondary_name,
                )
            else:
                # 图标加载失败，回退到圆形散点
                self._plot_body_marker_2d(
                    ax, primary_pos, "#2E86AB", "#1A5276", self.primary_body_size, primary_name
                )
                self._plot_body_marker_2d(
                    ax,
                    secondary_pos,
                    "#95A5A6",
                    "#566573",
                    self.secondary_body_size,
                    secondary_name,
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
        """使用最近邻算法排序散点，使绘制的连线不交叉。

        采用贪心最近邻策略：从离原点最远的点出发，每步选择最近的未访问点。
        选择最远点作为起点是为了让排序方向与轨道自然延伸一致。

        Args:
            x: 散点的 x 坐标数组。
            y: 散点的 y 坐标数组。

        Returns:
            (sorted_x, sorted_y) 排序后的坐标元组。
        """
        points = np.column_stack((x, y))
        if len(points) <= 2:
            return x, y
        order = self._greedy_nearest_neighbor_order(points)
        return points[order, 0], points[order, 1]

    def _sort_3d_points_by_nearest_neighbor(self, x, y, z):
        """使用最近邻算法排序 3D 散点，使绘制的连线不交叉。

        三维版本，逻辑与 :meth:`_sort_points_by_nearest_neighbor` 相同，
        但距离计算包含 z 分量。

        Args:
            x: 散点的 x 坐标数组。
            y: 散点的 y 坐标数组。
            z: 散点的 z 坐标数组。

        Returns:
            (sorted_x, sorted_y, sorted_z) 排序后的坐标元组。
        """
        points = np.column_stack((x, y, z))
        if len(points) <= 2:
            return x, y, z
        order = self._greedy_nearest_neighbor_order(points)
        return points[order, 0], points[order, 1], points[order, 2]

    @staticmethod
    def _greedy_nearest_neighbor_order(points: np.ndarray) -> np.ndarray:
        """贪心最近邻遍历，返回点的访问顺序索引数组。

        从离原点最远的点出发，每步选最近的未访问点。维数由 ``points.shape[1]`` 决定。

        Args:
            points: ``(n, d)`` 的坐标数组，``d >= 1``。

        Returns:
            ``(n,)`` 的整数索引数组，按访问顺序排列。
        """
        n = len(points)
        visited = np.zeros(n, dtype=bool)
        order = np.empty(n, dtype=int)
        start_idx = int(np.argmax(np.sum(points * points, axis=1)))
        current = start_idx
        for i in range(n):
            visited[current] = True
            order[i] = current
            if i == n - 1:
                break
            # 计算到所有未访问点的平方距离，已访问置 inf
            diff = points - points[current]
            dists = np.einsum("ij,ij->i", diff, diff)
            dists[visited] = np.inf
            current = int(np.argmin(dists))
        return order
