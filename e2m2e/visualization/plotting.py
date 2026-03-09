"""
轨道可视化模块 (Orbit Visualization Module)

提供圆形限制性三体问题（CR3BP）中轨道的各种可视化功能，包括：
- 3D轨道绘制
- 2D投影（XY, XZ, YZ平面）
- 平动点（Libration Points）标注
- 主天体和次天体绘制
- 庞加莱截面（Poincaré Section）
- 轨道族可视化
- Jacobi常数变化图
- 稳定性分析图

使用指南：
==========

1. 基本使用：
   ```python
   from e2m2e.core.system import CR3BP_System
   from e2m2e.visualization.plotting import OrbitVisualizer

   # 创建系统
   system = CR3BP_System.from_known_system("earth_moon")
   system.set_characteristic_scales(384400, 27.32 * 86400)
   system.compute_libration_points()

   # 创建可视化器
   viz = OrbitVisualizer(system)

   # 绘制轨道（假设orbit是Orbit对象或状态数组）
   viz.plot_2d_projection(orbit, plane='xy')
   viz.plot_primary_bodies()
   viz.plot_libration_points()
   viz.show()
   ```

2. 高级功能：
   - 创建概览图：viz.create_overview_plot(orbit)
   - 绘制3D轨道：viz.plot_3d_orbit(orbit)
   - 绘制轨道族：viz.plot_orbit_family(family_result)
   - 绘制庞加莱截面：viz.plot_poincare_section(orbit, plane='y', value=0.0)
   - 绘制Jacobi常数：viz.plot_jacobi_constant(orbit)

3. 自定义设置：
   ```python
   # 修改图形大小
   viz.figsize = (10, 6)

   # 修改轨道样式
   viz.orbit_linewidth = 2.0
   viz.orbit_alpha = 0.9

   # 修改天体颜色
   viz.primary_body_color = 'orange'
   viz.secondary_body_color = 'gray'
   ```

4. 保存图形：
   ```python
   viz.save('orbit_plot.png', dpi=300)  # 保存为PNG
   viz.save('orbit_plot.pdf')           # 保存为PDF
   ```

依赖：
------
- numpy >= 1.20.0
- matplotlib >= 3.5.0

注意：
------
1. 在Jupyter notebook中使用时，可能需要添加 `%matplotlib inline`
2. 可以多次调用绘图函数在同一图形上叠加多个轨道
3. 使用 `viz.figure` 和 `viz.axes` 可以访问底层的matplotlib对象进行进一步自定义
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
    """轨道可视化器 (Orbit Visualizer)

    提供圆形限制性三体问题（CR3BP）中轨道的各种可视化方法。

    主要功能：
    ----------
    1. 3D轨道可视化
    2. 2D投影（XY, XZ, YZ平面）
    3. 平动点（L1-L5）标注
    4. 主天体和次天体绘制
    5. 轨道族可视化
    6. 庞加莱截面
    7. Jacobi常数变化图
    8. 稳定性分析图

    属性：
    -----
    system : CR3BP_System
        关联的CR3BP系统对象
    figsize : tuple
        图形大小，默认 (12, 8)
    dpi : int
        图形分辨率，默认 100
    orbit_linewidth : float
        轨道线宽，默认 1.5
    orbit_alpha : float
        轨道透明度，默认 0.8
    primary_body_color : str
        主天体颜色，默认 "gold"
    secondary_body_color : str
        次天体颜色，默认 "silver"

    示例：
    -----
    ```python
    # 创建可视化器
    viz = OrbitVisualizer(system)

    # 绘制2D投影
    viz.plot_2d_projection(orbit, plane='xy')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.show()

    # 创建概览图
    viz.create_overview_plot(orbit)
    viz.show()
    ```
    """

    DEFAULT_FIGURE_SIZE = (12, 8)
    DEFAULT_DPI = 100

    def __init__(self, system):
        """初始化轨道可视化器

        参数：
        ----------
        system : CR3BP_System
            关联的CR3BP系统对象。用于获取系统参数（如质量参数mu）和计算平动点位置。

        注意：
        ----
        系统对象是必需的，因为可视化需要知道：
        1. 质量参数mu（用于定位主天体和次天体）
        2. 平动点位置
        3. 特征尺度（用于坐标转换）

        示例：
        -----
        ```python
        from e2m2e.core.system import CR3BP_System
        from e2m2e.visualization.plotting import OrbitVisualizer

        # 创建系统
        system = CR3BP_System.from_known_system("earth_moon")
        system.set_characteristic_scales(384400, 27.32 * 86400)
        system.compute_libration_points()

        # 创建可视化器
        viz = OrbitVisualizer(system)
        ```
        """
        self.system = system
        self.mu = system.mu

        # 图形对象
        self.figure = None  # 当前图形对象
        self.axes = None  # 当前2D坐标轴
        self.axes_3d = None  # 当前3D坐标轴

        # 设置
        self.figsize = self.DEFAULT_FIGURE_SIZE  # 图形大小
        self.dpi = self.DEFAULT_DPI  # 分辨率

        # 绘图样式
        self.orbit_linewidth = 1.5  # 轨道线宽
        self.orbit_alpha = 0.8  # 轨道透明度
        self.color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # 颜色循环
        self.color_index = 0  # 当前颜色索引

        # 天体颜色和大小
        self.primary_body_color = "gold"  # 主天体颜色（如地球）
        self.primary_body_size = 200  # 主天体大小
        self.secondary_body_color = "silver"  # 次天体颜色（如月球）
        self.secondary_body_size = 100  # 次天体大小

        # 平动点设置（L1-L5）
        self.libration_point_colors = ["red", "blue", "green", "purple", "orange"]  # 颜色
        self.libration_point_markers = ["o", "s", "^", "D", "*"]  # 标记符号
        self.libration_point_sizes = [100, 100, 100, 150, 150]  # 大小
        self.libration_point_labels = ["L1", "L2", "L3", "L4", "L5"]  # 标签

    def _get_next_color(self):
        """获取下一个颜色"""
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def plot_3d_orbit(self, orbit, color=None, label=None, ax=None, show_start=True):
        """绘制3D轨道

        在3D空间中绘制轨道，可以显示轨道的三维形状和空间分布。

        参数：
        ----------
        orbit : Orbit 或 numpy.ndarray
            轨道数据。可以是：
            - Orbit对象（包含states属性）
            - 形状为 (n, 6) 的numpy数组，每行包含 [x, y, z, vx, vy, vz]
        color : str, 可选
            轨道颜色。如果为None，使用自动颜色循环
        label : str, 可选
            图例标签
        ax : matplotlib.axes._subplots.Axes3DSubplot, 可选
            现有的3D坐标轴。如果为None，创建新的坐标轴
        show_start : bool, 可选
            是否在轨道起点标记点，默认 True

        返回：
        -------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D坐标轴对象

        示例：
        -----
        ```python
        # 绘制3D轨道
        ax = viz.plot_3d_orbit(orbit, color='blue', label='Lyapunov Orbit')

        # 添加天体和平动点
        viz.plot_primary_bodies(ax=ax, is_3d=True)
        viz.plot_libration_points(ax=ax, is_3d=True)

        # 添加图例
        ax.legend()

        # 显示图形
        viz.show()
        ```

        注意：
        ----
        1. 如果orbit是Orbit对象，会自动提取states属性
        2. 可以多次调用此函数在同一坐标轴上叠加多个轨道
        3. 使用ax参数可以将轨道绘制到现有的3D坐标轴上
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

        将3D轨道投影到指定的2D平面上，便于分析轨道在特定平面上的形状。

        参数：
        ----------
        orbit : Orbit 或 numpy.ndarray
            轨道数据。可以是：
            - Orbit对象（包含states属性）
            - 形状为 (n, 6) 的numpy数组
        plane : ProjectionPlane 或 str, 可选
            投影平面，默认 ProjectionPlane.XY
            可选值：
            - 'xy' 或 ProjectionPlane.XY: XY平面投影
            - 'xz' 或 ProjectionPlane.XZ: XZ平面投影  
            - 'yz' 或 ProjectionPlane.YZ: YZ平面投影
        color : str, 可选
            轨道颜色。如果为None，使用自动颜色循环
        label : str, 可选
            图例标签
        ax : matplotlib.axes._axes.Axes, 可选
            现有的2D坐标轴。如果为None，创建新的坐标轴
        show_start : bool, 可选
            是否在轨道起点标记点，默认 True

        返回：
        -------
        ax : matplotlib.axes._axes.Axes
            2D坐标轴对象

        示例：
        -----
        ```python
        # 绘制XY平面投影
        ax = viz.plot_2d_projection(orbit, plane='xy', color='red', label='XY Projection')
        
        # 添加天体和平动点
        viz.plot_primary_bodies(ax=ax)
        viz.plot_libration_points(ax=ax)
        
        # 显示图形
        viz.show()
        
        # 绘制XZ平面投影
        viz.plot_2d_projection(orbit, plane='xz', color='green', label='XZ Projection')
        viz.show()
        ```

        注意：
        ----
        1. 投影平面使用无量纲坐标
        2. 自动设置等比例坐标轴（ax.set_aspect('equal')）
        3. 添加网格线便于观察
        4. 可以多次调用此函数在同一坐标轴上叠加多个轨道
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

        创建一个包含四个子图的综合视图，显示轨道的3D视图和三个2D投影。
        这是快速了解轨道整体特性的最佳方式。

        参数：
        ----------
        orbit : Orbit 或 numpy.ndarray
            轨道数据。可以是：
            - Orbit对象（包含states属性）
            - 形状为 (n, 6) 的numpy数组

        返回：
        -------
        figure : matplotlib.figure.Figure
            matplotlib图形对象

        子图布局：
        ---------
        1. 左上 (221): 3D轨道视图
        2. 右上 (222): XY平面投影（包含天体和平动点）
        3. 左下 (223): XZ平面投影
        4. 右下 (224): YZ平面投影

        示例：
        -----
        ```python
        # 创建概览图
        fig = viz.create_overview_plot(orbit)
        
        # 显示图形
        viz.show()
        
        # 保存图形
        viz.save('orbit_overview.png', dpi=300)
        ```

        注意：
        ----
        1. 图形大小为 (16, 12)，适合详细观察
        2. 使用tight_layout自动调整子图间距
        3. XY投影子图中会自动添加天体和平动点
        4. 3D子图中只显示轨道，不添加其他元素以保持清晰
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
