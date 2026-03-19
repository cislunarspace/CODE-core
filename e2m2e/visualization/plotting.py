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

from __future__ import annotations

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D

from ..core import CR3BP_Dynamics
from ..core.system import CR3BP_System, LibrationPoint
from ..core.orbit import Orbit


def configure_academic_fonts():
    """配置学术规范的字体设置

    此函数配置matplotlib使用学术出版标准的字体：
    - 英文使用Times New Roman
    - 设置合适的字号大小
    - 配置数学字体为stix（符合学术规范）
    """
    # 使用Times New Roman作为主字体，DejaVu Serif作为后备处理特殊符号
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9

    # 设置数学字体 - 使用stix但允许特殊符号回退
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["mathtext.rm"] = "serif"
    matplotlib.rcParams["mathtext.it"] = "serif:italic"
    matplotlib.rcParams["mathtext.bf"] = "serif:bold"
    # 允许Unicode字符
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 改进图例样式
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["legend.framealpha"] = 0.9
    matplotlib.rcParams["legend.fancybox"] = True
    matplotlib.rcParams["legend.shadow"] = False


class ProjectionPlane(Enum):
    """投影平面枚举"""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


def compute_stability_for_family(family_result, system):
    """计算轨道族的稳定性指数

    稳定性指数（Stability Index）是判断轨道长期稳定性的重要指标。
    在圆形限制性三体问题中，通过计算单值矩阵（Monodromy Matrix）
    的特征值来确定轨道的稳定性：
        - λ_max = 1：线性中性稳定
        - λ_max < 1：渐近稳定（小扰动会衰减）
        - λ_max > 1：不稳定（小扰动会放大）

    参数：
        family_result: OrbitFamily对象，包含多条轨道的轨道族
        system: CR3BP_System对象，用于提供动力学模型

    返回：
        list: 稳定性指数列表，每个元素对应一条轨道的最大特征值模长
    """
    # 创建CR3BP动力学模型，用于后续计算状态转移矩阵
    dynamics = CR3BP_Dynamics(system)

    stability_values = []

    # 处理空轨道族的边界情况
    if family_result is None or len(family_result) == 0:
        return stability_values

    # 遍历轨道族中的每一条轨道，计算其稳定性指数
    for i, orbit in enumerate(family_result):
        # 确保每条轨道关联到指定的系统（用于动力学计算）
        if orbit.system is None:
            orbit.system = system

        try:
            # 如果轨道没有周期信息，假设为中性稳定（稳定性指数=1.0）  \\TODO 这也是不合适的，正是这个假设，使得我之前画图的时候存在间断点。
            if orbit.period is None:
                stability_values.append(1.0)
                continue

            # =========================================================
            # 核心计算步骤：
            # 1. 计算单值矩阵（Monodromy Matrix）
            #    沿轨道积分一个周期得到的状态转移矩阵 M
            # 2. 求单值矩阵的特征值 λ_i（CR3BP是4维状态空间，有4个特征值）
            # 3. 取特征值模长的最大值作为稳定性指数
            # =========================================================

            # 计算单值矩阵：从轨道起点出发，积分一个周期返回的状态转移矩阵
            monodromy = dynamics.compute_state_transition_matrix(orbit.states[0], orbit.period)

            # 计算单值矩阵的特征值
            eigenvalues = np.linalg.eigvals(monodromy)

            # 取所有特征值的模长
            magnitudes = np.abs(eigenvalues)

            # 稳定性指数 = 最大特征值模长
            stability_idx = np.max(magnitudes)
            stability_values.append(stability_idx)

        except Exception:
            # 计算失败时，假设为中性稳定 //TODO 这个假设可能不太好，正是这个假设，使得我之前画图的时候存在间断点。
            stability_values.append(1.0)

    return stability_values


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

    def __init__(self, system: CR3BP_System, apply_academic_style: bool = True) -> None:
        """初始化轨道可视化器

        参数：
        ----------
        system : CR3BP_System
            关联的CR3BP系统对象。用于获取系统参数（如质量参数mu）和计算平动点位置。
        apply_academic_style : bool
            是否应用学术规范的字体和样式设置，默认True

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
        # 应用学术字体配置
        if apply_academic_style:
            configure_academic_fonts()

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

        # 天体图标样式（使用Unicode符号或标记）
        self.primary_body_use_marker = True  # 是否使用标记符号代替真实图标
        self.secondary_body_use_marker = True
        self.primary_body_marker = "o"  # 地球标记形状
        self.secondary_body_marker = "o"  # 月球标记形状

        # 平动点设置（L1-L5）
        self.libration_point_colors = ["red", "blue", "green", "purple", "orange"]  # 颜色
        self.libration_point_markers = ["o", "s", "^", "D", "*"]  # 标记符号
        self.libration_point_sizes = [100, 100, 100, 150, 150]  # 大小
        self.libration_point_labels = ["L1", "L2", "L3", "L4", "L5"]  # 标签

    def _get_next_color(self) -> str:
        """获取下一个颜色"""
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def plot_3d_orbit(
        self,
        orbit: Union[Orbit, npt.ArrayLike],
        color: Optional[str] = None,
        label: Optional[str] = None,
        ax: Optional[Any] = None,
        show_start: bool = True,
    ) -> Any:
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
        self,
        orbit: Union[Orbit, npt.ArrayLike],
        plane: Union[ProjectionPlane, str] = ProjectionPlane.XY,
        color: Optional[str] = None,
        label: Optional[str] = None,
        ax: Optional[Any] = None,
        show_start: bool = True,
    ) -> Any:
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

    def plot_libration_points(
        self, ax: Optional[Any] = None, show_labels: bool = True, is_3d: bool = False
    ) -> Any:
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

    def plot_primary_bodies(self, ax: Optional[Any] = None, is_3d: bool = False) -> Any:
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

        # 获取天体名称
        primary_name = getattr(self.system, "primary_body", None) or "Earth"
        secondary_name = getattr(self.system, "secondary_body", None) or "Moon"

        if is_3d:
            ax.scatter(
                *[-self.mu, 0, 0],
                color=self.primary_body_color,
                s=self.primary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=primary_name,
            )
            ax.scatter(
                *[1 - self.mu, 0, 0],
                color=self.secondary_body_color,
                s=self.secondary_body_size,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                label=secondary_name,
            )
        else:
            # 简化的绘制方式：使用蓝色代表地球，灰色代表月球
            # 地球 - 蓝色圆形
            ax.scatter(
                *primary_pos,
                color="#2E86AB",  # 蓝色
                s=self.primary_body_size,
                edgecolors="#1A5276",  # 深蓝色边框
                linewidth=1.5,
                zorder=10,
                label=primary_name,
            )

            # 月球 - 灰色圆形
            ax.scatter(
                *secondary_pos,
                color="#95A5A6",  # 灰色
                s=self.secondary_body_size,
                edgecolors="#566573",  # 深灰色边框
                linewidth=1.5,
                zorder=10,
                label=secondary_name,
            )

        return ax

    def _plot_body_with_image(
        self,
        ax: Any,
        position: np.ndarray,
        image: Any,
        size: float,
        label: Optional[str] = None,
        zorder: int = 10,
    ):
        """使用图像绘制天体

        参数：
            ax: 坐标轴
            position: 位置 [x, y]
            image: PIL Image对象
            size: 图像大小（points²）
            label: 图例标签
            zorder: 绘制顺序
        """
        try:
            from PIL import Image

            # 计算图像在数据坐标中的尺寸
            # 获取坐标轴的数据limits和图形尺寸
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fig = ax.get_figure()
            fig_width, fig_height = fig.get_size_inches() * fig.dpi

            # 将points转换为数据坐标
            # 假设图像是圆形的，计算直径
            data_width = (xlim[1] - xlim[0]) * size / fig_width
            data_height = (ylim[1] - ylim[0]) * size / fig_height
            data_size = min(data_width, data_height)

            # 创建图像 annotation
            imagebox = offsetbox.OffsetImage(image, zoom=0.15, resample=True)

            # 创建 annotation
            ab = offsetbox.AnnotationBbox(
                imagebox,
                position,
                frameon=False,
                boxcoords="data",
                box_alignment=(0.5, 0.5),
                zorder=zorder,
            )
            ax.add_artist(ab)

            # 添加图例（使用标记）
            if label:
                ax.scatter(
                    [position[0]], [position[1]], marker="o", s=0, zorder=zorder - 1, label=label
                )
        except Exception as e:
            # 如果失败，回退到简单绘制
            print(f"图像绘制失败: {e}")
            ax.scatter(
                *position,
                color="blue" if "Earth" in label or "earth" in label else "gray",
                s=size,
                edgecolors="black",
                linewidth=1,
                zorder=zorder,
                label=label,
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

    def create_overview_plot(self, orbit: Union[Orbit, npt.ArrayLike]) -> Any:
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

    def plot_3d_orbit_family(
        self,
        family_result,
        jacobi_values: Optional[List[float]] = None,
        center: Tuple[float, float, float] = (0.99, 0.0, 0.0),
        radius: float = 0.40,
        show_colorbar: bool = True,
        show_legend: bool = False,
        seed_label: Optional[str] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """绘制3D轨道族（局部放大视图）

        在3D空间中绘制轨道族，聚焦于指定区域（类似2D放大图）。

        参数：
        ----------
        family_result : OrbitFamily
            轨道族对象，包含多条轨道
        jacobi_values : list of float, 可选
            Jacobi常数列表，用于颜色映射。如果为None，将自动计算
        center : tuple of float, 可选
            3D视图中心坐标 (x, y, z)，默认 (0.99, 0.0, 0.0)
        radius : float, 可选
            视图半径，默认 0.40
        show_colorbar : bool, 可选
            是否显示颜色条，默认 True
        show_legend : bool, 可选
            是否显示图例，默认 False
        seed_label : str, 可选
            种子轨道的图例标签
        ax : matplotlib.axes._subplots.Axes3DSubplot, 可选
            现有的3D坐标轴。如果为None，创建新的坐标轴

        返回：
        -------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D坐标轴对象

        示例：
        -----
        ```python
        # 绘制3D轨道族（局部放大）
        ax = viz.plot_3d_orbit_family(
            family_result,
            jacobi_values=jacobi_values,
            center=(0.99, 0.0, 0.0),
            radius=0.40,
            seed_label="Seed DRO"
        )
        viz.show()
        ```

        注意：
        ----
        1. 该方法类似于2D的局部放大图，但是是在3D空间中展示
        2. 使用coolwarm颜色映射，Jacobi常数从低到高（能量高到能量低）
        3. 种子轨道（第一条）用红色绘制，后续轨道用颜色映射
        """
        n_orbits = len(family_result) if family_result is not None else 0
        if n_orbits == 0:
            return ax

        # 创建坐标轴
        if ax is None:
            self.figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = self.figure.add_subplot(111, projection="3d")

        # 获取或计算Jacobi常数
        if jacobi_values is None:
            if hasattr(family_result, "get_jacobi_constants"):
                jacobi_values = family_result.get_jacobi_constants().tolist()
            else:
                jacobi_values = [3.0] * n_orbits  # 默认值

        # 处理空列表情况
        if not jacobi_values:
            jacobi_values = [3.0] * n_orbits

        # 颜色映射
        cmap = matplotlib.colormaps["coolwarm"]
        jacobi_min = min(jacobi_values)
        jacobi_max = max(jacobi_values)
        jacobi_range = jacobi_max - jacobi_min if jacobi_max != jacobi_min else 1.0

        # 绘制种子轨道（第一条）
        if n_orbits > 0:
            seed_orbit = family_result[0]
            self.plot_3d_orbit(
                seed_orbit,
                color="red",
                label=seed_label or "Seed DRO",
                ax=ax,
                show_start=True,
            )

        # 绘制其他轨道
        for idx in range(1, n_orbits):
            orbit = family_result[idx]
            norm_jacobi = (jacobi_values[idx] - jacobi_min) / jacobi_range
            color = cmap(norm_jacobi)
            self.plot_3d_orbit(
                orbit,
                color=color,
                ax=ax,
                show_start=False,
            )

        # 设置坐标轴范围（局部放大）
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        # 添加主次天体（地球和月球）到3D图中
        self.plot_primary_bodies(ax=ax, is_3d=True)

        # 添加拉格朗日点（平动点）到3D图中
        self.plot_libration_points(ax=ax, show_labels=True, is_3d=True)

        # 坐标轴标签
        ax.set_xlabel("X (nondimensional)", fontsize=12)
        ax.set_ylabel("Y (nondimensional)", fontsize=12)
        ax.set_zlabel("Z (nondimensional)", fontsize=12)

        # 标题
        ax.set_title(
            f"DRO Family (3D Zoomed View)\n"
            f"X: [{center[0] - radius:.2f}, {center[0] + radius:.2f}], "
            f"Y/Z: [±{radius:.2f}], {n_orbits} orbits",
            fontsize=12,
        )

        # 颜色条
        if show_colorbar and jacobi_values:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=mcolors.Normalize(vmin=jacobi_min, vmax=jacobi_max)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label("Jacobi Constant", fontsize=11)

        # 图例
        if show_legend:
            ax.legend(loc="upper right", fontsize=10)

        # 调整视角：地球在左侧、月球在右侧（azim=90从+Y方向看）
        ax.view_init(elev=20, azim=90)

        self.axes_3d = ax
        return ax

    def plot_resonant_orbit_family(
        self,
        family_result,
        label: str = "RO",
        target_period: Optional[float] = None,
        jacobi_values: Optional[List[float]] = None,
        stability_values: Optional[List[float]] = None,
        show_plots: bool = True,
    ) -> Tuple[Any, ...]:
        """绘制共振轨道族（Resonant Orbit Family）四视图

        创建四个子图展示共振轨道族的完整信息：
        1. XY平面投影（全局视图）
        2. XY平面投影（局部放大视图）
        3. 3D全局视图
        4. 3D局部放大视图

        参数：
        ----------
        family_result : OrbitFamily
            轨道族对象，包含多条共振轨道
        label : str, 可选
            轨道族标签，默认 "RO"。用于标题，如 "3:2 RO"
        target_period : float, 可选
            目标轨道周期。如果为None，自动选择周期最接近的轨道
        jacobi_values : list of float, 可选
            Jacobi常数列表。如果为None，自动计算
        stability_values : list of float, 可选
            稳定性指数列表。如果为None，自动计算
        show_plots : bool, 可选
            是否调用plt.show()显示图形，默认True

        返回：
        -------
        tuple : (fig_xy, ax_zoom, fig_3d, ax_3d_zoom)
            四个图像/坐标轴对象的元组

        示例：
        -----
        ```python
        # 创建可视化器
        viz = OrbitVisualizer(system)

        # 绘制RO族四视图
        fig_xy, ax_zoom, fig_3d, ax_3d_zoom = viz.plot_resonant_orbit_family(
            family_result,
            label="3:2",
            target_period=12.566,
        )

        # 保存图形
        fig_xy.savefig('ro_family_xy.png', dpi=300)
        ```
        """
        n_orbits = len(family_result) if family_result else 0
        if n_orbits == 0:
            return None, None, None, None

        # 确保system关联
        if family_result.system is None:
            family_result.system = self.system
        for orbit in family_result:
            if orbit.system is None:
                orbit.system = self.system

        # 计算Jacobi常数
        if jacobi_values is None:
            jacobi_values = family_result.get_jacobi_constants().tolist()

        # 计算稳定性指数
        if stability_values is None:
            stability_values = compute_stability_for_family(family_result, self.system)

        # 颜色映射
        cmap = matplotlib.colormaps["coolwarm"]
        jacobi_min = min(jacobi_values)
        jacobi_max = max(jacobi_values)
        jacobi_range = jacobi_max - jacobi_min if jacobi_max != jacobi_min else 1.0

        # 打印轨道信息摘要
        print(f"\n{label} RO族信息:")
        print(f"  轨道数量: {n_orbits}")
        print(f"  Jacobi常数: {jacobi_min:.6f} ~ {jacobi_max:.6f}")
        if stability_values:
            print(f"  稳定性指数: {min(stability_values):.6f} ~ {max(stability_values):.6f}")

        # 选择目标轨道
        if target_period is not None:
            periods = family_result.get_periods()
            idx_target = np.argmin(np.abs(periods - target_period))
        else:
            idx_target = 0
        orbit_target = family_result[idx_target]

        # 计算轨道范围
        all_x = np.concatenate([family_result[i].states[:, 0] for i in range(n_orbits)])
        all_y = np.concatenate([family_result[i].states[:, 1] for i in range(n_orbits)])
        zoom_center_x = (np.min(all_x) + np.max(all_x)) / 2
        zoom_range_x = (np.max(all_x) - np.min(all_x)) / 2 + 0.1
        zoom_range_y = np.max(np.abs(all_y)) + 0.1

        # ============================================================
        # 图1: RO族轨道（XY平面投影）- 全局视图
        # ============================================================
        fig_xy, ax_xy = plt.subplots(figsize=(12, 8))

        for idx in range(n_orbits):
            orbit = family_result[idx]
            norm_jacobi = (jacobi_values[idx] - jacobi_min) / jacobi_range
            color = cmap(norm_jacobi)
            self.plot_2d_projection(orbit, plane="xy", color=color, show_start=False, ax=ax_xy)

        # 标记目标RO
        self.plot_2d_projection(
            orbit_target,
            plane="xy",
            color="black",
            label=f"Target {label} (T={orbit_target.period:.4f})",
            ax=ax_xy,
        )

        # 添加天体和拉格朗日点
        self.plot_primary_bodies(ax=ax_xy)
        self.plot_libration_points(ax=ax_xy)

        # 颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=jacobi_min, vmax=jacobi_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_xy, shrink=0.8)
        cbar.set_label("Jacobi Constant", fontsize=12)

        # 标签
        ax_xy.set_xlabel("X (nondimensional)", fontsize=12)
        ax_xy.set_ylabel("Y (nondimensional)", fontsize=12)
        stability_str = ""
        if stability_values:
            stability_str = (
                f", lambda_max = [{min(stability_values):.4f}, {max(stability_values):.4f}]"
            )
        ax_xy.set_title(
            f"{label} RO Family in Earth-Moon CR3BP (XY Plane) - {n_orbits} orbits\n"
            f"C = [{jacobi_min:.4f}, {jacobi_max:.4f}]{stability_str}",
            fontsize=12,
        )
        ax_xy.legend(loc="upper right")
        ax_xy.set_aspect("equal")
        plt.tight_layout()

        # ============================================================
        # 图2: 局部放大图（RO族区域）- 2D视图
        # ============================================================
        fig_zoom, ax_zoom = plt.subplots(figsize=(10, 8))

        for idx in range(n_orbits):
            orbit = family_result[idx]
            norm_jacobi = (jacobi_values[idx] - jacobi_min) / jacobi_range
            color = cmap(norm_jacobi)
            self.plot_2d_projection(orbit, plane="xy", color=color, show_start=False, ax=ax_zoom)

        # 标记目标RO
        self.plot_2d_projection(
            orbit_target,
            plane="xy",
            color="black",
            label=f"Target {label} (T={orbit_target.period:.4f})",
            ax=ax_zoom,
        )

        # 添加天体和拉格朗日点
        self.plot_primary_bodies(ax=ax_zoom)
        self.plot_libration_points(ax=ax_zoom)

        # 设置坐标轴范围
        ax_zoom.set_xlim(zoom_center_x - zoom_range_x, zoom_center_x + zoom_range_x)
        ax_zoom.set_ylim(-zoom_range_y, zoom_range_y)

        ax_zoom.set_xlabel("X (nondimensional)", fontsize=12)
        ax_zoom.set_ylabel("Y (nondimensional)", fontsize=12)
        ax_zoom.set_title(
            f"{label} RO Family (Full View)\n"
            f"X: [{zoom_center_x - zoom_range_x:.2f}, {zoom_center_x + zoom_range_x:.2f}], "
            f"Y: [{-zoom_range_y:.2f}, {zoom_range_y:.2f}]",
            fontsize=12,
        )
        ax_zoom.legend(loc="upper right")
        ax_zoom.set_aspect("equal")
        plt.tight_layout()

        # ============================================================
        # 图3: 全局三维视图
        # ============================================================
        fig_3d, ax_3d = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d"})

        global_radius = 1.2  # 覆盖整个RO族范围

        # 绘制轨道
        for idx in range(n_orbits):
            orbit = family_result[idx]
            norm_jacobi = (jacobi_values[idx] - jacobi_min) / jacobi_range
            color = cmap(norm_jacobi)
            self.plot_3d_orbit(orbit, color=color, ax=ax_3d, show_start=False)

        # 标记目标RO
        self.plot_3d_orbit(
            orbit_target,
            color="black",
            label=f"Target {label} (T={orbit_target.period:.4f})",
            ax=ax_3d,
            show_start=True,
        )

        # 设置坐标轴范围
        ax_3d.set_xlim(-global_radius, global_radius)
        ax_3d.set_ylim(-global_radius, global_radius)
        ax_3d.set_zlim(-global_radius, global_radius)

        # 添加天体
        self.plot_primary_bodies(ax=ax_3d, is_3d=True)
        self.plot_libration_points(ax=ax_3d, show_labels=True, is_3d=True)

        ax_3d.set_xlabel("X (nondimensional)", fontsize=12)
        ax_3d.set_ylabel("Y (nondimensional)", fontsize=12)
        ax_3d.set_zlabel("Z (nondimensional)", fontsize=12)
        ax_3d.set_title(
            f"{label} RO Family in Earth-Moon CR3BP (3D View) - {n_orbits} orbits\n"
            f"C = [{jacobi_min:.4f}, {jacobi_max:.4f}]",
            fontsize=12,
        )

        # 颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=jacobi_min, vmax=jacobi_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_3d, shrink=0.6, pad=0.1)
        cbar.set_label("Jacobi Constant", fontsize=11)

        ax_3d.legend(loc="upper right")
        ax_3d.view_init(elev=0, azim=-90)
        plt.tight_layout()

        # ============================================================
        # 图4: 全范围三维视图（使用已有的plot_3d_orbit_family）
        # ============================================================
        seed_label_3d = f"Target {label} (C={jacobi_values[idx_target]:.4f})"
        radius_3d = max(zoom_range_x, zoom_range_y)
        ax_3d_zoom = self.plot_3d_orbit_family(
            family_result,
            jacobi_values=jacobi_values,
            center=(zoom_center_x, 0.0, 0.0),
            radius=radius_3d,
            show_colorbar=True,
            show_legend=True,
            seed_label=seed_label_3d,
        )

        ax_3d_zoom.view_init(elev=0, azim=-90)
        plt.tight_layout()

        # 一次性显示所有图表
        if show_plots:
            plt.show()

        return fig_xy, ax_zoom, fig_3d, ax_3d_zoom

    def show(self) -> None:
        """显示图形"""
        plt.show()

    def save(self, filename: str, dpi: Optional[int] = None) -> None:
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
