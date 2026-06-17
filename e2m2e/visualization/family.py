"""轨道族可视化模块

提供轨道族的 2D/3D 绘图、Jacobi-周期-稳定性分析图和概览图。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..core.cr3bp_system import CR3BP_System
from .base import OrbitVisualizer
from .config import PlotConfig

if TYPE_CHECKING:
    pass


class FamilyPlotter(OrbitVisualizer):
    """轨道族可视化器，继承自 OrbitVisualizer。

    支持按 Jacobi 常数着色的轨道族 2D/3D 绘图，以及 Jacobi-周期-稳定性组合图。

    Args:
        system: CR3BP 系统对象。
        config: 绘图配置。
    """

    def __init__(self, system: CR3BP_System, config: PlotConfig | None = None) -> None:
        """初始化轨道族可视化器。

        Args:
            system: CR3BP 系统对象。
            config: 绘图配置。
        """
        super().__init__(system, config)

    def plot(self, data: Any, config: object = None, **kwargs) -> Any:
        """统一绘图入口，委托到 plot_family_2d。
        data 应为可迭代的轨道集合（OrbitFamily、List[Orbit] 等）。

        Args:
            data: 轨道族数据（OrbitFamily 或轨道列表）。
            config: 可选的 PlotConfig 配置对象。
            **kwargs: 传递给 plot_family_2d 的额外参数（如 jacobi_values 等）。

        Returns:
            (fig, ax) 元组。
        """
        jacobi_values = kwargs.pop("jacobi_values", None)
        if jacobi_values is None:
            jacobi_values = []
        return self.plot_family_2d(data, jacobi_values=jacobi_values, **kwargs)

    def _get_jacobi_norm(self, jacobi_values):
        """计算 Jacobi 常数的归一化范围 [0, 1]。

        Args:
            jacobi_values: Jacobi 常数列表。

        Returns:
            (jmin, jmax, jrange) 元组，jrange 防止除零（全相等时为 1.0）。
        """
        if not jacobi_values:
            return 0.0, 1.0, 1.0
        jmin = min(jacobi_values)
        jmax = max(jacobi_values)
        jrange = jmax - jmin if jmax != jmin else 1.0
        return jmin, jmax, jrange

    def _draw_orbit_loop_2d(
        self, family_result, jacobi_values, ax, plane="xy", start=0, end=None, step=1
    ):
        """按 Jacobi 常数着色批量绘制轨道族的 2D 投影。

        Args:
            family_result: 轨道族（可迭代的轨道集合）。
            jacobi_values: 各轨道对应的 Jacobi 常数。
            ax: 目标 axes 对象。
            plane: 投影平面（"xy"/"xz"/"yz"）。
            start: 起始轨道索引。
            end: 终止轨道索引（含），None 表示到末尾。
            step: 绘图步长（用于降采样）。
        """
        jmin, jmax, jrange = self._get_jacobi_norm(jacobi_values)
        cmap = self.config.get_cmap()
        n = len(family_result) if end is None else min(end + 1, len(family_result))
        for idx in range(start, n, step):
            orbit = family_result[idx]
            norm_j = (jacobi_values[idx] - jmin) / jrange
            color = cmap(norm_j)
            self.plot_2d_projection(orbit, plane=plane, color=color, show_start=False, ax=ax)

    def _draw_orbit_loop_3d(self, family_result, jacobi_values, ax, start=0, end=None, step=1):
        """按 Jacobi 常数着色批量绘制轨道族的 3D 视图。

        Args:
            family_result: 轨道族（可迭代的轨道集合）。
            jacobi_values: 各轨道对应的 Jacobi 常数。
            ax: 目标 3D axes 对象。
            start: 起始轨道索引。
            end: 终止轨道索引（含），None 表示到末尾。
            step: 绘图步长（用于降采样）。
        """
        jmin, jmax, jrange = self._get_jacobi_norm(jacobi_values)
        cmap = self.config.get_cmap()
        n = len(family_result) if end is None else min(end + 1, len(family_result))
        for idx in range(start, n, step):
            orbit = family_result[idx]
            norm_j = (jacobi_values[idx] - jmin) / jrange
            color = cmap(norm_j)
            self.plot_3d_orbit(orbit, color=color, ax=ax, show_start=False)

    def _add_colorbar(self, ax, jacobi_values, shrink=0.8, pad=None):
        """添加 Jacobi 常数颜色条到 axes 旁。

        Args:
            ax: 目标 axes 对象。
            jacobi_values: Jacobi 常数列表，用于确定颜色条范围。
            shrink: 颜色条高度缩放比例。
            pad: 颜色条与 axes 的间距。

        Returns:
            matplotlib Colorbar 对象。
        """
        jmin, jmax, _ = self._get_jacobi_norm(jacobi_values)
        cmap = self.config.get_cmap()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=jmin, vmax=jmax))
        sm.set_array([])
        kwargs = {"shrink": shrink}
        if pad is not None:
            kwargs["pad"] = pad
        cbar = plt.colorbar(sm, ax=ax, **kwargs)
        cbar.set_label("Jacobi Constant", fontsize=self.config.colorbar)
        cbar.ax.tick_params(labelsize=self.config.tick)
        return cbar

    def _style_2d_ax(self, ax, xlabel="X (nondimensional)", ylabel="Y (nondimensional)"):
        """设置 2D axes 的标签、刻度和等比例。

        Args:
            ax: 目标 axes 对象。
            xlabel: x 轴标签文本。
            ylabel: y 轴标签文本。
        """
        ax.set_xlabel(xlabel, fontsize=self.config.label)
        ax.set_ylabel(ylabel, fontsize=self.config.label)
        ax.tick_params(labelsize=self.config.tick)
        ax.set_aspect("equal")

    def _style_3d_ax(self, ax):
        """设置 3D axes 的标签和刻度。

        Args:
            ax: 目标 3D axes 对象。
        """
        ax.set_xlabel("X (nondimensional)", fontsize=self.config.label)
        ax.set_ylabel("Y (nondimensional)", fontsize=self.config.label)
        ax.set_zlabel("Z (nondimensional)", fontsize=self.config.label)
        ax.tick_params(labelsize=self.config.tick)

    def plot_family_2d(
        self,
        family_result,
        jacobi_values: list[float],
        title: str = "",
        plane: str = "xy",
        xlim=None,
        ylim=None,
        show_bodies: bool = True,
        show_libration: bool = True,
        show_colorbar: bool = True,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        save_path: str | None = None,
        show: bool = True,
    ):
        """绘制轨道族的 2D 投影图，按 Jacobi 常数着色。

        Args:
            family_result: 轨道族（OrbitFamily 或 List[Orbit]）。
            jacobi_values: 各轨道对应的 Jacobi 常数。
            title: 图标题。
            plane: 投影平面。
            xlim: x 轴范围。
            ylim: y 轴范围。
            show_bodies: 是否绘制天体标记。
            show_libration: 是否绘制平动点。
            show_colorbar: 是否显示 Jacobi 颜色条。
            start: 起始索引。
            end: 终止索引。
            step: 步长（用于降采样）。
            save_path: 保存路径。
            show: 是否显示窗口。

        Returns:
            (fig, ax) 元组。
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_2d, dpi=self.config.dpi)

        self._draw_orbit_loop_2d(
            family_result, jacobi_values, ax, plane=plane, start=start, end=end, step=step
        )

        if show_bodies:
            self.plot_primary_bodies(ax=ax)
        if show_libration:
            self.plot_libration_points(ax=ax)
        if show_colorbar:
            self._add_colorbar(ax, jacobi_values)

        xlabel = "X (nondimensional)"
        ylabel = "Y (nondimensional)"
        if plane == "xz":
            ylabel = "Z (nondimensional)"
        elif plane == "yz":
            xlabel = "Y (nondimensional)"
            ylabel = "Z (nondimensional)"
        self._style_2d_ax(ax, xlabel, ylabel)

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        if title:
            ax.set_title(title, fontsize=self.config.title, y=self.config.title_y_offset)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_family_3d(
        self,
        family_result,
        jacobi_values: list[float],
        title: str = "",
        center: tuple[float, float, float] = (0.5, 0.0, 0.0),
        radius: float = 0.65,
        elev: int = 0,
        azim: int = -90,
        show_bodies: bool = True,
        show_libration: bool = True,
        show_colorbar: bool = True,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        save_path: str | None = None,
        show: bool = True,
    ):
        """绘制轨道族的 3D 视图，按 Jacobi 常数着色。

        Args:
            family_result: 轨道族。
            jacobi_values: 各轨道对应的 Jacobi 常数。
            title: 图标题。
            center: 3D 视图中心坐标。
            radius: 3D 视图半径范围。
            elev: 仰角。
            azim: 方位角。
            show_bodies: 是否绘制天体标记。
            show_libration: 是否绘制平动点。
            show_colorbar: 是否显示颜色条。
            start: 起始索引。
            end: 终止索引。
            step: 步长。
            save_path: 保存路径。
            show: 是否显示窗口。

        Returns:
            (fig, ax) 元组。
        """
        fig = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        if not family_result or len(family_result) == 0:
            self._style_3d_ax(ax)
            return fig, ax

        if jacobi_values is None:
            jacobi_values = [0.0] * len(family_result)

        # 天体/平动点用 plot+marker 而非 scatter，保证 3D 深度排序正确
        # （scatter 的 Path3DCollection 会被整体推到前景，导致挡住后方轨道线）
        if show_bodies:
            self.plot_primary_bodies(ax=ax, is_3d=True)
        if show_libration:
            self.plot_libration_points(ax=ax, show_labels=True, is_3d=True)
        self._draw_orbit_loop_3d(family_result, jacobi_values, ax, start=start, end=end, step=step)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        self._style_3d_ax(ax)
        ax.view_init(elev=elev, azim=azim)

        if show_colorbar:
            self._add_colorbar(ax, jacobi_values, shrink=0.6, pad=0.1)

        if title:
            ax.set_title(title, fontsize=self.config.title, y=self.config.title_y_offset_3d)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_jacobi_period(
        self,
        jacobi_values: list[float],
        periods,
        title: str = "",
        target_period: float | None = None,
        save_path: str | None = None,
        show: bool = True,
    ):
        """绘制 Jacobi 常数 vs 周期的单 Y 轴折线图。

        按 Jacobi 常数排序后绘制周期曲线，可选 target_period 参考线。

        Args:
            jacobi_values: Jacobi 常数序列。
            periods: 轨道周期序列。
            title: 图标题。
            target_period: 目标周期参考线。
            save_path: 保存路径。
            show: 是否显示窗口。

        Returns:
            (fig, ax) 元组。
        """
        sorted_indices = sorted(range(len(jacobi_values)), key=lambda i: jacobi_values[i])
        j_sorted = [jacobi_values[i] for i in sorted_indices]
        p_sorted = [periods[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=self.config.figsize_dual, dpi=self.config.dpi)

        color_period = "tab:blue"
        ax.set_xlabel("Jacobi Constant", fontsize=self.config.label)
        ax.set_ylabel("Period (nondimensional)", color=color_period, fontsize=self.config.label)
        ax.plot(j_sorted, p_sorted, "-", color=color_period, linewidth=2, label="Period")
        ax.tick_params(axis="y", labelcolor=color_period, labelsize=self.config.tick)
        ax.tick_params(axis="x", labelsize=self.config.tick)

        if target_period is not None:
            ax.axhline(
                y=target_period,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Target T={target_period:.3f}",
            )

        ax.legend(loc="upper right", fontsize=self.config.legend)

        if title:
            ax.set_title(title, fontsize=self.config.title, y=self.config.title_y_offset_dual)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_jacobi_period_stability(
        self,
        jacobi_values: list[float],
        periods,
        stability_values: list[float],
        title: str = "",
        target_period: float | None = None,
        save_path: str | None = None,
        show: bool = True,
    ):
        """绘制双 Y 轴图：Jacobi 常数 vs 周期 + 稳定性指标。

        左轴为周期，右轴为最大 Lyapunov 指标（λmax），按 Jacobi 常数排序。

        Args:
            jacobi_values: Jacobi 常数序列。
            periods: 轨道周期序列。
            stability_values: 稳定性指标序列。
            title: 图标题。
            target_period: 目标周期参考线。
            save_path: 保存路径。
            show: 是否显示窗口。

        Returns:
            (fig, ax) 元组。
        """
        fig, ax1 = plt.subplots(figsize=self.config.figsize_dual, dpi=self.config.dpi)

        # 按 Jacobi 常数排序以便绘制单调曲线
        sorted_indices = sorted(range(len(jacobi_values)), key=lambda i: jacobi_values[i])
        j_sorted = [jacobi_values[i] for i in sorted_indices]
        p_sorted = [periods[i] for i in sorted_indices]
        s_sorted = [stability_values[i] for i in sorted_indices]

        color_period = "tab:blue"
        ax1.set_xlabel("Jacobi Constant", fontsize=self.config.label)
        ax1.set_ylabel("Period (nondimensional)", color=color_period, fontsize=self.config.label)
        (line_period,) = ax1.plot(
            j_sorted, p_sorted, "-", color=color_period, linewidth=2, label="Period"
        )
        ax1.tick_params(axis="y", labelcolor=color_period, labelsize=self.config.tick)
        ax1.tick_params(axis="x", labelsize=self.config.tick)

        if target_period is not None:
            ax1.axhline(
                y=target_period,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Target T={target_period:.3f}",
            )

        ax2 = ax1.twinx()
        color_stability = "tab:red"
        ax2.set_ylabel("Stability Index (λmax)", color=color_stability, fontsize=self.config.label)
        (line_stability,) = ax2.plot(
            j_sorted,
            s_sorted,
            "-",
            color=color_stability,
            linewidth=2,
            label="Stability Index (λmax)",
        )
        ax2.tick_params(axis="y", labelcolor=color_stability, labelsize=self.config.tick)

        lines = [line_period, line_stability]
        labels_str = [str(line.get_label()) for line in lines]
        if target_period is not None:
            lines.append(ax1.get_lines()[-1])
            labels_str.append(f"Target T={target_period:.3f}")
        ax1.legend(lines, labels_str, loc="upper right", fontsize=self.config.legend)

        if title:
            ax1.set_title(title, fontsize=self.config.title, y=self.config.title_y_offset_dual)

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax1

    def plot_family_overview(
        self,
        family_result,
        jacobi_values: list[float],
        periods,
        stability_values: list[float],
        suptitle: str = "",
        plane: str = "xy",
        center_3d: tuple[float, float, float] = (0.5, 0.0, 0.0),
        radius_3d: float = 0.65,
        zoom_xlim=None,
        zoom_ylim=None,
        elev: int = 0,
        azim: int = -90,
        target_period: float | None = None,
        step: int = 1,
        save_path: str | None = None,
        show: bool = True,
    ):
        """绘制轨道族概览图（2x2 子图布局）。

        布局：
        - 左上：全局 2D 投影（Jacobi 着色 + 颜色条）
        - 右上：局部放大 2D 投影
        - 左下：Jacobi vs 周期 + 稳定性双 Y 轴图
        - 右下：3D 视图

        Args:
            family_result: 轨道族。
            jacobi_values: Jacobi 常数序列。
            periods: 周期序列。
            stability_values: 稳定性指标序列。
            suptitle: 超标题。
            plane: 2D 投影平面。
            center_3d: 3D 视图中心。
            radius_3d: 3D 视图半径。
            zoom_xlim: 放大视图的 x 轴范围。
            zoom_ylim: 放大视图的 y 轴范围。
            elev: 3D 仰角。
            azim: 3D 方位角。
            target_period: 目标周期参考线。
            step: 绘图步长。
            save_path: 保存路径。
            show: 是否显示窗口。

        Returns:
            matplotlib Figure 对象。
        """
        n_orbits = len(family_result)
        fig = plt.figure(figsize=self.config.figsize_overview, dpi=self.config.dpi)
        fs = self.config  # 缩短引用

        # 子图 1：全局 2D 投影（Jacobi 着色）
        ax1 = fig.add_subplot(221)
        self._draw_orbit_loop_2d(family_result, jacobi_values, ax1, plane=plane, step=step)
        self.plot_primary_bodies(ax=ax1)
        self.plot_libration_points(ax=ax1)
        self._add_colorbar(ax1, jacobi_values)
        ax1.set_title(
            f"Global {plane.upper()} View ({n_orbits} orbits)",
            fontsize=fs.title,
            y=fs.title_y_offset_subplot,
        )
        xlabel = "X" if plane in ("xy", "xz") else "Y"
        ylabel = "Y" if plane == "xy" else "Z"
        ax1.set_xlabel(xlabel, fontsize=fs.label)
        ax1.set_ylabel(ylabel, fontsize=fs.label)
        ax1.tick_params(labelsize=fs.tick)
        ax1.set_aspect("equal")

        # 子图 2：局部放大 2D 视图
        ax2 = fig.add_subplot(222)
        self._draw_orbit_loop_2d(family_result, jacobi_values, ax2, plane=plane, step=step)
        self.plot_primary_bodies(ax=ax2)
        self.plot_libration_points(ax=ax2)
        if zoom_xlim:
            ax2.set_xlim(*zoom_xlim)
        if zoom_ylim:
            ax2.set_ylim(*zoom_ylim)
        ax2.set_title(
            f"Zoomed {plane.upper()} View", fontsize=fs.title, y=fs.title_y_offset_subplot
        )
        ax2.set_xlabel(xlabel, fontsize=fs.label)
        ax2.set_ylabel(ylabel, fontsize=fs.label)
        ax2.tick_params(labelsize=fs.tick)
        ax2.set_aspect("equal")

        # 子图 3：Jacobi vs 周期 & 稳定性（双 Y 轴）
        ax3 = fig.add_subplot(223)
        ax3.set_xlabel("Jacobi Constant", fontsize=fs.label)
        ax3.set_ylabel("Period", color="tab:blue", fontsize=fs.label)
        (line_p,) = ax3.plot(jacobi_values, periods, "o-", color="tab:blue", markersize=4)
        ax3.tick_params(axis="y", labelcolor="tab:blue", labelsize=fs.tick)
        ax3.tick_params(axis="x", labelsize=fs.tick)
        if target_period is not None:
            ax3.axhline(y=target_period, color="green", linestyle="--", linewidth=1.5)
        ax3_right = ax3.twinx()
        ax3_right.set_ylabel("λmax", color="tab:red", fontsize=fs.label)
        (line_s,) = ax3_right.plot(
            jacobi_values, stability_values, "s-", color="tab:red", markersize=4
        )
        ax3_right.tick_params(axis="y", labelcolor="tab:red", labelsize=fs.tick)
        ax3.set_title(
            "Jacobi vs Period & Stability", fontsize=fs.title, y=fs.title_y_offset_subplot
        )
        ax3.legend([line_p, line_s], ["Period", "λmax"], loc="upper right", fontsize=fs.legend)
        ax3.grid(True, alpha=0.3)

        # 子图 4：3D 视图
        ax4 = fig.add_subplot(224, projection="3d")
        self.plot_primary_bodies(ax=ax4, is_3d=True)
        self._draw_orbit_loop_3d(family_result, jacobi_values, ax4, step=step)
        ax4.set_xlim(center_3d[0] - radius_3d, center_3d[0] + radius_3d)
        ax4.set_ylim(center_3d[1] - radius_3d, center_3d[1] + radius_3d)
        ax4.set_zlim(center_3d[2] - radius_3d, center_3d[2] + radius_3d)
        ax4.set_title("3D View", fontsize=fs.title, y=fs.title_y_offset_3d)
        ax4.set_xlabel("X", fontsize=fs.label)
        ax4.set_ylabel("Y", fontsize=fs.label)
        ax4.set_zlabel("Z", fontsize=fs.label)
        ax4.tick_params(labelsize=fs.tick)
        ax4.view_init(elev=elev, azim=azim)

        if suptitle:
            fig.suptitle(suptitle, fontsize=fs.suptitle, fontweight="bold")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig
