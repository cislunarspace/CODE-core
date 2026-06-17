"""转移轨迹可视化模块

提供转移轨道、搜索结果的可视化工具。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..core.cr3bp_system import CR3BP_System
from .base import OrbitVisualizer
from .config import PlotConfig

if TYPE_CHECKING:
    pass


class TransferPlotter(OrbitVisualizer):
    """转移轨迹可视化器，继承自 OrbitVisualizer。

    支持绘制转移搜索结果的参数空间散点图和 3D 转移轨道。

    Args:
        system: CR3BP 系统对象。
        config: 绘图配置。
    """

    def __init__(self, system: CR3BP_System, config: PlotConfig | None = None) -> None:
        """初始化转移轨迹可视化器。

        Args:
            system: CR3BP 系统对象。
            config: 绘图配置。
        """
        super().__init__(system, config)

    def plot(self, data: Any, config: object = None, **kwargs) -> Any:
        """统一绘图入口，委托到 plot_solution_plane。
        data 应为搜索结果列表（TransferOptimizationResult 列表或 dict 列表）。

        Args:
            data: 搜索结果列表。
            config: 可选的 PlotConfig 配置对象。
            **kwargs: 传递给 plot_solution_plane 的额外参数。

        Returns:
            matplotlib axes 对象。
        """
        return self.plot_solution_plane(data, **kwargs)

    def plot_solution_plane(
        self,
        results,
        color_by: str | None = None,
        ax: Any | None = None,
        show_colorbar: bool = True,
    ) -> Any:
        """绘制搜索结果散点图（转移时间 vs 总 Δv）。

        Args:
            results: 搜索结果列表（dict 或 TransferOptimizationResult）。
            color_by: 着色依据，如 "transfer_type"。
            ax: 目标 axes 对象。
            show_colorbar: 是否显示颜色条。

        Returns:
            matplotlib axes 对象。
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.config.dpi)

        # 解析结果，过滤优化失败的解
        parsed = self._parse_solution_results(results)
        valid = [r for r in parsed if r["success"]]
        if not valid:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax.set_xlabel("Transfer Time (T)")
            ax.set_ylabel(r"Total $\Delta v$")
            return ax

        times = np.array([r["transfer_time"] for r in valid])
        dvs = np.array([r["delta_v1"] + r["delta_v2"] for r in valid])

        # 按转移类型定义颜色映射
        type_colors = {
            "direct": "#1f77b4",
            "lga": "#ff7f0e",
            "external": "#2ca02c",
        }

        if color_by == "transfer_type":
            for ttype, color in type_colors.items():
                mask = np.array([str(r.get("transfer_type", "")).lower() == ttype for r in valid])
                if mask.any():
                    ax.scatter(
                        times[mask], dvs[mask], c=color, s=10, alpha=0.7, label=ttype.upper()
                    )
            ax.legend()
        else:
            ax.scatter(times, dvs, s=10, alpha=0.7)

        ax.set_xlabel("Transfer Time (T)")
        ax.set_ylabel(r"Total $\Delta v$")
        ax.grid(True, alpha=0.3, linestyle="--")
        return ax

    def plot_transfer_orbit(
        self,
        departure_orbit: Any,
        arrival_orbit: Any,
        transfer_trajectory: npt.ArrayLike,
        departure_state: npt.ArrayLike,
        insertion_state: npt.ArrayLike,
        ax: Any | None = None,
        label: str | None = None,
        color: str | None = None,
    ) -> Any:
        """绘制 3D 转移轨道（出发轨道 + 到达轨道 + 转移弧段）。

        Args:
            departure_orbit: 出发轨道（DRO）。接受任何满足 OrbitContainer Protocol 的对象。
            arrival_orbit: 到达轨道（RO）。接受任何满足 OrbitContainer Protocol 的对象。
            transfer_trajectory: 转移轨迹状态数组 (n, 6)。
            departure_state: 出发点状态 [6]。
            insertion_state: 插入点状态 [6]。
            ax: 目标 3D axes 对象。
            label: 转移弧段图例标签。
            color: 转移弧段颜色。

        Returns:
            matplotlib 3D axes 对象。
        """
        transfer_states = np.asarray(transfer_trajectory)
        dep_states = self._extract_states(departure_orbit)
        arr_states = self._extract_states(arrival_orbit)

        if ax is None:
            fig = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
            self.figure = fig
            ax = fig.add_subplot(111, projection="3d")
            self.axes_3d = ax

        if color is None:
            color = self._get_next_color()

        # 分别绘制 DRO、RO 和转移弧段（DRO=蓝色，RO=橙色）
        ax.plot(
            dep_states[:, 0],
            dep_states[:, 1],
            dep_states[:, 2],
            color="steelblue",
            linewidth=self.orbit_linewidth,
            alpha=self.orbit_alpha,
            label="DRO",
        )
        ax.plot(
            arr_states[:, 0],
            arr_states[:, 1],
            arr_states[:, 2],
            color="darkorange",
            linewidth=self.orbit_linewidth,
            alpha=self.orbit_alpha,
            label="RO",
        )
        ax.plot(
            transfer_states[:, 0],
            transfer_states[:, 1],
            transfer_states[:, 2],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=label,
        )

        # 标记出发点和插入点
        dep = np.asarray(departure_state)
        ins = np.asarray(insertion_state)
        ax.scatter(
            dep[0],
            dep[1],
            dep[2],
            color="green",
            marker="^",
            s=80,
            edgecolors="black",
            linewidth=1,
            zorder=10,
            label="Departure",
        )
        ax.scatter(
            ins[0],
            ins[1],
            ins[2],
            color="red",
            marker="v",
            s=80,
            edgecolors="black",
            linewidth=1,
            zorder=10,
            label="Insertion",
        )

        self.plot_primary_bodies(ax=ax, is_3d=True)
        self.plot_libration_points(ax=ax, is_3d=True)

        ax.set_xlabel("X (nondimensional)")
        ax.set_ylabel("Y (nondimensional)")
        ax.set_zlabel("Z (nondimensional)")
        ax.legend()
        return ax

    def _parse_solution_results(self, results) -> list:
        """将搜索结果解析为统一的 dict 格式。

        支持两种输入：原始 dict 直接透传，TransferOptimizationResult 对象提取字段。

        Args:
            results: 搜索结果列表，元素为 dict 或 TransferOptimizationResult。

        Returns:
            统一的 dict 列表，每个 dict 包含 transfer_time/delta_v1/delta_v2/
            objective_value/success/transfer_type 字段。
        """
        if not results:
            return []
        parsed = []
        for r in results:
            if isinstance(r, dict):
                parsed.append(r)
            else:
                parsed.append(
                    {
                        "transfer_time": r.transfer_time,
                        "delta_v1": r.delta_v1,
                        "delta_v2": r.delta_v2,
                        "objective_value": getattr(r, "total_delta_v", r.delta_v1 + r.delta_v2),
                        "success": r.success,
                        # transfer_type 可能是枚举值（.value 取字符串）
                        # 也可能已经是字符串（来自反序列化结果），两种情况统一处理
                        "transfer_type": r.transfer_type.value
                        if hasattr(getattr(r, "transfer_type", None), "value")
                        else str(getattr(r, "transfer_type", "")),
                    }
                )
        return parsed
