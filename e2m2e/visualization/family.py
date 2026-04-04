from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .base import OrbitVisualizer, ProjectionPlane
from .config import PlotConfig
from .stability import compute_stability_for_family
from ..core.system import CR3BP_System


class FamilyPlotter(OrbitVisualizer):
    def __init__(self, system: CR3BP_System, config: Optional[PlotConfig] = None) -> None:
        super().__init__(system, config)

    def _get_jacobi_norm(self, jacobi_values):
        jmin = min(jacobi_values)
        jmax = max(jacobi_values)
        jrange = jmax - jmin if jmax != jmin else 1.0
        return jmin, jmax, jrange

    def _draw_orbit_loop_2d(self, family_result, jacobi_values, ax,
                            plane="xy", start=0, end=None, step=1):
        jmin, jmax, jrange = self._get_jacobi_norm(jacobi_values)
        cmap = self.config.get_cmap()
        n = len(family_result) if end is None else min(end + 1, len(family_result))
        for idx in range(start, n, step):
            orbit = family_result[idx]
            norm_j = (jacobi_values[idx] - jmin) / jrange
            color = cmap(norm_j)
            self.plot_2d_projection(
                orbit, plane=plane, color=color, show_start=False, ax=ax)

    def _draw_orbit_loop_3d(self, family_result, jacobi_values, ax,
                            start=0, end=None, step=1):
        jmin, jmax, jrange = self._get_jacobi_norm(jacobi_values)
        cmap = self.config.get_cmap()
        n = len(family_result) if end is None else min(end + 1, len(family_result))
        for idx in range(start, n, step):
            orbit = family_result[idx]
            norm_j = (jacobi_values[idx] - jmin) / jrange
            color = cmap(norm_j)
            self.plot_3d_orbit(orbit, color=color, ax=ax, show_start=False)

    def _add_colorbar(self, ax, jacobi_values, shrink=0.8, pad=None):
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

    def _style_2d_ax(self, ax, xlabel="X (nondimensional)",
                     ylabel="Y (nondimensional)"):
        ax.set_xlabel(xlabel, fontsize=self.config.label)
        ax.set_ylabel(ylabel, fontsize=self.config.label)
        ax.tick_params(labelsize=self.config.tick)
        ax.set_aspect("equal")

    def _style_3d_ax(self, ax):
        ax.set_xlabel("X (nondimensional)", fontsize=self.config.label)
        ax.set_ylabel("Y (nondimensional)", fontsize=self.config.label)
        ax.set_zlabel("Z (nondimensional)", fontsize=self.config.label)
        ax.tick_params(labelsize=self.config.tick)

    def plot_family_2d(
        self,
        family_result,
        jacobi_values: List[float],
        title: str = "",
        plane: str = "xy",
        xlim=None,
        ylim=None,
        show_bodies: bool = True,
        show_libration: bool = True,
        show_colorbar: bool = True,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        fig, ax = plt.subplots(figsize=self.config.figsize_2d, dpi=self.config.dpi)

        self._draw_orbit_loop_2d(family_result, jacobi_values, ax,
                                 plane=plane, start=start, end=end, step=step)

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
        jacobi_values: List[float],
        title: str = "",
        center: Tuple[float, float, float] = (0.5, 0.0, 0.0),
        radius: float = 0.65,
        elev: int = 0,
        azim: int = -90,
        show_bodies: bool = True,
        show_libration: bool = True,
        show_colorbar: bool = True,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        fig = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        self._draw_orbit_loop_3d(family_result, jacobi_values, ax,
                                 start=start, end=end, step=step)

        if show_bodies:
            self.plot_primary_bodies(ax=ax, is_3d=True)
        if show_libration:
            self.plot_libration_points(ax=ax, show_labels=True, is_3d=True)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        self._style_3d_ax(ax)
        ax.view_init(elev=elev, azim=azim)

        if show_colorbar:
            self._add_colorbar(ax, jacobi_values, shrink=0.6, pad=0.1)

        if title:
            ax.set_title(title, fontsize=self.config.title,
                         y=self.config.title_y_offset_3d)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_jacobi_period_stability(
        self,
        jacobi_values: List[float],
        periods,
        stability_values: List[float],
        title: str = "",
        target_period: Optional[float] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        fig, ax1 = plt.subplots(figsize=self.config.figsize_dual, dpi=self.config.dpi)

        sorted_indices = sorted(range(len(jacobi_values)), key=lambda i: jacobi_values[i])
        j_sorted = [jacobi_values[i] for i in sorted_indices]
        p_sorted = [periods[i] for i in sorted_indices]
        s_sorted = [stability_values[i] for i in sorted_indices]

        color_period = "tab:blue"
        ax1.set_xlabel("Jacobi Constant", fontsize=self.config.label)
        ax1.set_ylabel("Period (nondimensional)", color=color_period,
                        fontsize=self.config.label)
        (line_period,) = ax1.plot(
            j_sorted, p_sorted, "-", color=color_period,
            linewidth=2, label="Period")
        ax1.tick_params(axis="y", labelcolor=color_period, labelsize=self.config.tick)
        ax1.tick_params(axis="x", labelsize=self.config.tick)

        if target_period is not None:
            ax1.axhline(y=target_period, color="green", linestyle="--",
                        linewidth=1.5, label=f"Target T={target_period:.3f}")

        ax2 = ax1.twinx()
        color_stability = "tab:red"
        ax2.set_ylabel("Stability Index (λmax)", color=color_stability,
                        fontsize=self.config.label)
        (line_stability,) = ax2.plot(
            j_sorted, s_sorted, "-", color=color_stability,
            linewidth=2, label="Stability Index (λmax)")
        ax2.tick_params(axis="y", labelcolor=color_stability, labelsize=self.config.tick)

        lines = [line_period, line_stability]
        labels_str = [str(l.get_label()) for l in lines]
        if target_period is not None:
            lines.append(ax1.get_lines()[-1])
            labels_str.append(f"Target T={target_period:.3f}")
        ax1.legend(lines, labels_str, loc="upper right", fontsize=self.config.legend)

        if title:
            ax1.set_title(title, fontsize=self.config.title,
                          y=self.config.title_y_offset_dual)

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
        jacobi_values: List[float],
        periods,
        stability_values: List[float],
        suptitle: str = "",
        plane: str = "xy",
        center_3d: Tuple[float, float, float] = (0.5, 0.0, 0.0),
        radius_3d: float = 0.65,
        zoom_xlim=None,
        zoom_ylim=None,
        elev: int = 0,
        azim: int = -90,
        target_period: Optional[float] = None,
        step: int = 1,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        n_orbits = len(family_result)
        fig = plt.figure(figsize=self.config.figsize_overview, dpi=self.config.dpi)
        fs = self.config

        # Subplot 1: Global 2D
        ax1 = fig.add_subplot(221)
        self._draw_orbit_loop_2d(family_result, jacobi_values, ax1, plane=plane, step=step)
        self.plot_primary_bodies(ax=ax1)
        self.plot_libration_points(ax=ax1)
        self._add_colorbar(ax1, jacobi_values)
        ax1.set_title(f"Global {plane.upper()} View ({n_orbits} orbits)",
                       fontsize=fs.title, y=fs.title_y_offset_subplot)
        xlabel = "X" if plane in ("xy", "xz") else "Y"
        ylabel = "Y" if plane == "xy" else "Z"
        ax1.set_xlabel(xlabel, fontsize=fs.label)
        ax1.set_ylabel(ylabel, fontsize=fs.label)
        ax1.tick_params(labelsize=fs.tick)
        ax1.set_aspect("equal")

        # Subplot 2: Zoomed 2D
        ax2 = fig.add_subplot(222)
        self._draw_orbit_loop_2d(family_result, jacobi_values, ax2, plane=plane, step=step)
        self.plot_primary_bodies(ax=ax2)
        self.plot_libration_points(ax=ax2)
        if zoom_xlim:
            ax2.set_xlim(*zoom_xlim)
        if zoom_ylim:
            ax2.set_ylim(*zoom_ylim)
        ax2.set_title(f"Zoomed {plane.upper()} View", fontsize=fs.title,
                       y=fs.title_y_offset_subplot)
        ax2.set_xlabel(xlabel, fontsize=fs.label)
        ax2.set_ylabel(ylabel, fontsize=fs.label)
        ax2.tick_params(labelsize=fs.tick)
        ax2.set_aspect("equal")

        # Subplot 3: Jacobi-Period-Stability
        ax3 = fig.add_subplot(223)
        ax3.set_xlabel("Jacobi Constant", fontsize=fs.label)
        ax3.set_ylabel("Period", color="tab:blue", fontsize=fs.label)
        (line_p,) = ax3.plot(jacobi_values, periods, "o-",
                              color="tab:blue", markersize=4)
        ax3.tick_params(axis="y", labelcolor="tab:blue", labelsize=fs.tick)
        ax3.tick_params(axis="x", labelsize=fs.tick)
        if target_period is not None:
            ax3.axhline(y=target_period, color="green", linestyle="--", linewidth=1.5)
        ax3_right = ax3.twinx()
        ax3_right.set_ylabel("λmax", color="tab:red", fontsize=fs.label)
        (line_s,) = ax3_right.plot(jacobi_values, stability_values, "s-",
                                    color="tab:red", markersize=4)
        ax3_right.tick_params(axis="y", labelcolor="tab:red", labelsize=fs.tick)
        ax3.set_title("Jacobi vs Period & Stability", fontsize=fs.title,
                       y=fs.title_y_offset_subplot)
        ax3.legend([line_p, line_s], ["Period", "λmax"],
                    loc="upper right", fontsize=fs.legend)
        ax3.grid(True, alpha=0.3)

        # Subplot 4: 3D
        ax4 = fig.add_subplot(224, projection="3d")
        self._draw_orbit_loop_3d(family_result, jacobi_values, ax4, step=step)
        self.plot_primary_bodies(ax=ax4, is_3d=True)
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
