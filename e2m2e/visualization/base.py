from __future__ import annotations

from typing import Any, Optional, Union

import matplotlib
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from enum import Enum

from ..core.system import CR3BP_System, LibrationPoint
from ..core.orbit import Orbit
from .config import PlotConfig


class ProjectionPlane(Enum):
    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class OrbitVisualizer:
    def __init__(self, system: CR3BP_System, config: Optional[PlotConfig] = None) -> None:
        self.system = system
        self.mu = system.mu
        self.config = config or PlotConfig()

        self.figure = None
        self.axes = None
        self.axes_3d = None

        self.orbit_linewidth = self.config.orbit_linewidth
        self.orbit_alpha = self.config.orbit_alpha
        self.color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_index = 0

        self.primary_body_color = self.config.primary_body_color
        self.primary_body_size = self.config.primary_body_size
        self.secondary_body_color = self.config.secondary_body_color
        self.secondary_body_size = self.config.secondary_body_size

        self.primary_body_use_marker = True
        self.secondary_body_use_marker = True
        self.primary_body_marker = "o"
        self.secondary_body_marker = "o"

        self.libration_point_colors = list(self.config.lp_colors)
        self.libration_point_markers = list(self.config.lp_markers)
        self.libration_point_sizes = list(self.config.lp_sizes)
        self.libration_point_labels = ["L1", "L2", "L3", "L4", "L5"]
        self.libration_point_fontsize = self.config.lp_label

    def _get_next_color(self) -> str:
        color = self.color_cycle[self.color_index % len(self.color_cycle)]
        self.color_index += 1
        return color

    def _extract_states(self, orbit):
        if hasattr(orbit, "states"):
            states = orbit.states
        else:
            states = np.array(orbit)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return states

    def plot_3d_orbit(
        self,
        orbit: Union[Orbit, npt.ArrayLike],
        color: Optional[str] = None,
        label: Optional[str] = None,
        ax: Optional[Any] = None,
        show_start: bool = True,
    ) -> Any:
        if ax is None:
            if self.axes_3d is None:
                self.figure = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
                self.axes_3d = self.figure.add_subplot(111, projection="3d")
            ax = self.axes_3d

        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        if color is None:
            color = self._get_next_color()

        ax.plot(x, y, z, color=color, label=label,
                linewidth=self.orbit_linewidth, alpha=self.orbit_alpha)

        if show_start and len(x) > 0:
            ax.scatter(x[0], y[0], z[0], color=color, marker="o", s=50,
                       edgecolors="black", linewidth=1)

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
        if ax is None:
            if self.axes is None:
                self.figure, self.axes = plt.subplots(
                    1, 1, figsize=self.config.figsize_2d, dpi=self.config.dpi)
            ax = self.axes

        states = self._extract_states(orbit)
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        if color is None:
            color = self._get_next_color()

        if isinstance(plane, str):
            plane = ProjectionPlane(plane)

        if plane == ProjectionPlane.XY:
            px, py = x, y
        elif plane == ProjectionPlane.XZ:
            px, py = x, z
        elif plane == ProjectionPlane.YZ:
            px, py = y, z
        else:
            raise ValueError(f"Unknown projection plane: {plane}")

        ax.plot(px, py, color=color, label=label,
                linewidth=self.orbit_linewidth, alpha=self.orbit_alpha)

        if show_start and len(px) > 0:
            ax.scatter(px[0], py[0], color=color, marker="o", s=50,
                       edgecolors="black", linewidth=1)

        return ax

    def plot_libration_points(
        self, ax: Optional[Any] = None, show_labels: bool = True, is_3d: bool = False
    ) -> Any:
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
                ax.scatter(coord[0], coord[1], coord[2], color=color,
                           marker=marker, s=size, zorder=5)
                if show_labels:
                    ax.text(coord[0], coord[1], coord[2] + 0.02, label_text,
                            fontsize=self.libration_point_fontsize, ha="center")
            else:
                ax.scatter(coord[0], coord[1], color=color, marker=marker,
                           s=size, zorder=5)
                if show_labels:
                    ax.annotate(
                        label_text, (coord[0], coord[1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=self.libration_point_fontsize,
                    )
        return ax

    def plot_primary_bodies(self, ax: Optional[Any] = None, is_3d: bool = False) -> Any:
        if self.mu is None:
            return ax
        if ax is None:
            ax = self.axes_3d if is_3d else self.axes
            if ax is None:
                return ax

        primary_name = getattr(self.system, "primary_body", None) or "Earth"
        secondary_name = getattr(self.system, "secondary_body", None) or "Moon"

        if is_3d:
            ax.scatter(-self.mu, 0, 0, color=self.primary_body_color,
                       s=self.primary_body_size, edgecolors="black", linewidth=1,
                       zorder=10, label=primary_name)
            ax.scatter(1 - self.mu, 0, 0, color=self.secondary_body_color,
                       s=self.secondary_body_size, edgecolors="black", linewidth=1,
                       zorder=10, label=secondary_name)
        else:
            primary_pos = np.array([-self.mu, 0])
            secondary_pos = np.array([1 - self.mu, 0])
            ax.scatter(*primary_pos, color="#2E86AB", s=self.primary_body_size,
                       edgecolors="#1A5276", linewidth=1.5, zorder=10, label=primary_name)
            ax.scatter(*secondary_pos, color="#95A5A6", s=self.secondary_body_size,
                       edgecolors="#566573", linewidth=1.5, zorder=10, label=secondary_name)
        return ax

    def show(self) -> None:
        plt.show()

    def save(self, filename: str, dpi: Optional[int] = None) -> None:
        if self.figure is not None:
            self.figure.savefig(filename, dpi=dpi or self.config.dpi,
                                bbox_inches="tight", pad_inches=0.1)

    def _sort_points_by_nearest_neighbor(self, x, y):
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
        points = np.column_stack((x, y, z))
        n = len(points)
        if n <= 2:
            return x, y, z
        distances_from_origin = np.sqrt(
            points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
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
