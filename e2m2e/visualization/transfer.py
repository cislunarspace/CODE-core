from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .base import OrbitVisualizer
from .config import PlotConfig
from ..core.orbit import Orbit
from ..core.system import CR3BP_System


class TransferPlotter(OrbitVisualizer):
    def __init__(self, system: CR3BP_System, config: Optional[PlotConfig] = None) -> None:
        super().__init__(system, config)

    def plot_solution_plane(
        self,
        results,
        color_by: Optional[str] = None,
        ax: Optional[Any] = None,
        show_colorbar: bool = True,
    ) -> Any:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.config.dpi)

        parsed = self._parse_solution_results(results)
        valid = [r for r in parsed if r["success"]]
        if not valid:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            ax.set_xlabel("Transfer Time (T)")
            ax.set_ylabel(r"Total $\Delta v$")
            return ax

        times = np.array([r["transfer_time"] for r in valid])
        dvs = np.array([r["delta_v1"] + r["delta_v2"] for r in valid])

        type_colors = {
            "direct": "#1f77b4",
            "lga": "#ff7f0e",
            "external": "#2ca02c",
        }

        if color_by == "transfer_type":
            for ttype, color in type_colors.items():
                mask = np.array(
                    [str(r.get("transfer_type", "")).lower() == ttype for r in valid])
                if mask.any():
                    ax.scatter(times[mask], dvs[mask], c=color, s=10, alpha=0.7,
                               label=ttype.upper())
            ax.legend()
        else:
            ax.scatter(times, dvs, s=10, alpha=0.7)

        ax.set_xlabel("Transfer Time (T)")
        ax.set_ylabel(r"Total $\Delta v$")
        ax.grid(True, alpha=0.3, linestyle="--")
        return ax

    def plot_transfer_orbit(
        self,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        transfer_trajectory: npt.ArrayLike,
        departure_state: npt.ArrayLike,
        insertion_state: npt.ArrayLike,
        ax: Optional[Any] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Any:
        transfer_states = np.asarray(transfer_trajectory)
        dep_states = self._extract_states(departure_orbit)
        arr_states = self._extract_states(arrival_orbit)

        if ax is None:
            self.figure = plt.figure(figsize=self.config.figsize_3d, dpi=self.config.dpi)
            ax = self.figure.add_subplot(111, projection="3d")
            self.axes_3d = ax

        if color is None:
            color = self._get_next_color()

        ax.plot(dep_states[:, 0], dep_states[:, 1], dep_states[:, 2],
                color="steelblue", linewidth=self.orbit_linewidth,
                alpha=self.orbit_alpha, label="DRO")
        ax.plot(arr_states[:, 0], arr_states[:, 1], arr_states[:, 2],
                color="darkorange", linewidth=self.orbit_linewidth,
                alpha=self.orbit_alpha, label="RO")
        ax.plot(transfer_states[:, 0], transfer_states[:, 1], transfer_states[:, 2],
                color=color, linewidth=2.0, alpha=0.9, label=label)

        dep = np.asarray(departure_state)
        ins = np.asarray(insertion_state)
        ax.scatter(dep[0], dep[1], dep[2], color="green", marker="^", s=80,
                   edgecolors="black", linewidth=1, zorder=10, label="Departure")
        ax.scatter(ins[0], ins[1], ins[2], color="red", marker="v", s=80,
                   edgecolors="black", linewidth=1, zorder=10, label="Insertion")

        self.plot_primary_bodies(ax=ax, is_3d=True)
        self.plot_libration_points(ax=ax, is_3d=True)

        ax.set_xlabel("X (nondimensional)")
        ax.set_ylabel("Y (nondimensional)")
        ax.set_zlabel("Z (nondimensional)")
        ax.legend()
        return ax

    def _parse_solution_results(self, results) -> list:
        if not results:
            return []
        parsed = []
        for r in results:
            if isinstance(r, dict):
                parsed.append(r)
            else:
                parsed.append({
                    "transfer_time": r.transfer_time,
                    "delta_v1": r.delta_v1,
                    "delta_v2": r.delta_v2,
                    "objective_value": r.objective_value,
                    "success": r.success,
                    "transfer_type": r.transfer_type.value
                    if hasattr(r.transfer_type, "value") else str(r.transfer_type),
                })
        return parsed
