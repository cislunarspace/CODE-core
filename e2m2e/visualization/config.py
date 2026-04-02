from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib


@dataclass
class PlotConfig:
    title: float = 16
    label: float = 14
    tick: float = 13
    legend: float = 14
    colorbar: float = 13
    suptitle: float = 18
    lp_label: float = 16

    colormap: str = "coolwarm"
    primary_body_color: str = "blue"
    primary_body_size: int = 200
    secondary_body_color: str = "silver"
    secondary_body_size: int = 100
    lp_colors: List[str] = field(default_factory=lambda: ["gray"] * 5)
    lp_markers: List[str] = field(default_factory=lambda: ["^"] * 5)
    lp_sizes: List[int] = field(default_factory=lambda: [60] * 5)

    orbit_linewidth: float = 1.5
    orbit_alpha: float = 0.8
    figsize_2d: tuple = (12, 10)
    figsize_3d: tuple = (14, 10)
    figsize_dual: tuple = (12, 7)
    figsize_overview: tuple = (18, 14)
    dpi: int = 100

    title_y_offset: float = -0.12
    title_y_offset_3d: float = -0.08
    title_y_offset_dual: float = -0.18
    title_y_offset_subplot: float = -0.15

    def apply_rcparams(self) -> None:
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["mathtext.rm"] = "serif"
        matplotlib.rcParams["mathtext.it"] = "serif:italic"
        matplotlib.rcParams["mathtext.bf"] = "serif:bold"
        matplotlib.rcParams["axes.unicode_minus"] = False

        matplotlib.rcParams["legend.frameon"] = True
        matplotlib.rcParams["legend.framealpha"] = 0.9
        matplotlib.rcParams["legend.fancybox"] = True
        matplotlib.rcParams["legend.shadow"] = False

        plt.rcParams.update({
            "font.size": self.tick,
            "axes.titlesize": self.title,
            "axes.labelsize": self.label,
            "xtick.labelsize": self.tick,
            "ytick.labelsize": self.tick,
            "legend.fontsize": self.legend,
        })

    def get_cmap(self):
        return matplotlib.colormaps[self.colormap]
