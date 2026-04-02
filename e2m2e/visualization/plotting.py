from __future__ import annotations

from .config import PlotConfig
from .base import OrbitVisualizer, ProjectionPlane
from .family import FamilyPlotter
from .transfer import TransferPlotter
from .stability import compute_stability_for_family


def configure_academic_fonts():
    import matplotlib
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["mathtext.rm"] = "serif"
    matplotlib.rcParams["mathtext.it"] = "serif:italic"
    matplotlib.rcParams["mathtext.bf"] = "serif:bold"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["legend.framealpha"] = 0.9
    matplotlib.rcParams["legend.fancybox"] = True
    matplotlib.rcParams["legend.shadow"] = False
