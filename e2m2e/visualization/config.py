from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_STANDARD_DPI = 96.0


def _detect_system_scale() -> float:
    env = os.environ.get("MPL_SCALE")
    if env is not None:
        try:
            return max(1.0, float(env))
        except ValueError:
            pass

    for var in ("GDK_SCALE", "QT_SCALE_FACTOR"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1.0, float(val))
            except ValueError:
                pass

    try:
        r = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, timeout=3,
        )
        best_dpi = _STANDARD_DPI
        for line in r.stdout.splitlines():
            if " connected" not in line or "mm" not in line:
                continue
            parts = line.split()
            res_token = None
            for p in parts[2:]:
                if "x" in p and any(c.isdigit() for c in p):
                    res_token = p
                    break
            if not res_token:
                continue
            try:
                pw_s, rest = res_token.split("x", 1)
                ph_s = rest.split("+")[0].split("-")[0]
                pw, ph = int(pw_s), int(ph_s)
            except (ValueError, IndexError):
                continue
            mm_w = mm_h = None
            for i, p in enumerate(parts):
                if (p.endswith("mm") and i + 2 < len(parts)
                        and parts[i + 1] == "x" and parts[i + 2].endswith("mm")):
                    mm_w = int(p.rstrip("mm"))
                    mm_h = int(parts[i + 2].rstrip("mm"))
                    break
            if not mm_w or not mm_h or mm_w <= 0 or mm_h <= 0:
                continue
            dpi_w = pw / (mm_w / 25.4)
            dpi_h = ph / (mm_h / 25.4)
            dpi = (dpi_w + dpi_h) / 2
            if dpi > best_dpi:
                best_dpi = dpi
        if best_dpi > _STANDARD_DPI * 1.25:
            return round(best_dpi / _STANDARD_DPI, 2)
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return 1.0


_detected_scale = _detect_system_scale()
if _detected_scale > 1.01:
    os.environ.setdefault("TK_SCALE", str(_detected_scale))
    import tkinter as _tk
    import shutil as _shutil
    _tk_scaling_val = _detected_scale * 96.0 / 72.0
    _orig_tk_init = _tk.Tk.__init__
    _orig_toplevel_init = _tk.Toplevel.__init__

    def _patched_tk_init(self, *args, **kwargs):
        _orig_tk_init(self, *args, **kwargs)
        try:
            self.tk.call("tk", "scaling", _tk_scaling_val)
        except Exception:
            pass

    def _patched_toplevel_init(self, *args, **kwargs):
        _orig_toplevel_init(self, *args, **kwargs)
        try:
            self.tk.call("tk", "scaling", _tk_scaling_val)
        except Exception:
            pass

    _tk.Tk.__init__ = _patched_tk_init
    _tk.Toplevel.__init__ = _patched_toplevel_init

    if _shutil.which("zenity"):
        import tkinter.filedialog as _fd
        _orig_askopen = _fd.askopenfilename
        _orig_asksave = _fd.asksaveasfilename

        def _zenity_save(title="Save file", initialdir=None, initialfile=None,
                         filetypes=None, defaultextension=None, **kwargs):
            cmd = ["zenity", "--file-selection", "--save", "--confirm-overwrite"]
            if title:
                cmd.extend(["--title", title])
            if initialfile:
                import pathlib as _p
                d = _p.Path(initialdir) / initialfile if initialdir else _p.Path(initialfile)
                cmd.extend(["--filename", str(d)])
            elif initialdir:
                import pathlib as _p
                cmd.extend(["--filename", str(_p.Path(initialdir) / "")])
            if filetypes:
                for name, patterns in filetypes:
                    for pat in patterns.split():
                        cmd.extend(["--file-filter", f"{name} | {pat}"])
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if r.returncode == 0:
                    return r.stdout.strip()
            except Exception:
                pass
            return ""

        def _zenity_open(title="Open file", initialdir=None, filetypes=None, **kwargs):
            cmd = ["zenity", "--file-selection"]
            if title:
                cmd.extend(["--title", title])
            if initialdir:
                import pathlib as _p
                cmd.extend(["--filename", str(_p.Path(initialdir) / "")])
            if filetypes:
                for name, patterns in filetypes:
                    for pat in patterns.split():
                        cmd.extend(["--file-filter", f"{name} | {pat}"])
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if r.returncode == 0:
                    return r.stdout.strip()
            except Exception:
                pass
            return ""

        _fd.asksaveasfilename = _zenity_save
        _fd.askopenfilename = _zenity_open

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

    auto_scale: bool = True
    scale_factor: float = field(default_factory=lambda: _detected_scale)

    def apply_rcparams(self) -> None:
        import matplotlib.pyplot as plt

        if self.auto_scale and self.scale_factor > 1.01:
            logger.info("auto_scale=%.2fx (tk scaling applied)", self.scale_factor)

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
