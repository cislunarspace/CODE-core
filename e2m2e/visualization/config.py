"""绘图配置模块

定义 PlotConfig 配置类，统一管理 matplotlib 的字体、颜色、尺寸等绘图参数。
包含自动检测系统 DPI 缩放的逻辑，确保高分辨率屏幕上的正确显示。

v4.0 MBSE 重构：从 dataclass 迁移到 Pydantic BaseModel，获得运行时验证。
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_STANDARD_DPI = 96.0  # 标准 DPI，作为缩放计算的基准


def _detect_system_scale() -> float:
    """检测系统显示缩放倍数。

    按优先级依次尝试：
    1. 环境变量 MPL_SCALE（用户手动指定）
    2. 环境变量 GDK_SCALE / QT_SCALE_FACTOR（桌面环境缩放）
    3. 解析 xrandr 输出计算实际 DPI

    Returns:
        缩放倍数，1.0 表示标准 DPI，大于 1.0 表示高分辨率屏幕。
    """
    # 优先级 1：用户通过 MPL_SCALE 环境变量手动指定
    env = os.environ.get("MPL_SCALE")
    if env is not None:
        try:
            return max(1.0, float(env))
        except ValueError:
            pass

    # 优先级 2：桌面环境的缩放设置
    for var in ("GDK_SCALE", "QT_SCALE_FACTOR"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1.0, float(val))
            except ValueError:
                pass

    # 优先级 3：通过 xrandr 查询实际显示器 DPI
    try:
        r = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, timeout=3,
        )
        best_dpi = _STANDARD_DPI
        for line in r.stdout.splitlines():
            # 跳过未连接或无物理尺寸信息的行
            if " connected" not in line or "mm" not in line:
                continue
            parts = line.split()
            # 从分辨率信息中提取像素尺寸（如 "1920x1080"）
            res_token = None
            for p in parts[2:]:
                if "x" in p and any(c.isdigit() for c in p):
                    res_token = p
                    break
            if not res_token:
                continue
            # 解析像素宽度和高度
            try:
                pw_s, rest = res_token.split("x", 1)
                ph_s = rest.split("+")[0].split("-")[0]
                pw, ph = int(pw_s), int(ph_s)
            except (ValueError, IndexError):
                continue
            # 查找物理尺寸（mm 单位，如 "345mm x 194mm"）
            mm_w = mm_h = None
            for i, p in enumerate(parts):
                if (p.endswith("mm") and i + 2 < len(parts)
                        and parts[i + 1] == "x" and parts[i + 2].endswith("mm")):
                    mm_w = int(p.rstrip("mm"))
                    mm_h = int(parts[i + 2].rstrip("mm"))
                    break
            if not mm_w or not mm_h or mm_w <= 0 or mm_h <= 0:
                continue
            # 通过像素和物理尺寸计算实际 DPI
            dpi_w = pw / (mm_w / 25.4)
            dpi_h = ph / (mm_h / 25.4)
            dpi = (dpi_w + dpi_h) / 2
            if dpi > best_dpi:
                best_dpi = dpi
        # 如果实际 DPI 超过标准值 25% 以上，计算缩放倍数
        if best_dpi > _STANDARD_DPI * 1.25:
            return round(best_dpi / _STANDARD_DPI, 2)
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return 1.0


_detected_scale = _detect_system_scale()
# 检测系统缩放，若大于标准值则自动补丁 tkinter 以适配高 DPI
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


class PlotConfig(BaseModel):
    """统一绘图配置，管理字体大小、颜色、线宽、图像尺寸等参数。

    基于 Pydantic BaseModel，提供运行时类型验证。
    提供 apply_rcparams() 方法可将配置直接应用到 matplotlib 全局设置。
    支持高 DPI 屏幕的自动缩放。

    Attributes:
        title: 标题字体大小。
        label: 坐标轴标签字体大小。
        tick: 刻度标签字体大小。
        legend: 图例字体大小。
        colorbar: 颜色条标签字体大小。
        suptitle: 超标题字体大小。
        lp_label: 平动点标签字体大小。
        colormap: 颜色映射名称（如 "coolwarm"）。
        primary_body_color: 主天体标记颜色。
        primary_body_size: 主天体标记大小。
        secondary_body_color: 次天体标记颜色。
        secondary_body_size: 次天体标记大小。
        lp_colors: 平动点标记颜色列表（5个元素）。
        lp_markers: 平动点标记形状列表（5个元素）。
        lp_sizes: 平动点标记大小列表（5个元素）。
        orbit_linewidth: 轨道线条宽度。
        orbit_alpha: 轨道线条透明度。
        figsize_2d: 2D 图像尺寸 (宽, 高)。
        figsize_3d: 3D 图像尺寸 (宽, 高)。
        figsize_dual: 双图并排图像尺寸。
        figsize_overview: 概览图图像尺寸。
        dpi: 输出图像 DPI。
        title_y_offset: 标题 y 方向偏移量（避免与子图重叠）。
        auto_scale: 是否启用自动 DPI 缩放。
        scale_factor: 实际缩放倍数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 字体大小参数
    title: float = 16
    label: float = 14
    tick: float = 13
    legend: float = 14
    colorbar: float = 13
    suptitle: float = 18
    lp_label: float = 16

    # 颜色和标记参数
    colormap: str = "coolwarm"
    primary_body_color: str = "blue"
    primary_body_size: int = 200
    secondary_body_color: str = "silver"
    secondary_body_size: int = 100
    lp_colors: List[str] = Field(default_factory=lambda: ["gray"] * 5)
    lp_markers: List[str] = Field(default_factory=lambda: ["^"] * 5)
    lp_sizes: List[int] = Field(default_factory=lambda: [60] * 5)

    # 线条和图像尺寸参数
    orbit_linewidth: float = 1.5
    orbit_alpha: float = 0.8
    figsize_2d: tuple = (12, 10)
    figsize_3d: tuple = (14, 10)
    figsize_dual: tuple = (12, 7)
    figsize_overview: tuple = (18, 14)
    dpi: int = 100

    # 标题偏移参数（用于不同布局下的标题位置调整）
    title_y_offset: float = -0.12
    title_y_offset_3d: float = -0.08
    title_y_offset_dual: float = -0.18
    title_y_offset_subplot: float = -0.15

    # 缩放参数
    auto_scale: bool = True
    scale_factor: float = Field(default_factory=lambda: _detected_scale)

    def apply_rcparams(self) -> None:
        """将配置应用到 matplotlib 全局参数。

        设置字体族、数学文本字体、图例样式、字体大小等。
        在高 DPI 屏幕下会自动记录缩放信息。
        """
        import matplotlib.pyplot as plt

        # 高 DPI 屏幕自动缩放
        if self.auto_scale and self.scale_factor > 1.01:
            logger.info("auto_scale=%.2fx (tk scaling applied)", self.scale_factor)

        # 设置全局字体：优先 Times New Roman，数学文本使用 STIX 字体
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["mathtext.rm"] = "serif"
        matplotlib.rcParams["mathtext.it"] = "serif:italic"
        matplotlib.rcParams["mathtext.bf"] = "serif:bold"
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 图例样式：带边框、半透明背景、无阴影（学术论文标准样式）
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
        """获取配置指定的颜色映射对象。"""
        return matplotlib.colormaps[self.colormap]
