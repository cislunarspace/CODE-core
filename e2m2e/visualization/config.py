"""绘图配置模块

定义 PlotConfig 配置类，统一管理 matplotlib 的字体、颜色、尺寸等绘图参数。
高 DPI 缩放适配逻辑封装在 :func:`configure_dpi_scaling` 中，import 本模块
时不执行任何副作用；交互式绘图场景下需由调用方显式调用以适配高分辨率屏幕。
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_STANDARD_DPI = 96.0  # 标准 DPI，作为缩放计算的基准

# 环境变量名：天体图标缩放系数（地球、月球同步缩放）。
# 调用方通过设置此环境变量影响 PlotConfig.from_env() 构造的实例。
BODY_ICON_SCALE_ENV = "E2M2E_BODY_ICON_SCALE"

# 环境变量名：天体图标目录。允许用户在不修改代码的情况下切换图标位置。
# 调用方通过设置此环境变量影响 PlotConfig.from_env() 构造的实例。
BODY_ICON_PATH_ENV = "E2M2E_BODY_ICON_PATH"


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
            logger.debug("Invalid MPL_SCALE value: %s", env)

    # 优先级 2：桌面环境的缩放设置
    for var in ("GDK_SCALE", "QT_SCALE_FACTOR"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1.0, float(val))
            except ValueError:
                logger.debug("Invalid %s value: %s", var, val)

    # 优先级 3：通过 xrandr 查询实际显示器 DPI
    try:
        r = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            timeout=3,
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
                if (
                    p.endswith("mm")
                    and i + 2 < len(parts)
                    and parts[i + 1] == "x"
                    and parts[i + 2].endswith("mm")
                ):
                    mm_w = int(p.removesuffix("mm"))
                    mm_h = int(parts[i + 2].removesuffix("mm"))
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
        logger.debug("xrandr not found, skipping DPI detection")
    except Exception:
        logger.debug("xrandr query failed", exc_info=True)

    return 1.0


# import 时不执行任何副作用：_detected_scale 默认为 1.0（标准 DPI），
# 由 configure_dpi_scaling() 在需要高 DPI 适配时显式检测并更新。
_detected_scale = 1.0
_dpi_configured = False


def configure_dpi_scaling() -> float:
    """显式启用高 DPI 缩放适配（opt-in）。

    本模块 import 时不执行任何副作用（不 fork xrandr、不修改环境变量、
    不打补丁）。需要在交互式绘图场景下适配高分辨率屏幕时，由调用方
    显式调用本函数。本函数会：

    1. 调用 :func:`_detect_system_scale` 检测系统显示缩放（可能 fork
       ``xrandr`` 子进程）。
    2. 若缩放大于 1.01：设置 ``TK_SCALE`` 环境变量，对
       ``tkinter.Tk``/``tkinter.Toplevel`` 的 ``__init__`` 打补丁以应用
       tk scaling，并在 ``zenity`` 可用时用它替换
       ``tkinter.filedialog`` 的 ``askopenfilename``/``asksaveasfilename``。

    本函数幂等：重复调用直接返回已检测的缩放值，不会重复打补丁。

    Returns:
        检测到的系统缩放倍数（1.0 表示标准 DPI）。返回值同时写入
        模块级 ``_detected_scale``，供 :class:`PlotConfig` 的
        ``scale_factor`` 字段默认值使用。
    """
    global _detected_scale, _dpi_configured
    if _dpi_configured:
        return _detected_scale
    _dpi_configured = True

    scale = _detect_system_scale()
    _detected_scale = scale

    # 检测系统缩放，若大于标准值则自动补丁 tkinter 以适配高 DPI
    if scale > 1.01:
        os.environ.setdefault("TK_SCALE", str(scale))
        import shutil as _shutil
        import tkinter as _tk

        # tkinter scaling 使用 point 为单位（1 point = 1/72 inch），
        # 而系统 DPI 基于每英寸像素数（1 inch = 96 px 在标准 DPI 下），
        # 因此缩放倍数需乘以 96/72 将 DPI 比率转换为 point 缩放比率
        _tk_scaling_val = scale * 96.0 / 72.0
        _orig_tk_init = _tk.Tk.__init__
        _orig_toplevel_init = _tk.Toplevel.__init__

        def _patched_tk_init(self, *args, **kwargs):
            _orig_tk_init(self, *args, **kwargs)
            with contextlib.suppress(Exception):
                self.tk.call("tk", "scaling", _tk_scaling_val)

        def _patched_toplevel_init(self, *args, **kwargs):
            _orig_toplevel_init(self, *args, **kwargs)
            with contextlib.suppress(Exception):
                self.tk.call("tk", "scaling", _tk_scaling_val)

        _tk.Tk.__init__ = _patched_tk_init  # type: ignore[method-assign]
        _tk.Toplevel.__init__ = _patched_toplevel_init  # type: ignore[method-assign]

        # 在高 DPI 环境下，tkinter 原生文件对话框无法跟随缩放，
        # 会出现极小的窗口。用 zenity（Linux 桌面原生对话框）替代，
        # zenity 由 GTK 渲染，自动适配系统缩放设置
        if _shutil.which("zenity"):
            import tkinter.filedialog as _fd

            def _zenity_save(
                title="Save file",
                initialdir=None,
                initialfile=None,
                filetypes=None,
                defaultextension=None,
                **kwargs,
            ):
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
                    logger.debug("zenity file dialog failed", exc_info=True)
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
                    logger.debug("zenity file dialog failed", exc_info=True)
                return ""

            _fd.asksaveasfilename = _zenity_save
            _fd.askopenfilename = _zenity_open

    return scale


import matplotlib  # noqa: E402


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
        icon_path: 天体图标目录，None 时由 ``icons.resolve_icon_dir`` 回退。
        primary_body_icon: 主天体图标文件名。
        secondary_body_icon: 次天体图标文件名。
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
        title_y_offset: 标题 y 方向偏移量（2D 标准图）。
        title_y_offset_3d: 标题 y 方向偏移量（3D 图，因 3D 轴标签占位不同）。
        title_y_offset_dual: 标题 y 方向偏移量（双 Y 轴图，留出右轴标签空间）。
        title_y_offset_subplot: 标题 y 方向偏移量（子图布局，避免与相邻子图重叠）。
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
    # 天体图标缩放系数（用于 PNG 图标显示大小微调，不影响散点回退）
    primary_body_icon_scale: float = 1.0
    secondary_body_icon_scale: float = 1.0
    # 天体图标目录：None 时由 icons.resolve_icon_dir() 按以下优先级回退——
    # 环境变量 E2M2E_BODY_ICON_PATH → ~/Downloads（向后兼容默认）。
    # 支持 ~、${VAR}/$VAR 占位符、相对/绝对路径。
    icon_path: str | None = None
    # 主天体/次天体图标文件名（相对于 icon_path）
    primary_body_icon: str = "地球.png"
    secondary_body_icon: str = "月球.png"
    lp_colors: list[str] = Field(default_factory=lambda: ["gray"] * 5)
    lp_markers: list[str] = Field(default_factory=lambda: ["^"] * 5)
    lp_sizes: list[int] = Field(default_factory=lambda: [60] * 5)

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

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **overrides,
    ) -> PlotConfig:
        """从环境变量构造 PlotConfig，overrides 优先级最高。

        当前支持的环境变量：

        - ``BODY_ICON_SCALE_ENV`` (``E2M2E_BODY_ICON_SCALE``)：浮点数，同时
          应用于 ``primary_body_icon_scale`` 和 ``secondary_body_icon_scale``。
          解析失败（非数字、≤ 0）时静默回退到字段默认值，不抛异常。
        - ``BODY_ICON_PATH_ENV`` (``E2M2E_BODY_ICON_PATH``)：天体图标目录，
          写入 ``icon_path`` 字段。空字符串或缺失时静默忽略。

        Args:
            env: 环境变量映射，``None`` 时使用 ``os.environ``。便于测试注入。
            **overrides: 显式字段覆盖，优先级高于环境变量。

        Returns:
            构造好的 PlotConfig 实例。
        """
        source = os.environ if env is None else env

        env_kwargs: dict[str, Any] = {}

        raw_path = source.get(BODY_ICON_PATH_ENV)
        if raw_path:
            env_kwargs["icon_path"] = raw_path

        raw = source.get(BODY_ICON_SCALE_ENV)
        if raw is not None:
            try:
                value = float(raw)
            except ValueError:
                logger.debug("Invalid %s value: %r (expected float)", BODY_ICON_SCALE_ENV, raw)
            else:
                if value > 0:
                    env_kwargs["primary_body_icon_scale"] = value
                    env_kwargs["secondary_body_icon_scale"] = value
                else:
                    logger.debug(
                        "Non-positive %s value: %r (expected > 0)",
                        BODY_ICON_SCALE_ENV,
                        raw,
                    )

        # overrides 覆盖 env 值
        merged = {**env_kwargs, **overrides}
        return cls(**merged)

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

        plt.rcParams.update(
            {
                "font.size": self.tick,
                "axes.titlesize": self.title,
                "axes.labelsize": self.label,
                "xtick.labelsize": self.tick,
                "ytick.labelsize": self.tick,
                "legend.fontsize": self.legend,
            }
        )

    def get_cmap(self):
        """获取配置指定的颜色映射对象。

        Returns:
            matplotlib.colors.Colormap 颜色映射实例。
        """
        return matplotlib.colormaps[self.colormap]
