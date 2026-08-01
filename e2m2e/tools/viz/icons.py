"""天体图标加载与渲染辅助模块

封装 PNG 图标加载、2D AnnotationBbox 包装、3D Billboard 深度驱动 Patch 等
可视化辅助逻辑。原位于 ``base.py``，独立为模块便于复用与单元测试。

约定：所有路径解析走 :func:`resolve_icon_dir`，避免在业务代码中硬编码
``~/Downloads``。优先级：显式参数 → 环境变量 ``E2M2E_BODY_ICON_PATH`` →
``~/Downloads``（向后兼容默认）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# 环境变量名：天体图标目录。允许用户在不修改代码的情况下切换图标位置。
BODY_ICON_PATH_ENV = "E2M2E_BODY_ICON_PATH"


def resolve_icon_dir(
    icon_path: str | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """解析天体图标目录。

    优先级：
    1. 显式参数 ``icon_path``（支持 ``~``、``${VAR}``/``$VAR``、相对/绝对路径）
    2. 环境变量 ``E2M2E_BODY_ICON_PATH``
    3. ``~/Downloads``（向后兼容默认）

    Args:
        icon_path: 用户指定的图标目录，未指定时走回退链。
        env: 环境变量字典，``None`` 时使用 ``os.environ``。便于测试注入。

    Returns:
        解析后的绝对路径 ``Path`` 对象。
    """
    source_env = os.environ if env is None else env

    raw = icon_path if icon_path is not None else source_env.get(BODY_ICON_PATH_ENV)
    if raw is None or raw == "":
        return Path.home() / "Downloads"

    # 展开 ~ 与 ${VAR}/$VAR 占位符
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return Path(expanded).expanduser()


def load_body_icons(
    icon_dir: Path | str,
    primary_name: str,
    secondary_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """从 ``icon_dir`` 加载主、次天体 PNG 图标为 RGBA numpy 数组。

    任一文件缺失或 PIL 不可用时，对应槽位返回 ``None``，不抛异常。

    Args:
        icon_dir: 图标目录。
        primary_name: 主天体图标文件名（如 "地球.png"）。
        secondary_name: 次天体图标文件名（如 "月球.png"）。

    Returns:
        ``(primary_image, secondary_image)`` 元组，缺失或加载失败时对应位置返回 ``None``。
    """
    primary: np.ndarray | None = None
    secondary: np.ndarray | None = None

    try:
        from PIL import Image
    except ImportError:
        logger.debug("PIL 未安装，无法加载天体图标")
        return primary, secondary

    base = Path(icon_dir)

    primary_path = base / primary_name
    if primary_path.exists():
        try:
            img = Image.open(primary_path).convert("RGBA")
            primary = np.array(img)
            logger.debug("已加载主天体图标: %s", primary_path)
        except Exception as e:
            logger.debug("加载主天体图标失败: %s", e)
    else:
        logger.debug("主天体图标不存在: %s", primary_path)

    secondary_path = base / secondary_name
    if secondary_path.exists():
        try:
            img = Image.open(secondary_path).convert("RGBA")
            secondary = np.array(img)
            logger.debug("已加载次天体图标: %s", secondary_path)
        except Exception as e:
            logger.debug("加载次天体图标失败: %s", e)
    else:
        logger.debug("次天体图标不存在: %s", secondary_path)

    return primary, secondary


def make_offset_image(image: np.ndarray, size: int) -> Any:
    """根据 ``image`` 和目标像素 ``size`` 构造 matplotlib ``OffsetImage``。

    Args:
        image: 已加载的 RGBA numpy 数组。
        size: 目标像素大小。

    Returns:
        ``OffsetImage`` 实例。``dpi_cor=False`` 避免保存时根据 dpi 自动放大。
    """
    from matplotlib.offsetbox import OffsetImage

    orig_size = max(image.shape[0], image.shape[1])
    zoom = size / orig_size if orig_size > 0 else 1.0
    return OffsetImage(image, zoom=zoom, dpi_cor=False)


class _DepthDriverPatch(mpatches.Patch):
    """利用 Axes3D 的 do_3d_projection 钩子驱动 Billboard 图标的深度排序。

    Axes3D.draw() 在渲染前会对所有可见 Collection 和 Patch 调用
    do_3d_projection()，这是唯一能在每帧渲染前获取到正确投影矩阵 M 的时机。
    本 Patch 利用这个钩子来：

    1. 更新 AnnotationBbox 的投影位置（跟随视角变化）。
    2. 根据图标与场景中 Line3D 的深度比较动态调整 AnnotationBbox 的 zorder。

    这比 draw_event 方案更可靠——后者在渲染之后才触发，导致 zorder 更新延迟一帧，
    旋转时出现遮挡关系闪烁。本方案在渲染前同步更新，消除延迟。
    """

    def __init__(self, annotation_box: Any, position_3d: tuple[float, float, float]) -> None:
        super().__init__(
            visible=True,
            fill=False,
            facecolor="none",
            edgecolor="none",
            linewidth=0,
        )
        self._ab = annotation_box
        self._pos = position_3d
        self._last_zorder: int = 10

    def get_path(self) -> Any:
        from matplotlib.path import Path

        return Path(np.empty((0, 2)))

    def draw(self, renderer: Any) -> None:
        pass

    def do_3d_projection(self) -> float:
        from mpl_toolkits.mplot3d import proj3d  # type: ignore[import-untyped]

        axes = self.axes
        if axes is None:
            return 0.0
        M = getattr(axes, "M", None)
        if M is None:
            return 0.0

        x3, y3, z3 = self._pos
        x2, y2, z2 = proj3d.proj_transform(x3, y3, z3, M)
        self._ab.xy = (x2, y2)
        self._ab.xybox = (x2, y2)

        line_zs = []
        for line in axes.lines:
            verts = getattr(line, "_verts3d", None)
            if verts is None or not line.get_visible():
                continue
            xs3d, ys3d, zs3d = verts
            if len(xs3d) == 0:
                continue
            _, _, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, M)
            line_zs.append(zs)

        if not line_zs:
            self._ab.set_zorder(10)
            return z2

        all_zs = np.concatenate(line_zs)
        if all_zs.size == 0:
            self._ab.set_zorder(10)
            return z2

        # proj_z 越小越靠近相机；与中位数比较决定遮挡关系
        median_z = np.median(all_zs)
        z_range = all_zs.max() - all_zs.min()
        margin = z_range * 0.1

        if z2 < median_z - margin:
            new_zorder = 10
        elif z2 > median_z + margin:
            new_zorder = 1
        else:
            new_zorder = self._last_zorder

        self._last_zorder = new_zorder
        self._ab.set_zorder(new_zorder)
        return z2


def add_3d_billboard_icon(
    ax: Any,
    offset_img: Any,
    position: tuple[float, float, float],
    label: str,
) -> None:
    """在 3D Axes 上以 Billboard 方式渲染 PNG 图标，支持动态深度遮挡。

    matplotlib 3D 的 AnnotationBbox 是 2D 元素，不参与自动深度排序。
    通过 :class:`_DepthDriverPatch` 挂接到 Axes3D.draw() 的
    do_3d_projection 钩子，在每帧渲染**之前**同步更新图标位置和 zorder，
    确保旋转交互时遮挡关系无延迟地反映空间深度。

    Args:
        ax: 3D axes 对象。
        offset_img: 已经构造好的 ``OffsetImage``。
        position: 天体的 (x, y, z) 旋转系坐标。
        label: 图例标签。
    """
    from matplotlib.offsetbox import AnnotationBbox
    from mpl_toolkits.mplot3d import proj3d  # type: ignore[import-untyped]

    x3, y3, z3 = position

    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    ab = AnnotationBbox(
        offset_img,
        (x2, y2),
        xycoords="data",
        frameon=False,
        pad=0.0,
        annotation_clip=False,
        zorder=10,
    )
    ab.set_clip_on(False)
    ax.add_artist(ab)

    # 深度驱动：不可见 Patch，通过 do_3d_projection 钩子
    # 在每帧渲染前同步更新 AnnotationBbox 的位置和 zorder
    driver = _DepthDriverPatch(ab, position)
    ax.add_patch(driver)

    # 图例占位（invisible scatter），与 2D 路径一致
    ax.scatter([], [], [], color="white", label=label)


def add_2d_icon(
    ax: Any,
    offset_img: Any,
    position: tuple[float, float],
    label: str,
) -> None:
    """在 2D Axes 上以 ``AnnotationBbox`` 方式渲染 PNG 图标。

    同时添加 invisible scatter 以填充图例条目，与 3D 路径的图例占位策略一致。

    Args:
        ax: 2D axes 对象。
        offset_img: 已经构造好的 ``OffsetImage``。
        position: 天体的 (x, y) 旋转系坐标。
        label: 图例标签。
    """
    from matplotlib.offsetbox import AnnotationBbox

    # 先添加图例条目（用 invisible scatter）
    ax.scatter([], [], color="white", label=label)

    ab = AnnotationBbox(
        offset_img,
        (float(position[0]), float(position[1])),
        frameon=False,
        zorder=10,
    )
    ax.add_artist(ab)
