"""SPICE 帧旋转查询。

数据层（ADR 0011 迁移，源：``core/spice.py`` 帧查询部分 / 既有
``standard_axes`` 中的 ``pxform`` 用法）。SPICE 帧定义（PCK/FK）由
``SPICEManager`` 加载内核提供；本模块只做帧旋转矩阵查询，转换**算法**
在 ``algorithm/coordinate/``。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..kernels._spice_loader import get_spiceypy


def frame_rotation(frame_from: str, frame_to: str, et: float) -> npt.NDArray[np.floating]:
    """SPICE 帧旋转矩阵（3×3）：``frame_from`` → ``frame_to`` 在 ET 时刻。"""
    return np.array(get_spiceypy().pxform(frame_from, frame_to, et))
