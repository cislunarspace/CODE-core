"""SPICE 内核管理 shim（ADR 0011 迁移）。

``SPICEManager`` 完整实现已迁至 ``e2m2e.data.kernels.manager``（数据层，
加载/缓存/校验、时间转换、状态查询、帧查询），旧路径保持可用。

内部符号（``_GM_VALUES``/``_NAIF_IDS``/``_LEAPSECOND_SEARCH_PATHS``/
``_find_leapseconds_kernel``）一并 re-export，供既有测试与下游引用。
"""

from __future__ import annotations

from e2m2e.data.kernels.manager import (
    _GM_VALUES,
    _LEAPSECOND_SEARCH_PATHS,
    _NAIF_IDS,
    SPICEManager,
)
from e2m2e.data.kernels.manager import (
    _find_leapseconds_kernel as _find_leapseconds_kernel_impl,
)

__all__ = [
    "SPICEManager",
    "_GM_VALUES",
    "_NAIF_IDS",
    "_LEAPSECOND_SEARCH_PATHS",
    "_find_leapseconds_kernel",
]


def _find_leapseconds_kernel():
    """在 shim 模块的搜索路径下查找闰秒内核（patch 友好包装）。"""
    return _find_leapseconds_kernel_impl(_LEAPSECOND_SEARCH_PATHS)
