"""SPICE 内核管理器：加载/缓存/校验。

实现状态：骨架。完整实现待从 ``core/spice.py`` 的 ``SPICEManager`` 迁入，
并对上层暴露 ``EphemerisProvider`` 接口（见 ``provider.py``）。
"""

from __future__ import annotations

__all__ = ["SPICEManager"]


class SPICEManager:
    """SPICE 内核管理器。

    实现状态：待迁入（源 ``core/spice.py``）。职责：内核加载/卸载/缓存、UTC↔ET
    时间转换、天体状态查询、帧旋转查询。
    """

    def __init__(self) -> None:  # pragma: no cover - 骨架
        raise NotImplementedError("SPICEManager 待从 core/spice.py 迁入")
