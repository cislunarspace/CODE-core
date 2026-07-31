"""SPICE 内核管理与星历数据提供者抽象。

实现状态：骨架。``SPICEManager`` 待从 ``core/spice.py`` 迁入，``EphemerisProvider``
待从现有 ``SPICEManager`` 接口化。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["EphemerisProvider", "SPICEManager"]


class EphemerisProvider:
    """星历数据提供者：对上层屏蔽数据来源（SPICE/r2s2/解析）。

    单点 + 批量两类方法；时间（utc_to_tdb/et_to_utc/utc_to_tai/tai_to_tt/tt_to_tdb/
    jd_tdb_to_et）、状态（body_position/body_state/body_rotation）、帧（pxform）三类。

    实现状态：接口待定稿。SPICE 实现 = 现有 ``SPICEManager`` 接口化；r2s2 实现 =
    ``frames/r2s2.py`` 适配器。
    """

    def __init__(self) -> None:  # pragma: no cover - 骨架
        raise NotImplementedError(
            "EphemerisProvider 实现未完成（待从 SPICEManager 接口化）"
        )

    def utc_to_tdb(self, utc: str) -> float:
        """UTC → TDB（ET 秒）。"""
        raise NotImplementedError  # pragma: no cover - 骨架

    def body_position(self, body: str, et: float) -> npt.NDArray[np.floating]:
        """天体位置（ICRF km）。"""
        raise NotImplementedError  # pragma: no cover - 骨架


class SPICEManager(EphemerisProvider):
    """SPICE 内核管理器（源：``core/spice.py``）。

    实现状态：待迁入。
    """
