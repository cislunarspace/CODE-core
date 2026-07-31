"""r2s2 库适配器（时空参考系数据）。

r2s2（中科院地月空间时空坐标系库）提供 TDT+GCRS ↔ TDB+EBCRS 相对论时空转换。
本模块管**句柄管理**（历表打开/校验，进程级单例注意）；转换接口由
``EphemerisProvider`` 提供、转换算法在 ``algorithm/coordinate/``。

已知限制：r2s2 的 ``R2S2.init_E`` 是进程级全局状态，多历表实例会互相覆盖
（ADR 0010/0015）。

实现状态：骨架。完整实现待从 ``core/coordinate/gcrs_ebcrs.py`` 迁入。
"""

from __future__ import annotations

__all__ = ["R2S2Adapter"]


class R2S2Adapter:
    """r2s2 历表适配器。

    实现状态：待迁入（源 ``core/coordinate/gcrs_ebcrs.py`` 的 GCRSEBCRSSystem）。
    职责：历表句柄管理 + 时间星历校验 + TT↔TDB + EMB 平移。
    """

    def __init__(self, ephemeris_path: str | list[str]) -> None:  # pragma: no cover
        raise NotImplementedError("R2S2Adapter 待从 core/coordinate/gcrs_ebcrs.py 迁入")
