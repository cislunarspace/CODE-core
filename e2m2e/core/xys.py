"""IAU X, Y, s 参数提供器实现。"""

from __future__ import annotations

import abc
import math

import erfa

from .gmat_data import CoordinateDataError
from .gmat_eop import JD_MJD_OFFSET


class XysProvider(abc.ABC):
    """为 ITRF native reduction 提供 IAU CIP/CIO ``X,Y,s``。

    保留为 ABC（而非折叠为唯一实现 ErfaXysProvider）：ADR 0003 明确记录了
    日后用 GMAT 表格支撑的 ``GMATXysProvider`` 替换 pyerfa 来源以实现与 GMAT
    XYS 插值精确一致的需求——erfa 可替换性是真实设计意图，非"万一"式死灵活性。
    """

    @abc.abstractmethod
    def xys(self, tt_mjd: float) -> tuple[float, float, float]:
        """返回 TT MJD 对应的 ``X,Y,s``（弧度）。"""
        raise NotImplementedError


class ErfaXysProvider(XysProvider):
    """基于 pyerfa/SOFA 的 IAU 2006/2000A ``X,Y,s`` provider。"""

    def xys(self, tt_mjd: float) -> tuple[float, float, float]:
        if not math.isfinite(tt_mjd):
            raise CoordinateDataError(f"Invalid TT MJD for XYS provider: {tt_mjd!r}")
        jd_tt = tt_mjd + JD_MJD_OFFSET
        x, y, s = erfa.xys06a(jd_tt, 0.0)
        return float(x), float(y), float(s)
