"""r2s2 库适配器（时空参考系数据）。

r2s2（中科院地月空间时空坐标系库）提供 TDT+GCRS ↔ TDB+EBCRS 相对论时空
转换。本模块管**句柄管理**（历表打开/校验，进程级单例注意）与时间星历
校验；转换接口由 ``EphemerisProvider`` 提供、转换算法在
``algorithm/coordinate/``（源：``core/coordinate/gcrs_ebcrs.py`` 句柄管理
部分，ADR 0011/0015 迁移）。

已知限制：r2s2 的 ``R2S2.init_E`` 是进程级全局状态，多历表实例会互相
覆盖（ADR 0010/0015）。

历表要求：必须含内置时间星历（TT−TDB），推荐 JPL ``de440t.bsp``（注意带
``t`` 后缀的变体）；INPOP21a 的 spice 格式历表对（主文件 +
``*_time.bsp``）可作为路径列表传入。普通的 ``de440s.bsp``/``de440.bsp``
不含时间星历，会在构造时报错。
"""

from __future__ import annotations

import math
import os

import numpy as np
import numpy.typing as npt
import R2S2
from calcephpy import CalcephBin, Constants, NaifId

from .eop import CoordinateDataError

_CONSTANTS_KMS = Constants.UNIT_KM | Constants.UNIT_SEC | Constants.USE_NAIFID


class R2S2Adapter:
    """r2s2 历表句柄管理：打开/校验/EMB 平移/JD 拆分。

    Args:
        ephemeris_path: 含内置时间星历的行星历表路径（如 ``de440t.bsp``），
            或路径列表（如 INPOP 主文件 + 时间星历文件）。
    """

    def __init__(self, ephemeris_path: str | os.PathLike | list[str]) -> None:
        if isinstance(ephemeris_path, (list, tuple)):
            paths: list[str] = [str(p) for p in ephemeris_path]
        else:
            paths = [str(ephemeris_path)]
        for path in paths:
            if not os.path.exists(path):
                raise CoordinateDataError(f"历表文件不存在: {path}")
        source: str | list[str] = paths[0] if len(paths) == 1 else paths
        R2S2.init_E(source)
        self._eph = CalcephBin.open(source)
        self._check_time_ephemeris()

    def _check_time_ephemeris(self) -> None:
        """构造时验证历表含 TT−TDB 时间星历，否则 r2s2 转换无法进行。"""
        try:
            self._eph.compute_unit(
                2451545.0,
                0.0,
                NaifId.TIME_TTMTDB,
                NaifId.TIME_CENTER,
                _CONSTANTS_KMS,
            )
        except Exception as exc:
            raise CoordinateDataError(
                "历表不含内置时间星历（TT−TDB），r2s2 时空转换需要 de440t.bsp "
                "（带 t 后缀变体）或 INPOP21a spice 历表对（含 *_time.bsp）"
            ) from exc

    def emb_position(self, jd1: float, jd2: float) -> npt.NDArray[np.floating]:
        """地月质心相对太阳系质心的位置（km），轴向同 BCRS/ICRS。"""
        pv = self._eph.compute_unit(
            jd1,
            jd2,
            NaifId.EARTH_MOON_BARYCENTER,
            NaifId.SOLAR_SYSTEM_BARYCENTER,
            _CONSTANTS_KMS,
        )
        return np.array(pv[:3], dtype=float)

    @staticmethod
    def split_jd(jd: float) -> tuple[float, float]:
        """把单精度儒略日拆成整数与小数两段，保持 r2s2 的时间分辨率。"""
        jd1 = float(math.floor(jd))
        return jd1, jd - jd1
