"""TDT+GCRS ↔ TDB+EBCRS 时空坐标转换（r2s2 后端）。

TDT 是 TT（地球时）的旧称；EBCRS 是地月质心天球参考系，轴向与 BCRS/ICRS
一致，原点在地月质心。

r2s2 直接覆盖 (TT, GCRS 地心位置) ↔ (TDB, BCRS 太阳系质心位置) 的相对论
时空转换（``TT2TDB`` / ``TDB2TT``）。EBCRS 与 BCRS 只差原点平移，本模块用
同一历表中的地月质心位置补上这一缺口：``x_ebcrs = xs - x_emb(t)``。

历表要求：必须含内置时间星历（TT−TDB），推荐 JPL ``de440t.bsp``（注意带
``t`` 后缀的变体）；INPOP21a 的 spice 格式历表对（主文件 +
``*_time.bsp``）可作为路径列表传入。普通的 ``de440s.bsp``/``de440.bsp``
不含时间星历，会在构造时报错；INPOP 原生格式（.dat）主文件与时间星历
分属两个文件，calceph 不支持多开，不可用。
"""

from __future__ import annotations

import math
import os

import numpy as np
import numpy.typing as npt
import R2S2
from calcephpy import CalcephBin, Constants, NaifId

from ...data.frames.eop import CoordinateDataError

_CONSTANTS_KMS = Constants.UNIT_KM | Constants.UNIT_SEC | Constants.USE_NAIFID


class GCRSEBCRSSystem:
    """TDT+GCRS ↔ TDB+EBCRS 时空坐标转换器。

    转换同时作用于时间与空间：输入输出均为 ``(儒略日, 位置三元组 km)``，
    时间尺度随方向在 TT（TDT）与 TDB 之间切换。r2s2 只转换位置，速度不在
    转换范围内。

    Args:
        ephemeris_path: 含内置时间星历的行星历表路径（如 ``de440t.bsp``），
            或路径列表（如 INPOP 主文件 + 时间星历文件）。

    Note:
        r2s2 的历表句柄是进程级全局状态（``R2S2.init_E``），用不同历表
        构造多个实例会互相覆盖，后构造者生效。
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

    def _emb_position(self, jd1: float, jd2: float) -> npt.NDArray[np.floating]:
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
    def _split_jd(jd: float) -> tuple[float, float]:
        """把单精度儒略日拆成整数与小数两段，保持 r2s2 的时间分辨率。"""
        jd1 = float(math.floor(jd))
        return jd1, jd - jd1

    def gcrs_to_ebcrs(
        self, jd_tt: float, position_gcrs: npt.ArrayLike
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """(TDT/TT 儒略日, GCRS 地心位置 km) → (TDB 儒略日, EBCRS 位置 km)。"""
        jd1, jd2 = self._split_jd(jd_tt)
        position = np.asarray(position_gcrs, dtype=float)
        tdb1, tdb2, xs = R2S2.TT2TDB(jd1, jd2, position)
        position_ebcrs = xs - self._emb_position(tdb1, tdb2)
        return tdb1 + tdb2, position_ebcrs

    def ebcrs_to_gcrs(
        self, jd_tdb: float, position_ebcrs: npt.ArrayLike
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """(TDB 儒略日, EBCRS 位置 km) → (TDT/TT 儒略日, GCRS 地心位置 km)。"""
        jd1, jd2 = self._split_jd(jd_tdb)
        position = np.asarray(position_ebcrs, dtype=float)
        xs = position + self._emb_position(jd1, jd2)
        tt1, tt2, Xs = R2S2.TDB2TT(jd1, jd2, xs)
        return tt1 + tt2, Xs

