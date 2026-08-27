"""TDT+GCRS ↔ TDB+EBCRS 时空坐标转换测试。

覆盖 ``GCRSEBCRSSystem``（r2s2 后端）：

- 双向往返一致性（位置 + 时间）；
- 与两条参照链路的差分量化：
  1. 同一历表下的牛顿式平移参照（DFH CoordinateTransform 的做法：
     GCRS 位置加上地心-地月质心偏移，时间尺度不区分 TT/TDB）；
  2. e2m2e 现有 SPICE 链路（SPICEManager + DE440s）。
- 地心处纯时间尺度转换与 ERFA ``dtdb`` 独立模型对比；
- 历表缺时间星历时的报错。

历表要求：含内置时间星历（TT−TDB）。优先 ``de440t.bsp``（JPL 带 t 变体，
https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440t.bsp），缺省时退到
INPOP21a spice 格式历表对（主文件 + 时间星历 ``*_time.bsp``，
https://ftp.imcce.fr/pub/ephem/planets/inpop21a/ 的
``inpop21a_TDB_m100_p100_spice.tar.gz`` 内）。两者都没有则跳过。
"""

import os

import erfa
import numpy as np
import pytest
from calcephpy import Constants, NaifId
from kernel_helpers import SPICE_KERNEL_DIR

from e2m2e.algorithm.coordinate import GCRSEBCRSSystem
from e2m2e.data.frames.gmat_fixture import CoordinateDataError

pytestmark = [
    pytest.mark.data,
    pytest.mark.spice,
]

_CONSTANTS_KMS = Constants.UNIT_KM | Constants.UNIT_SEC | Constants.USE_NAIFID

# 参照历元：(UTC 字符串, GCRS 地心位置 km)
# 第一、三个取自仓库公共参考历元；第二个取自 DFH CoordinateTransformAndJPLEPH
# main.cpp 的演示算例（2027-12-22，LEO 附近位置）；第三个为月球距离量级
# 位置，覆盖地月空间主使用域。
_CASES = [
    ("2025-06-21T11:00:06", np.array([-6441.728433012397, 2079.512716595627, 350.561489919255])),
    ("2027-12-22T00:00:00", np.array([-6441.728433012397, 2079.512716595627, 350.561489919255])),
    ("2025-06-21T11:00:06", np.array([300000.0, 200000.0, 50000.0])),
]

# 往返容差。r2s2 迭代收敛到 1 ns / 1 mm，但公开 API 用单精度儒略日
# （双精度 float 在 JD≈2.46e6 处分辨率约 40 µs），地月质心绕太阳系质心
# 约 30 km/s，折合约毫米级位置抖动；容差在此基础上留一个量级余量。
_ROUNDTRIP_POS_TOL_KM = 1e-2
_ROUNDTRIP_TIME_TOL_S = 1e-3


def _ephemeris_source():
    """定位含时间星历的历表，返回路径或路径列表；都没有则跳过。"""
    de440t = os.path.join(SPICE_KERNEL_DIR, "de440t.bsp")
    if os.path.exists(de440t):
        return de440t
    inpop = [
        os.path.join(SPICE_KERNEL_DIR, "inpop21a_TDB_m100_p100_spice.bsp"),
        os.path.join(SPICE_KERNEL_DIR, "inpop21a_TDB_m100_p100_spice_time.bsp"),
    ]
    if all(os.path.exists(p) for p in inpop):
        return inpop
    pytest.skip("含时间星历的历表（de440t.bsp 或 INPOP21a spice 对）不存在")


@pytest.fixture(scope="module")
def system():
    return GCRSEBCRSSystem(_ephemeris_source())


def _utc_to_tt_jd(utc_str: str) -> float:
    """UTC 字符串 → TT 儒略日（单段 float）。"""
    date_part, time_part = utc_str.split("T")
    year, month, day = (int(x) for x in date_part.split("-"))
    hour, minute, second = (int(x) for x in time_part.split(":"))
    d1, d2 = erfa.dtf2d("UTC", year, month, day, hour, minute, second)
    tai1, tai2 = erfa.utctai(d1, d2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    return tt1 + tt2


class TestRoundtrip:
    """双向往返一致性。"""

    @pytest.mark.parametrize(("utc", "position"), _CASES)
    def test_gcrs_ebcrs_gcrs(self, system, utc, position):
        """GCRS → EBCRS → GCRS 应回到原时空点。"""
        jd_tt = _utc_to_tt_jd(utc)
        jd_tdb, pos_ebcrs = system.gcrs_to_ebcrs(jd_tt, position)
        jd_tt_back, pos_gcrs_back = system.ebcrs_to_gcrs(jd_tdb, pos_ebcrs)

        assert abs(jd_tt_back - jd_tt) * 86400.0 < _ROUNDTRIP_TIME_TOL_S
        assert np.linalg.norm(pos_gcrs_back - position) < _ROUNDTRIP_POS_TOL_KM

    @pytest.mark.parametrize(("utc", "position"), _CASES)
    def test_ebcrs_gcrs_ebcrs(self, system, utc, position):
        """EBCRS → GCRS → EBCRS 应回到原时空点。"""
        # TDB 与 TT 相差毫秒级，用 TT 儒略日近似构造 TDB 输入即可
        jd_tdb = _utc_to_tt_jd(utc)
        jd_tt, pos_gcrs = system.ebcrs_to_gcrs(jd_tdb, position)
        jd_tdb_back, pos_ebcrs_back = system.gcrs_to_ebcrs(jd_tt, pos_gcrs)

        assert abs(jd_tdb_back - jd_tdb) * 86400.0 < _ROUNDTRIP_TIME_TOL_S
        assert np.linalg.norm(pos_ebcrs_back - position) < _ROUNDTRIP_POS_TOL_KM


class TestVsNewtonianReference:
    """与同历表牛顿式平移参照（DFH 做法）的差分量化。

    DFH CoordinateTransform 把 GCRS 位置平移 ``r_earth - r_emb`` 得到地月
    质心位置，且不区分 TT/TDB。r2s2 封装与其差即相对论修正：空间上主要
    是 (L_B−L_G) 尺度项与 c⁻² 项（地月距离量级下约米级），时间上是
    TDB−TT（±1.7 ms 周期项）。容差围绕这两个量级设置。
    """

    @pytest.mark.parametrize(("utc", "position"), _CASES)
    def test_relativistic_correction_magnitude(self, system, utc, position):
        jd_tt = _utc_to_tt_jd(utc)
        jd_tdb, pos_ebcrs = system.gcrs_to_ebcrs(jd_tt, position)

        # 牛顿式参照：同一历表、同一时刻（不区分 TT/TDB）
        jd1 = float(np.floor(jd_tdb))
        jd2 = jd_tdb - jd1
        earth = system._eph.compute_unit(
            jd1, jd2, NaifId.EARTH, NaifId.EARTH_MOON_BARYCENTER, _CONSTANTS_KMS
        )
        pos_newtonian = position + np.array(earth[:3], dtype=float)

        delta_pos = np.linalg.norm(pos_ebcrs - pos_newtonian)
        # 相对论修正应存在且不超过百米量级
        assert 1e-6 < delta_pos < 0.1
        # TDB−TT 应在 ±1.7 ms 周期项包络内
        assert abs(jd_tdb - jd_tt) * 86400.0 < 3e-3


class TestVsSpiceChain:
    """与 e2m2e 现有 SPICE 链路（SPICEManager + DE440s）的差分。

    SPICE 链路为牛顿式：r_ebcrs = r_gcrs + r(EARTH wrt EMB)，et 由 TDB
    儒略日直接换算。差值来源：相对论修正（米级）、DE440s 与转换所用历表
    的差异（de440t 时亚米级；INPOP 时可达数十米）、J2000/ICRS 轴向微差
    （EMB-SSB 距离上约十米）。容差取 1 km，用于抓接线性错误（轴向反、
    原点错、单位错），不用于评定精度。
    """

    @pytest.mark.parametrize(("utc", "position"), _CASES)
    def test_vs_spice_manager(self, system, utc, position, spice_manager):
        jd_tt = _utc_to_tt_jd(utc)
        jd_tdb, pos_ebcrs = system.gcrs_to_ebcrs(jd_tt, position)

        et = (jd_tdb - 2451545.0) * 86400.0
        earth_rel_emb = spice_manager.get_body_state("EARTH", et, "J2000", "EMB")[:3]
        pos_spice_chain = position + earth_rel_emb

        assert np.linalg.norm(pos_ebcrs - pos_spice_chain) < 1.0


class TestTimeScale:
    """纯时间尺度转换与 ERFA 独立模型对比。"""

    @pytest.mark.parametrize(("utc", "position"), _CASES)
    def test_geocenter_tdb_minus_tt_vs_erfa(self, system, utc, position):
        """地心处 TDB−TT 应与 erfa.dtdb 一致（亚毫秒）。

        地心处 r2s2 的位置相关项为零，时间差即历表时间星历给出的
        位置无关项；erfa.dtdb 是 Fairhead & Bretagnon 解析模型加行星项，
        与数值时间星历独立。
        """
        jd_tt = _utc_to_tt_jd(utc)
        jd_tdb, _pos = system.gcrs_to_ebcrs(jd_tt, np.zeros(3))

        delta_r2s2 = (jd_tdb - jd_tt) * 86400.0
        # dtdb 的日期参数取 TDB 儒略日（与 TT 差毫秒级，影响可忽略）
        jd1 = float(np.floor(jd_tdb))
        delta_erfa = erfa.dtdb(jd1, jd_tdb - jd1, 0.0, 0.0, 0.0, 0.0)

        assert abs(delta_r2s2 - delta_erfa) < 1e-3


class TestEphemerisValidation:
    """历表校验。"""

    def test_missing_file_raises(self):
        with pytest.raises(CoordinateDataError, match="历表文件不存在"):
            GCRSEBCRSSystem("kernels/__nonexistent__.bsp")

    def test_ephemeris_without_time_ephemeris_raises(self):
        """de440s.bsp 不含时间星历，构造应报 CoordinateDataError。"""
        de440s = os.path.join(SPICE_KERNEL_DIR, "de440s.bsp")
        if not os.path.exists(de440s):
            pytest.skip("de440s.bsp 不存在")
        with pytest.raises(CoordinateDataError, match="时间星历"):
            GCRSEBCRSSystem(de440s)
