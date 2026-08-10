"""GMAT fixture 解析器、时间转换与约化测试。

覆盖 TAI-UTC 表、EOP 文件、时间转换器与 ITRF 矩阵（动态对照 SPICE ITRF93）。
"""

import os

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.gmat_itrf import GmatItrfReduction
from e2m2e.algorithm.coordinate.gmat_time import TimeSystemConverter
from e2m2e.algorithm.coordinate.standard_axes import ITRFSpiceAxes
from e2m2e.algorithm.coordinate.xys import ErfaXysProvider
from e2m2e.data.frames.eop import ARCSEC_TO_RAD, EopFile
from e2m2e.data.frames.gmat_fixture import CoordinateDataError, gmat_fixture_path
from e2m2e.data.frames.leap_seconds import TaiUtcTable

pytestmark = pytest.mark.data


def test_tai_utc_table_preserves_post_2017_leap_seconds():
    table = TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))

    assert table.tai_minus_utc(57754.0) == pytest.approx(37.0)


def test_tai_utc_table_handles_pre_1972_drift_rows():
    table = TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))

    assert table.tai_minus_utc(37665.0) == pytest.approx(1.8458580)
    assert table.tai_minus_utc(37666.0) == pytest.approx(1.8469812)


def test_trimmed_eop_fixture_parses_known_2026_row():
    eop = EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))
    sample = eop.at_utc_mjd(61203.0)

    assert sample.x_rad == pytest.approx(0.193639 * ARCSEC_TO_RAD)
    assert sample.y_rad == pytest.approx(0.433418 * ARCSEC_TO_RAD)
    assert sample.ut1_utc == pytest.approx(0.0449320)
    assert sample.lod == pytest.approx(-0.0000082)


def test_eop_interpolates_xy_and_ut1_but_not_lod():
    eop = EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))
    left = eop.at_utc_mjd(61203.0)
    right = eop.at_utc_mjd(61204.0)
    mid = eop.at_utc_mjd(61203.5)

    assert mid.x_rad == pytest.approx((left.x_rad + right.x_rad) / 2.0)
    assert mid.y_rad == pytest.approx((left.y_rad + right.y_rad) / 2.0)
    assert mid.ut1_utc == pytest.approx((left.ut1_utc + right.ut1_utc) / 2.0)
    assert mid.lod == pytest.approx(left.lod)


def test_eop_fixture_gaps_raise_even_inside_outer_range():
    eop = EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))

    with pytest.raises(CoordinateDataError):
        eop.at_utc_mjd(59000.0)


def test_eop_out_of_range_raises_by_default_and_can_clamp():
    eop = EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))

    with pytest.raises(CoordinateDataError):
        eop.at_utc_mjd(50000.0)

    with pytest.raises(CoordinateDataError):
        eop.at_utc_mjd(70000.0)

    clamped = eop.at_utc_mjd(50000.0, extrapolation="clamp")
    assert clamped.mjd == pytest.approx(eop.start_mjd)


def test_time_converter_epoch_conversions_for_j2000_2017_and_2026():
    table = TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))
    converter = TimeSystemConverter(table)
    cases = [
        (51544.0, 32.0),
        (57754.0, 37.0),
        (61203.0, 37.0),
    ]

    for utc_mjd, tai_minus_utc in cases:
        tai_mjd = converter.utc_mjd_to_tai_mjd(utc_mjd)
        assert tai_mjd == pytest.approx(utc_mjd + tai_minus_utc / 86400.0)
        assert converter.tai_mjd_to_utc_mjd(tai_mjd) == pytest.approx(utc_mjd)
        assert converter.utc_mjd_to_a1_mjd(utc_mjd) == pytest.approx(
            utc_mjd + (tai_minus_utc + 0.0343817) / 86400.0
        )
        assert converter.utc_mjd_to_tt_mjd(utc_mjd) == pytest.approx(
            utc_mjd + (tai_minus_utc + 32.184) / 86400.0
        )


def test_time_converter_keeps_et_public_and_exposes_low_level_a1():
    table = TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))
    converter = TimeSystemConverter(table)

    assert converter.utc_mjd_to_tai_mjd(57754.0) == pytest.approx(57754.0 + 37.0 / 86400.0)
    assert converter.utc_mjd_to_a1_mjd(57754.0) == pytest.approx(
        57754.0 + (37.0 + 0.0343817) / 86400.0
    )
    assert converter.et_to_tt_mjd(0.0) == pytest.approx(51544.5, abs=1e-8)


def test_erfa_xys_provider_matches_reference_epoch_values():
    """参考值取自 SOFA ``iauXys06a``（``erfa.xys06a``，IAU 2006 岁差 + 2000A 章动模型）。

    三个历元分别对应 J2000（TT MJD 51544.5）、2017-01-01、2026-06-12
    （2026 年 EOP fixture 采样窗口）。SOFA IAU 2006/2000A 模型是物理定义
    （ADR 0013 决策 2 允许的文献值），非 golden 对照。实测与
    ``erfa.xys06a(jd, 0.0)`` 逐位一致。
    """
    provider = ErfaXysProvider()

    assert provider.xys(51544.5) == pytest.approx(
        (-2.694638014904722e-05, -2.8004721164764934e-05, -1.0133965177563803e-08)
    )
    assert provider.xys(57754.0) == pytest.approx(
        (0.0016391211394910907, -4.7004517266413526e-05, 3.543052006656307e-08)
    )
    assert provider.xys(61203.0) == pytest.approx(
        (0.002582493526172671, 2.9498181435383408e-05, -3.3505790903890935e-08)
    )


def test_erfa_xys_provider_rejects_invalid_time_input():
    provider = ErfaXysProvider()

    with pytest.raises(CoordinateDataError):
        provider.xys(np.nan)

    with pytest.raises(CoordinateDataError):
        provider.xys(np.inf)


def test_gmat_itrf_reduction_matches_spice_itrf93():
    """GMAT 原生约化的 ITRF 旋转矩阵动态对照 SPICE ITRF93（ADR 0003 决策 5）。

    硬编码矩阵已删除：golden 对照违反 ADR 0013（只证回归不证正确），
    正确性由动态对照 SPICE ITRF93 在 1e-7 量级裁决（原生链采用约化的 IAU
    岁差章动 + 线性插值 EOP，精度预期在 1e-7，非 SPICE 的 1e-12 量级）。
    保留物理定义断言 R @ R.T == I（正交性）。
    """
    from kernel_helpers import load_body_fixed_kernels, unload_kernels

    from e2m2e.data.kernels.manager import SPICEManager

    manager = SPICEManager()
    loaded = load_body_fixed_kernels(manager)
    if "earth_latest_high_prec.bpc" not in [os.path.basename(p) for p in loaded]:
        pytest.skip("需 ITRF93 BPC 内核做 SPICE 动态对照（ADR 0003 决策 5）")

    table = TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))
    eop = EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))
    reduction = GmatItrfReduction(
        TimeSystemConverter(table), eop, ErfaXysProvider(), eop_extrapolation="clamp"
    )

    rotation, rate = reduction.rotation_and_rate(0.0)
    expected_rotation, expected_rate = ITRFSpiceAxes(frame="ITRF93").rotation_and_rate(0.0)

    try:
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-7)
        np.testing.assert_allclose(rate, expected_rate, atol=1e-7)
        # 物理定义：旋转矩阵正交
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    finally:
        unload_kernels(manager, loaded)
