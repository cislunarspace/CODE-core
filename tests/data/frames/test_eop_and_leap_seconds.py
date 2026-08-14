"""data/frames/：EOP/闰秒解析与 SPICE 帧查询测试。"""

import numpy as np
import pytest

from e2m2e.data.frames import (
    ARCSEC_TO_RAD,
    CoordinateDataError,
    EopFile,
    TaiUtcTable,
    frame_rotation,
    gmat_fixture_path,
)

pytestmark = pytest.mark.data


class TestEopFile:
    @pytest.fixture
    def eop_file(self):
        return EopFile.from_file(gmat_fixture_path("eopc04_08.62-now.trimmed"))

    def test_parse_and_query(self, eop_file):
        sample = eop_file.at_utc_mjd(61203.0)
        assert sample.x_rad == pytest.approx(0.193639 * ARCSEC_TO_RAD)
        assert sample.y_rad == pytest.approx(0.433418 * ARCSEC_TO_RAD)
        assert sample.ut1_utc == pytest.approx(0.0449320)
        assert sample.lod == pytest.approx(-0.0000082)

    def test_interpolates_xy_and_ut1_but_not_lod(self, eop_file):
        left = eop_file.at_utc_mjd(61203.0)
        right = eop_file.at_utc_mjd(61204.0)
        mid = eop_file.at_utc_mjd(61203.5)

        assert mid.x_rad == pytest.approx((left.x_rad + right.x_rad) / 2.0)
        assert mid.y_rad == pytest.approx((left.y_rad + right.y_rad) / 2.0)
        assert mid.ut1_utc == pytest.approx((left.ut1_utc + right.ut1_utc) / 2.0)
        assert mid.lod == pytest.approx(left.lod)

    def test_fixture_gap_raises(self, eop_file):
        with pytest.raises(CoordinateDataError):
            eop_file.at_utc_mjd(59000.0)

    def test_out_of_range_raises(self, eop_file):
        with pytest.raises(CoordinateDataError):
            eop_file.at_utc_mjd(eop_file.start_mjd - 100)

    def test_empty_raises(self):
        with pytest.raises(CoordinateDataError, match="EOP table is empty"):
            EopFile(records=tuple())


class TestTaiUtcTable:
    @pytest.fixture
    def tai_table(self):
        return TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))

    def test_known_leap_seconds(self, tai_table):
        assert tai_table.tai_minus_utc(57754.0) == pytest.approx(37.0)
        assert tai_table.tai_minus_utc(37665.0) == pytest.approx(1.8458580)
        assert tai_table.tai_minus_utc(37666.0) == pytest.approx(1.8469812)

    def test_utc_tai_roundtrip(self, tai_table):
        utc_mjd = 59000.0
        tai_mjd = tai_table.utc_to_tai_mjd(utc_mjd)
        np.testing.assert_allclose(tai_table.tai_to_utc_mjd(tai_mjd), utc_mjd, atol=1e-8)

    def test_empty_raises(self):
        with pytest.raises(CoordinateDataError, match="TAI-UTC table is empty"):
            TaiUtcTable(records=tuple())


class TestSpiceFrames:
    @pytest.mark.spice
    def test_identity_rotation(self):
        rot = frame_rotation("J2000", "J2000", 0.0)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


def test_arcsec_to_rad_constant():
    assert pytest.approx(np.pi / (180.0 * 3600.0)) == ARCSEC_TO_RAD
