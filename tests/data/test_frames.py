"""data/frames/：EOP/闰秒解析与 SPICE 帧查询测试。"""

from pathlib import Path

import numpy as np
import pytest

from e2m2e.data.frames import (
    ARCSEC_TO_RAD,
    CoordinateDataError,
    EopFile,
    TaiUtcTable,
    frame_rotation,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "core" / "coordinate" / "fixtures" / "gmat"


class TestEopFile:
    @pytest.fixture
    def eop_file(self):
        path = _FIXTURE_DIR / "eopc04_08.62-now.trimmed"
        if not path.is_file():
            pytest.skip("GMAT EOP fixture not found")
        return EopFile.from_file(path)

    def test_parse_and_query(self, eop_file):
        sample = eop_file.at_utc_mjd(61203.0)
        assert sample.x_rad is not None
        assert abs(sample.x_rad) < 1e-3

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
        path = _FIXTURE_DIR / "tai-utc.dat"
        if not path.is_file():
            pytest.skip("GMAT tai-utc fixture not found")
        return TaiUtcTable.from_file(path)

    def test_known_leap_second(self, tai_table):
        # 2017 起 TAI-UTC = 37 s（近两年 MJD 在 58000 附近）
        assert tai_table.tai_minus_utc(58000.0) == pytest.approx(37.0, abs=0.5)

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
