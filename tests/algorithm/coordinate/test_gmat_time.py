"""GMAT-compatible 时间尺度与 IAU XYS 链路测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.gmat_time import TimeSystemConverter
from e2m2e.algorithm.coordinate.xys import ErfaXysProvider
from e2m2e.data.frames import (
    JD_MJD_OFFSET,
    CoordinateDataError,
    TaiUtcTable,
    gmat_fixture_path,
)

pytestmark = pytest.mark.data


def _tai_utc_table() -> TaiUtcTable:
    return TaiUtcTable.from_file(gmat_fixture_path("tai-utc.dat"))


def test_time_converter_preserves_utc_tai_tt_definitions():
    converter = TimeSystemConverter(_tai_utc_table())

    for utc_mjd, tai_minus_utc in [(51544.0, 32.0), (57754.0, 37.0), (61203.0, 37.0)]:
        tai_mjd = converter.utc_mjd_to_tai_mjd(utc_mjd)
        assert tai_mjd == pytest.approx(utc_mjd + tai_minus_utc / 86400.0)
        assert converter.tai_mjd_to_utc_mjd(tai_mjd) == pytest.approx(utc_mjd)
        assert converter.utc_mjd_to_a1_mjd(utc_mjd) == pytest.approx(
            utc_mjd + (tai_minus_utc + 0.0343817) / 86400.0
        )
        assert converter.utc_mjd_to_tt_mjd(utc_mjd) == pytest.approx(
            utc_mjd + (tai_minus_utc + 32.184) / 86400.0
        )


def test_time_converter_keeps_et_as_its_public_epoch():
    converter = TimeSystemConverter(_tai_utc_table())

    assert converter.et_to_tt_mjd(0.0) == pytest.approx(51544.5, abs=1e-8)


def test_erfa_xys_converts_tt_mjd_and_returns_plain_floats(monkeypatch):
    """适配器只负责 TT MJD→JD 转换与结果归一化，不以 pyerfa 输出作判据。"""
    observed = {}

    def xys06a(jd_part1, jd_part2):
        observed["arguments"] = (jd_part1, jd_part2)
        return np.float64(1.0), np.float64(2.0), np.float64(3.0)

    monkeypatch.setattr("e2m2e.algorithm.coordinate.xys.erfa.xys06a", xys06a)

    result = ErfaXysProvider().xys(51544.5)

    assert observed["arguments"] == (51544.5 + JD_MJD_OFFSET, 0.0)
    assert result == (1.0, 2.0, 3.0)
    assert all(isinstance(value, float) for value in result)


def test_erfa_xys_rejects_non_finite_epoch():
    provider = ErfaXysProvider()

    with pytest.raises(CoordinateDataError):
        provider.xys(np.nan)

    with pytest.raises(CoordinateDataError):
        provider.xys(np.inf)
