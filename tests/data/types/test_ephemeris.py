"""EPHEMERIDES_*.TXT 星历读写测试。"""

import numpy as np
import pytest

from e2m2e.data.types.trajectory import (
    EphemerisTable,
    parse_ephemeris,
    read_ephemeris,
    write_ephemeris,
)

pytestmark = pytest.mark.data


class TestReadEphemeris:
    def test_dac_fixture_row_count_and_columns(self, fixtures_dir):
        t = read_ephemeris(fixtures_dir / "EPHEMERIDES_DAC.TXT")
        assert len(t) == 20
        assert t.position_km.shape == (20, 3)
        assert t.velocity_mps.shape == (20, 3)
        assert t.synodic_position.shape == (20, 3)

    def test_dac_fixture_first_row_values(self, fixtures_dir):
        t = read_ephemeris(fixtures_dir / "EPHEMERIDES_DAC.TXT")
        assert (t.year[0], t.month[0], t.day[0]) == (2024, 1, 1)
        assert (t.hour[0], t.minute[0]) == (0, 0)
        assert t.second[0] == pytest.approx(0.0)
        # 首行坐标为负值——验证时间戳的 "-" 分隔符没有拆碎数值列负号
        np.testing.assert_allclose(
            t.position_km[0], [-437418.540296398744, 154148.054367489938, 132680.644840284513]
        )
        np.testing.assert_allclose(
            t.velocity_mps[0], [-402.366323971841, -786.631509834248, -407.081341514286]
        )
        np.testing.assert_allclose(
            t.synodic_position[0], [1.189646709914, 0.003599481287, 0.075806109154]
        )

    def test_negative_coordinates_not_broken_by_timestamp_split(self):
        raw = "2024-06-01-12-30-45.50  -1.0  -2.0  -3.0  -4.0  -5.0  -6.0  -7.0  -8.0  -9.0\n"
        t = parse_ephemeris(raw)
        assert len(t) == 1
        np.testing.assert_allclose(t.position_km[0], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(t.velocity_mps[0], [-4.0, -5.0, -6.0])
        np.testing.assert_allclose(t.synodic_position[0], [-7.0, -8.0, -9.0])
        assert (t.hour[0], t.minute[0]) == (12, 30)
        assert t.second[0] == pytest.approx(45.50)

    def test_wsb_extra_columns_ignored(self, fixtures_dir):
        # EPHEMERIDES_WSB.TXT 时间戳后有 12 个数值列，只取前 9 列
        t = read_ephemeris(fixtures_dir / "EPHEMERIDES_WSB.TXT")
        assert len(t) == 5
        np.testing.assert_allclose(
            t.position_km[0], [-5657.087405000595, -3550.627677198459, -1154.788678275638]
        )
        np.testing.assert_allclose(
            t.synodic_position[0], [0.011967440526, 0.011729857821, 0.001104551756]
        )

    def test_bad_lines_silently_skipped(self):
        raw = (
            "2024-01-01-00-00-00.00  1 2 3 4 5 6 7 8 9\n"
            "not a data line\n"
            "2024-13-99  1 2 3\n"
            "2024-01-02-00-00-00.00  9 8 7 6 5 4 3 2 1\n"
        )
        t = parse_ephemeris(raw)
        assert len(t) == 2
        assert t.day.tolist() == [1, 2]

    def test_empty_text(self):
        t = parse_ephemeris("")
        assert len(t) == 0


class TestWriteReadRoundtrip:
    def test_roundtrip(self, fixtures_dir, tmp_path):
        src = read_ephemeris(fixtures_dir / "EPHEMERIDES_DAC.TXT")
        out = tmp_path / "EPHEMERIDES_OUT.TXT"
        write_ephemeris(src, out)
        dst = read_ephemeris(out)

        assert len(dst) == len(src)
        for name in ("year", "month", "day", "hour", "minute"):
            np.testing.assert_array_equal(getattr(dst, name), getattr(src, name))
        # 写出格式为 12 位小数定点，读回误差不超过 1e-12
        np.testing.assert_allclose(dst.second, src.second, atol=1e-12)
        np.testing.assert_allclose(dst.position_km, src.position_km, atol=1e-12)
        np.testing.assert_allclose(dst.velocity_mps, src.velocity_mps, atol=1e-12)
        np.testing.assert_allclose(dst.synodic_position, src.synodic_position, atol=1e-12)

    def test_write_format(self, tmp_path):
        t = EphemerisTable(
            year=np.array([2024]),
            month=np.array([1]),
            day=np.array([2]),
            hour=np.array([3]),
            minute=np.array([4]),
            second=np.array([5.5]),
            position_km=np.array([[1.0, 2.0, 3.0]]),
            velocity_mps=np.array([[4.0, 5.0, 6.0]]),
            synodic_position=np.array([[7.0, 8.0, 9.0]]),
        )
        out = tmp_path / "eph.txt"
        write_ephemeris(t, out)
        raw = out.read_bytes().decode("utf-8")
        assert raw.endswith("\r\n")
        line = raw.split("\r\n")[0]
        assert line.startswith("2024-01-02-03-04-05.50")
        # 时间戳后 9 个 25 字符宽定点数
        assert len(line) == len("2024-01-02-03-04-05.50") + 9 * 25
