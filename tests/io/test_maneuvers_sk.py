"""MANEUVERS.TXT / SK_STATISTIC.TXT 解析测试。"""

import numpy as np
import pytest

from e2m2e.io import parse_maneuvers, parse_sk_statistic, read_maneuvers, read_sk_statistic


class TestManeuvers:
    def test_golden_fixture(self, fixtures_dir):
        m = read_maneuvers(fixtures_dir / "MANEUVERS.TXT")
        assert len(m) == 119
        assert m.mjd_tdb[0] == pytest.approx(51544.5007428704)
        # 首行脉冲为 Fortran 指数格式 0.000000000000000E+000
        assert m.delta_v_mps[0] == pytest.approx(0.0)
        assert m.delta_v_mps[1] == pytest.approx(0.152604623545358)
        assert m.mjd_tdb[-1] == pytest.approx(55084.5007428704)
        assert m.delta_v_mps[-1] == pytest.approx(0.179053046936329)

    def test_scientific_notation_and_negative(self):
        m = parse_maneuvers("  51544.5   1.234567890123456E-002\n 51545.5  -0.5\n")
        assert len(m) == 2
        assert m.delta_v_mps[0] == pytest.approx(1.234567890123456e-2)
        assert m.delta_v_mps[1] == pytest.approx(-0.5)

    def test_bad_lines_skipped(self):
        m = parse_maneuvers("51544.5 0.1\n只有一列文字\n\n51545.5 0.2\n")
        assert len(m) == 2

    def test_empty(self):
        m = parse_maneuvers("")
        assert len(m) == 0


class TestSKStatistic:
    def test_golden_fixture_three_columns(self, fixtures_dir):
        s = read_sk_statistic(fixtures_dir / "SK_STATISTIC.TXT")
        assert len(s) == 100
        assert s.rows.shape == (100, 3)
        assert not s.has_attitude
        assert s.num_failed == 0
        np.testing.assert_allclose(s.rows[0], [1, 1.14597492974386, 0.469637630079300])
        np.testing.assert_allclose(s.rows[-1], [100, 1.13712330634234, 0.344600257063161])
        # 第一列为从 1 开始的仿真索引
        np.testing.assert_array_equal(s.rows[:, 0], np.arange(1, 101))

    def test_five_columns_with_attitude(self):
        raw = (
            "           1   1.14597492974386   0.469637630079300   0.01   0.02\n"
            "           2   0.771507728014600   0.202344783270169   0.03   0.04\n"
            "           3   1.47288284429211   0.429789826762765   0.05   0.06\n"
            " The number of failed Monte Carlo tests is           2\n"
        )
        s = parse_sk_statistic(raw)
        assert s.rows.shape == (3, 5)
        assert s.has_attitude
        assert s.num_failed == 2
        np.testing.assert_allclose(s.rows[1], [2, 0.7715077280146, 0.202344783270169, 0.03, 0.04])

    def test_failure_line_not_treated_as_data(self):
        # 末行失败次数含数值但列数与数据行不同，不能混入数据行
        raw = "  1  0.5  0.2\n The number of failed Monte Carlo tests is           7\n"
        s = parse_sk_statistic(raw)
        assert len(s) == 1
        assert s.num_failed == 7

    def test_missing_failure_line(self):
        s = parse_sk_statistic("  1  0.5  0.2\n")
        assert s.num_failed is None

    def test_empty(self):
        s = parse_sk_statistic("")
        assert len(s) == 0
        assert s.num_failed is None
