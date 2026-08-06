"""RESULTS_HMN/LGA/WSB.TXT 解析测试。"""

import numpy as np
import pytest

from scripts.dfh_results import (
    parse_results_hmn,
    parse_results_multi,
    read_results_hmn,
    read_results_wsb,
)


class TestResultsHmn:
    def test_golden_fixture(self, fixtures_dir):
        r = read_results_hmn(fixtures_dir / "RESULTS_HMN.TXT")
        assert r.tli_epoch_mjd == pytest.approx(51569.1007428704)
        assert r.noi_epoch_mjd == pytest.approx(51574.5007428704)
        assert r.tof_day == pytest.approx(5.4)
        assert r.delta_v_noi == pytest.approx(802.368813537801)

    def test_tli_state_and_elements(self, fixtures_dir):
        r = read_results_hmn(fixtures_dir / "RESULTS_HMN.TXT")
        np.testing.assert_allclose(
            r.tli_state,
            [
                -6150.16833338456,
                2736.21211807013,
                794.798350110074,
                -6159.27369443213,
                -7338.48459156835,
                -4891.93023333589,
            ],
        )
        # 根数跨两行，中间隔一行 "a(km), e, ..." 标签
        np.testing.assert_allclose(
            r.tli_elements,
            [
                210611.162112202,
                0.969007300251942,
                30.0,
                -12.1835052364851,
                144.080343965411,
                8.915882209543727e-2,
            ],
        )
        np.testing.assert_allclose(
            r.noi_state[:3], [278240.964623737, -268583.277575811, -117733.473116215]
        )
        np.testing.assert_allclose(
            r.nominal_state[:3], [278241.509601575, -268583.803637638, -117733.473116235]
        )

    def test_raw_text_preserved(self, fixtures_dir):
        path = fixtures_dir / "RESULTS_HMN.TXT"
        r = read_results_hmn(path)
        assert r.raw_text == path.read_text(encoding="utf-8")

    def test_missing_fields_stay_nan(self):
        rr = parse_results_hmn(" Results of direct transfer orbit design.\n")
        assert np.isnan(rr.tli_epoch_mjd)
        assert np.all(np.isnan(rr.tli_state))


class TestResultsMulti:
    def test_wsb_fixture(self, fixtures_dir):
        r = read_results_wsb(fixtures_dir / "RESULTS_WSB.TXT")
        assert r.num_orbits == 10
        assert r.example_index == pytest.approx(7)
        assert "NMAX patched orbits" in r.summary

    def test_wsb_first_segment(self, fixtures_dir):
        r = read_results_wsb(fixtures_dir / "RESULTS_WSB.TXT")
        o = r.orbits[0]
        assert o.index == 1
        np.testing.assert_allclose(
            o.tli_state,
            [
                -5611.80916062855,
                -3603.02017004389,
                -1212.01291260449,
                4098.27085805472,
                -8473.68932997590,
                -5326.16294886928,
            ],
        )
        # 拼接点：位置共享，机动前后速度不同
        np.testing.assert_allclose(
            o.patch_state_before[:3], [1256103.62026374, 414821.027578271, 65678.5215439893]
        )
        np.testing.assert_allclose(
            o.patch_state_before[3:], [47.7953848487694, 143.527237840867, 82.2126062814997]
        )
        np.testing.assert_allclose(
            o.patch_state_after[3:], [-67.4294345855168, 181.224196823085, 144.937064848527]
        )
        np.testing.assert_allclose(
            o.target_state_before[3:], [1379.04210455353, 54.3250571080176, 107.001265336089]
        )
        assert o.delta_v_patch == pytest.approx(136.499734185447)
        assert o.delta_v_target == pytest.approx(20.0160859930045)
        assert o.delta_v_total == pytest.approx(156.515820178452)
        assert o.tof_total_day == pytest.approx(88.4774856782023)

    def test_wsb_epochs(self, fixtures_dir):
        r = read_results_wsb(fixtures_dir / "RESULTS_WSB.TXT")
        o = r.orbits[0]
        # UTC 历元跨两行：Y M D h m + s.s
        assert o.tli_epoch_utc.shape == (6,)
        assert not np.any(np.isnan(o.tli_epoch_utc))
        assert o.tli_epoch_utc[0] == pytest.approx(2023)

    def test_indices_are_one_based(self, fixtures_dir):
        r = read_results_wsb(fixtures_dir / "RESULTS_WSB.TXT")
        assert [o.index for o in r.orbits] == list(range(1, 11))

    def test_empty(self):
        r = parse_results_multi("")
        assert r.num_orbits == 0
        assert np.isnan(r.example_index)
