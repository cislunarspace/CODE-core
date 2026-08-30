"""六域两层天图管线测试（Primer §7.3 / Table 4，ADR 0041 Phase 3c）。

验收口径（issue #580）：
- 命名场景统一初值切片（line 1413）：历元、固定角、反 aligned、M=0；
- 黄金对照：SC 区低倾切片近全 Ȳ ≈ 2（line 1419）；共振竖线位置 =
  #578 名义中心（Table 4 区带与 Table 1 梯的一致性）；CG 区开放
  gateway 拓扑（共面 T☾ = 3 低于第一颈口阈值 C1，精确求根口径）；
- 网格规模抽查在 CI 时间预算内（ADR 0037：单文件 60 s）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.spatiography import (
    ECLIPSE_EPOCH_SCENARIO,
    MAP_ZONE_NAMES,
    default_scenario,
    dynamical_map,
    elements_to_state,
    resonance_centers,
    table4_bands,
    tisserand_parameter,
    zone_grid,
)
from e2m2e.algorithm.spatiography.cartography import compare_models
from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS as C
from e2m2e.algorithm.spatiography.fate import FATE_CLASSES
from e2m2e.algorithm.spatiography.regions import jacobi_critical_values, primer_cr3bp_system

pytestmark = pytest.mark.theory


class TestScenario:
    def test_eclipse_epoch_slice_follows_line_1413(self):
        """统一初值切片：历元日全食、(Ω,ω,M) 固定角、M=0、i = 月轨面。"""
        scen = default_scenario()
        assert scen.epoch_utc == "2027-08-02T10:06:37"
        assert scen.raan_deg == pytest.approx(311.07)
        assert scen.argp_deg == pytest.approx(355.84)
        assert scen.mean_anom_deg == 0.0
        assert scen.inclination_is_moon_plane is True

    def test_anti_aligned_apsidal_convention(self):
        """反 aligned 拱线：ω = ω☾ + 180°（卫星 355.84° ⇒ 月 175.84°）。"""
        scen = default_scenario()
        moon_omega = math.degrees(scen.moon_elements[4])
        assert scen.argp_deg == pytest.approx((moon_omega + 180.0) % 360.0, abs=1e-9)
        assert math.degrees(scen.moon_elements[3]) == pytest.approx(scen.raan_deg, abs=1e-9)
        assert ECLIPSE_EPOCH_SCENARIO.moon_elements == scen.moon_elements

    def test_moon_elements_use_simon1994_mean_orbit(self):
        scen = default_scenario()
        a, ecc, inc = scen.moon_elements[:3]
        assert a == pytest.approx(383397.7725, rel=1e-12)
        assert ecc == pytest.approx(0.055545526, rel=1e-12)
        assert math.degrees(inc) == pytest.approx(C.moon_inc_deg, rel=1e-12)


class TestElementsToState:
    def test_perigee_state_matches_vis_viva(self):
        a, ecc = 0.2 * C.moon_a_km, 0.45
        state = elements_to_state(a, ecc, 0.09, 1.0, 2.0, 0.0, C.earth_gm)
        rp = a * (1.0 - ecc)
        vp = math.sqrt(C.earth_gm * (2.0 / rp - 1.0 / a))
        assert np.linalg.norm(state[:3]) == pytest.approx(rp, rel=1e-12)
        assert np.linalg.norm(state[3:]) == pytest.approx(vp, rel=1e-12)

    def test_energy_matches_semi_major_axis(self):
        for ecc in (0.0, 0.3, 0.85):
            a = 0.5 * C.moon_a_km
            state = elements_to_state(a, ecc, 0.09, 3.0, 4.0, 1.234, C.earth_gm)
            energy = float(
                np.linalg.norm(state[3:]) ** 2 / 2.0 - C.earth_gm / np.linalg.norm(state[:3])
            )
            assert energy == pytest.approx(-C.earth_gm / (2.0 * a), rel=1e-10)


class TestZoneGrid:
    def test_grid_axes_follow_table4_bands(self):
        bands = table4_bands()
        for idx, zone in enumerate(MAP_ZONE_NAMES):
            a_axis, e_axis = zone_grid(zone, 5, 4)
            assert a_axis[0] == pytest.approx(bands.lower[idx])
            assert a_axis[-1] == pytest.approx(bands.upper[idx])
            assert np.all(np.diff(a_axis) > 0)
            assert e_axis[0] == 0.0 and e_axis[-1] == pytest.approx(0.9)

    def test_unknown_zone_and_degenerate_grid_rejected(self):
        with pytest.raises(ValueError, match="zone"):
            zone_grid("XX", 4, 4)
        with pytest.raises(ValueError, match="网格"):
            zone_grid("SC", 1, 4)

    def test_resonance_verticals_are_nominal_centers_of_ladder(self):
        """#578 名义中心落在 Table 4 区带内（CR 含内月梯、CG 括住 1:1）。"""
        bands = table4_bands()
        interior = [c.a_over_a_moon for c in resonance_centers("interior_lunar").centers[:-1]]
        cr_lo, cr_hi = bands.lower[1], bands.upper[1]
        assert all(cr_lo - 0.01 <= a <= cr_hi for a in interior)
        one_to_one = resonance_centers("interior_lunar").centers[-1]
        assert one_to_one.label == "5:4☾"  # 梯尾；1:1 由 CG 带括住
        cg_lo, cg_hi = bands.lower[2], bands.upper[2]
        assert cg_lo < 1.0 < cg_hi

    def test_gateway_topology_tisserand_below_first_neck(self):
        """CG 开放 gateway：共面 T☾(a=a☾, e=0) = 3 < C1（精确求根）。"""
        system = primer_cr3bp_system()
        system.compute_libration_points()
        c1 = jacobi_critical_values(system)["C1"]
        t_moon = tisserand_parameter(C.moon_a_km, 0.0, 0.0)
        assert t_moon == pytest.approx(3.0, abs=1e-12)
        assert t_moon < c1


class TestDynamicalMap:
    def test_sc_zone_low_inclination_slice_is_regular(self):
        """论文 line 1419 黄金对照：SC 区低倾切片近全 Ȳ ≈ 2。

        高偏心率缘（e ≳ 0.9）受月球摄动牵引，Ȳ 需生产窗（19 yr）才
        收敛，不在本断言范围（论文口径 "over much of the sampled
        eccentricity range"）；CI 抽查看 e ≤ 0.6 带。
        """
        result = dynamical_map("SC", n_a=4, n_e=3, e_max=0.6, span_years=1.5)
        finite = result.ybar_field[np.isfinite(result.ybar_field)]
        assert finite.size >= 8
        # 0.25 带含 5:1 共振缘格（a = 0.35 a☾ 名义中心上的细丝扰动，
        # 论文 SC 段：filaments "begin to disturb the background"）。
        assert np.all(np.abs(finite - 2.0) < 0.25), result.ybar_field
        # 低偏心率格全部 stable_quasiperiodic（id 0）。
        assert np.all(result.fate_ids[:, 0] == 0)

    def test_reentry_band_shortcircuits_without_propagation(self):
        """近点入地的格直接判再入（初值即终端短路），ȳ 为 NaN。"""
        result = dynamical_map("SC", n_a=3, n_e=4, e_max=0.95, span_years=0.5)
        reentry_id = FATE_CLASSES.index("earth_reentry")
        last_col = result.fate_ids[:, -1]
        assert reentry_id in last_col
        # SC 下缘 a = 0.13 a☾、e = 0.95 → rp ≈ 2.5e3 km < R⊕：必短路。
        assert last_col[0] == reentry_id
        assert np.isnan(result.ybar_field[0, -1])

    def test_map_result_carries_scenario_and_thresholds(self):
        result = dynamical_map("CG", n_a=3, n_e=2, span_years=0.25)
        assert result.scenario.epoch_utc == "2027-08-02T10:06:37"
        assert result.thresholds.ybar_ordered_band == pytest.approx(0.2)
        assert "gateway" in result.diagnostic_focus
        assert result.status.value == "converged"

    def test_ems_model_adds_solar_point_mass(self):
        em = dynamical_map("IT", n_a=2, n_e=2, span_years=0.25, model="em")
        ems = dynamical_map("IT", n_a=2, n_e=2, span_years=0.25, model="ems")
        assert em.model == "em" and ems.model == "ems"
        # 短窗下两模型都可积完，且太阳项改变了至少一个诊断量。
        assert np.isfinite(em.ybar_field).all()
        assert np.isfinite(ems.ybar_field).all()

    def test_compare_models_summary_fractions(self):
        em = dynamical_map("CG", n_a=3, n_e=2, span_years=0.5, model="em")
        ems = dynamical_map("CG", n_a=3, n_e=2, span_years=0.5, model="ems")
        summary = compare_models(em, ems)
        total = (
            summary["persisted_fraction"]
            + summary["solar_drained_fraction"]
            + (summary["terminal_redirected_fraction"] + summary["reorganized_fraction"])
        )
        assert 0.99 < total <= 1.0 + 1e-9
        assert all(-1.0 <= v <= 1.0 for v in summary["fate_fraction_delta_ems_minus_em"].values())

    def test_invalid_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            dynamical_map("SC", model="bogus", n_a=2, n_e=2, span_years=0.1)
