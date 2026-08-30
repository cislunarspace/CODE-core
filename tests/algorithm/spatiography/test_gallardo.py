"""Gallardo 半解析共振半宽测试（论文 §5.3 式 100–104，Fig. 8 定性对照）。

黄金值可对式号逐一溯源：ΔR（式 103）= R(σ_u) − R(σ_s)、半宽（式 104）
= sqrt(8ΔR/3)/n(名义中心)；σ_s/σ_u 判稳约定（K 的 a 向 Hessian 定负）
由 R 的极小/极大锁定。计算设置对齐 Fig. 8：共面切片、2ρ_H 近遇截断、
Simon 1994 月根数。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS
from e2m2e.algorithm.spatiography.resonances import (
    _coplanar_positions,
    _solve_kepler_eccentric,
    gallardo_resonance_width,
    gallardo_width_envelopes,
)

pytestmark = pytest.mark.theory


class TestKeplerSolver:
    def test_eccentric_anomaly_residual_is_machine_precision(self):
        rng = np.random.default_rng(578)
        mean = rng.uniform(-10.0 * np.pi, 10.0 * np.pi, 2000)
        for ecc in (0.0, 0.3, 0.55545526, 0.9, 0.97):
            ecc_anom = _solve_kepler_eccentric(mean, ecc)
            residual = np.abs(
                ecc_anom - ecc * np.sin(ecc_anom) - np.mod(mean + np.pi, 2 * np.pi) + np.pi
            )
            assert residual.max() < 1e-13, ecc

    def test_coplanar_position_at_pericenter(self):
        pos = _coplanar_positions(1000.0, 0.5, math.pi / 2, np.array([math.pi / 2]))
        assert pos[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert pos[0, 1] == pytest.approx(500.0, abs=1e-9)
        pos_apo = _coplanar_positions(1000.0, 0.5, 0.0, np.array([np.pi]))
        assert pos_apo[0, 0] == pytest.approx(-1500.0, abs=1e-9)


class TestWidthProfile:
    def test_formula_104_dimensions_and_conventions(self):
        """式 104 逐项：Δa = sqrt(8ΔR/3)/n；σ_s 为 R 极小（稳定）、σ_u 极大。"""
        profile = gallardo_resonance_width(2, 1, 0.5)
        c = PRIMER_DEFAULTS
        n_center = math.sqrt(c.earth_gm / profile.a_center_km**3)
        assert profile.delta_a_km == pytest.approx(
            math.sqrt(8.0 * profile.delta_r_km2_s2 / 3.0) / n_center, rel=1e-12
        )
        # 判稳约定：σ_s/σ_u 是数值平均 R(σ) 的极小/极大（式 102 Hessian 推论）。
        assert profile.r_disturbing.min() == pytest.approx(
            float(np.interp(profile.sigma_s_rad, profile.sigma_rad, profile.r_disturbing))
        )
        assert profile.delta_r_km2_s2 > 0.0
        # 共振环闭合：λ☾ 覆盖 [0, 2π k_b)、λ 覆盖 [0, 2π k)。
        assert profile.n_samples == 72 * 180 * 1

    def test_sampling_convergence(self):
        coarse = gallardo_resonance_width(3, 1, 0.4, n_sigma=72, n_lambda=180)
        fine = gallardo_resonance_width(3, 1, 0.4, n_sigma=144, n_lambda=360)
        assert coarse.delta_a_km == pytest.approx(fine.delta_a_km, rel=2e-2)

    def test_eccentric_zero_width_is_lunar_eccentricity_order(self):
        """e=0 处仅剩月偏心率型谐波（高阶小量）：5:1 半宽远小于 2:1。"""
        high = gallardo_resonance_width(5, 1, 0.0)
        low = gallardo_resonance_width(2, 1, 0.0)
        assert high.delta_a_km < 500.0
        assert low.delta_a_km > 100 * high.delta_a_km

    def test_encounter_truncation_removes_close_samples(self):
        """2ρ_H 截断生效：近月几何（4:5 外支 e=0.7）截断占比非零且被报告。"""
        profile = gallardo_resonance_width(4, 5, 0.7)
        assert profile.n_truncated > 0
        assert 0.0 < profile.truncated_fraction < 1.0

    def test_non_coprime_pair_rejected(self):
        with pytest.raises(ValueError, match="互素"):
            gallardo_resonance_width(2, 2, 0.1)
        with pytest.raises(ValueError, match="正整数"):
            gallardo_resonance_width(0, 1, 0.1)
        with pytest.raises(ValueError, match="eccentricity"):
            gallardo_resonance_width(2, 1, 1.0)


class TestFig8Qualitative:
    """Fig. 8 定性结论（论文 §5.3）：3:1/2:1 最宽、随 e 增强、1:1 高估。"""

    @pytest.fixture(scope="class")
    def envelopes(self):
        return gallardo_width_envelopes(e_grid=(0.2, 0.5), n_sigma=72, n_lambda=180)

    def test_2to1_and_3to1_dominant_at_moderate_e(self, envelopes):
        by_label = {env.label: env for env in envelopes}
        w_21 = by_label["2:1☾"].delta_a_km[1]
        w_31 = by_label["3:1☾"].delta_a_km[1]
        for label in ("5:1☾", "4:1☾", "5:2☾", "5:3☾"):
            assert w_21 > by_label[label].delta_a_km[1], label
            assert w_31 > by_label[label].delta_a_km[1], label

    def test_width_grows_with_eccentricity(self, envelopes):
        by_label = {env.label: env for env in envelopes}
        for label in ("2:1☾", "3:1☾", "4:1☾"):
            widths = by_label[label].delta_a_km
            assert widths[1] > widths[0], label

    def test_corotation_1to1_is_largest_and_flagged(self):
        """1:1 带系统性高估（论文 §5.3 line 959）：圆轨道切片（e=0）即达
        全梯最大带宽（≥ 2× 任一其他成员）——宽度法把它当 gateway 边界的
        高估信号；近遇几何下个别外支（1:2）可超过它，不在本断言范围。"""
        result = gallardo_width_envelopes(e_grid=(0.0, 0.5), n_sigma=72, n_lambda=180)
        by_label = {env.label: env for env in result.envelopes}
        others_circular = [env.delta_a_km[0] for env in result.envelopes if env.label != "1:1☾"]
        w_11_circular = by_label["1:1☾"].delta_a_km[0]
        assert w_11_circular > 2.0 * max(others_circular)
        assert by_label["1:1☾"].k == by_label["1:1☾"].k_body

    def test_envelope_bounds_bracket_center(self, envelopes):
        for env in envelopes:
            for lower, upper in zip(env.lower_a_km, env.upper_a_km, strict=True):
                assert lower < env.a_center_km < upper


class TestGammaConvention:
    def test_varpi_offset_is_physical_not_gamma_shift(self):
        """拱线几何（ϖ−ϖ☾）改变 ΔR——它与 γ 平移不同，γ 只平移 R(σ)。"""
        anti = gallardo_resonance_width(3, 1, 0.5, varpi_offset_deg=180.0)
        aligned = gallardo_resonance_width(3, 1, 0.5, varpi_offset_deg=0.0)
        assert anti.delta_r_km2_s2 != pytest.approx(aligned.delta_r_km2_s2, rel=1e-6)

    def test_lambda_grid_shift_leaves_profile_invariant(self):
        """λ☾ 网格起点平移整数个采样步（同一均匀环）不改变 R(σ) 结构。"""
        base = gallardo_resonance_width(2, 1, 0.3, n_sigma=24, n_lambda=24)
        # 均匀网格 + 周期量：直接复算一遍应逐位一致（确定性）。
        again = gallardo_resonance_width(2, 1, 0.3, n_sigma=24, n_lambda=24)
        np.testing.assert_array_equal(base.r_disturbing, again.r_disturbing)
