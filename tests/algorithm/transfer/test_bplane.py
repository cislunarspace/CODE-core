"""B-plane 与双曲渐近线参数化测试（#635，ADR 0013 定义级验证）。

只做定义级验证：正逆变换往返一致性、解析雅可比 vs 数值差分、闭式近心距
与教科书公式对照、基矢量不变量、退化路径。不与外部软件输出对拍。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    AsymptoteParams,
    BPlaneParams,
    asymptote_from_vinf_vector,
    bplane_from_state,
    bplane_jacobian_from_state,
    hyperbolic_time_to_periapsis,
    perilune_radius_closed_form,
    perilune_state_from_bplane,
    state_from_bplane,
    vinf_vector_from_asymptote,
)
from e2m2e.data.constants.datums import Datum

pytestmark = pytest.mark.theory

_MU = Datum.DE421.moon_gm


class TestAsymptoteRoundTrip:
    def test_vinf_vector_round_trip(self):
        """(RHA, DHA, C3) → v⃗∞ → (RHA, DHA, C3) 闭合。"""
        for rha in (0.0, 123.0, 359.0):
            for dha in (-80.0, -10.0, 40.0):
                a = AsymptoteParams(rha_deg=rha, dha_deg=dha, c3_km2_s2=2.5)
                v = vinf_vector_from_asymptote(a)
                assert abs(float(v @ v) - 2.5) < 1e-12
                b = asymptote_from_vinf_vector(v)
                assert abs(((b.rha_deg - rha + 180.0) % 360.0) - 180.0) < 1e-9
                assert abs(b.dha_deg - dha) < 1e-9
                assert abs(b.c3_km2_s2 - 2.5) < 1e-9

    def test_non_positive_c3_rejected(self):
        with pytest.raises(ValueError, match="C3"):
            vinf_vector_from_asymptote(AsymptoteParams(0.0, 0.0, 0.0))


class TestBplaneStateRoundTrip:
    def test_state_round_trip_closes(self):
        """B-plane → 状态 → B-plane 在给定球心距处闭合（入射分支）。"""
        radius = 20000.0
        worst = 0.0
        for v_inf in (0.7, 1.0, 1.3):
            for rha in (0.0, 45.0, 170.0, 300.0):
                for dha in (-30.0, 0.0, 25.0):
                    a = AsymptoteParams(rha, dha, v_inf**2)
                    for bdot_r, bdot_t in ((0.0, 5000.0), (2000.0, -3000.0), (-8000.0, 8000.0)):
                        st = state_from_bplane(a, bdot_r, bdot_t, _MU, radius)
                        bp = bplane_from_state(st, _MU)
                        st2 = state_from_bplane(a, bp.bdot_r_km, bp.bdot_t_km, _MU, radius)
                        worst = max(
                            worst,
                            float(np.linalg.norm(st2[:3] - st[:3]) / np.linalg.norm(st[:3])),
                            float(np.linalg.norm(st2[3:] - st[3:]) / np.linalg.norm(st[3:])),
                        )
                        assert abs(bp.bdot_r_km - bdot_r) < 1e-8
                        assert abs(bp.bdot_t_km - bdot_t) < 1e-8
                        assert abs(bp.dha_deg - dha) < 1e-8
        assert worst < 1e-10

    def test_perilune_state_is_hyperbolic_periapsis(self):
        """perilune_state_from_bplane 的月心距 = r_p，速度 ⊥ 位置。"""
        a = AsymptoteParams(60.0, 12.0, 1.21)
        st = perilune_state_from_bplane(a, 1500.0, -2500.0, _MU)
        r = float(np.linalg.norm(st[:3]))
        bp = bplane_from_state(state_from_bplane(a, 1500.0, -2500.0, _MU, 9000.0), _MU)
        assert abs(r - bp.perilune_radius_km) < 1e-9
        assert abs(float(st[:3] @ st[3:])) < 1e-6  # 近心点径向速度为 0


class TestBplaneJacobian:
    def test_analytic_matches_central_difference(self):
        """解析雅可比 vs 状态中心差分，逐分量相对误差 < 1e-5。"""
        st = state_from_bplane(AsymptoteParams(60.0, 12.0, 1.21), 1500.0, -2500.0, _MU, 9000.0)
        jac = bplane_jacobian_from_state(st, _MU)

        def residual_vec(state: np.ndarray) -> np.ndarray:
            bp = bplane_from_state(state, _MU)
            return np.array([bp.bdot_r_km, bp.bdot_t_km, bp.perilune_radius_km])

        fd = np.zeros((3, 6))
        for i in range(6):
            h = 1e-6 * max(1.0, abs(float(st[i])))
            sp = st.copy()
            sp[i] += h
            sm = st.copy()
            sm[i] -= h
            fd[:, i] = (residual_vec(sp) - residual_vec(sm)) / (2.0 * h)
        rel = np.abs(jac - fd) / (np.abs(fd) + 1e-6)
        assert float(rel.max()) < 1e-5, f"max rel err {rel.max():.3e}"

    def test_jacobian_shape(self):
        st = state_from_bplane(AsymptoteParams(30.0, -5.0, 1.0), 1000.0, 1000.0, _MU, 9000.0)
        assert bplane_jacobian_from_state(st, _MU).shape == (3, 6)

    def test_bound_state_rejected(self):
        st = state_from_bplane(AsymptoteParams(30.0, -5.0, 1.0), 1000.0, 1000.0, _MU, 9000.0)
        with pytest.raises(ValueError, match="非双曲"):
            bplane_jacobian_from_state(st, _MU * 1e6)  # 抬高 μ ⇒ v∞² ≤ 0


class TestPeriluneClosedForm:
    def test_closed_form_matches_state_and_textbook(self):
        """r_p 闭式 == 状态反演值 == 教科书 μ/v∞²·(√(1+(bv∞²/μ)²)−1)。"""
        for v_inf in (0.5, 1.0, 1.5):
            for b in (1000.0, 5000.0):
                a = AsymptoteParams(30.0, 5.0, v_inf**2)
                st = state_from_bplane(a, b, 0.0, _MU, 20000.0)
                bp = bplane_from_state(st, _MU)
                closed = perilune_radius_closed_form(v_inf, b, _MU)
                textbook = _MU / v_inf**2 * (np.sqrt(1.0 + (b * v_inf**2 / _MU) ** 2) - 1.0)
                assert abs(bp.perilune_radius_km - closed) < 1e-9
                assert abs(closed - textbook) < 1e-9

    def test_zero_v_inf_rejected(self):
        with pytest.raises(ValueError, match="v∞"):
            perilune_radius_closed_form(0.0, 1000.0, _MU)


class TestBplaneInvariants:
    def test_basis_and_impact_parameter(self):
        """基矢量单位正交、R̂=Ŝ×T̂、B⃗⊥Ŝ、|h|=b·v∞（守恒量）。"""
        a = AsymptoteParams(200.0, -20.0, 1.44)
        st = state_from_bplane(a, 2500.0, 1500.0, _MU, 10000.0)
        bp = bplane_from_state(st, _MU)
        assert isinstance(bp, BPlaneParams)
        assert abs(float(bp.s_hat @ bp.t_hat)) < 1e-12
        assert abs(float(bp.s_hat @ bp.r_hat)) < 1e-12
        np.testing.assert_allclose(bp.r_hat, np.cross(bp.s_hat, bp.t_hat), atol=1e-12)
        b_vec = bp.bdot_t_km * bp.t_hat + bp.bdot_r_km * bp.r_hat
        assert abs(float(b_vec @ bp.s_hat)) < 1e-9
        h_norm = float(np.linalg.norm(np.cross(st[:3], st[3:])))
        assert abs(h_norm - bp.b_mag_km * bp.v_inf_km_s) < 1e-6 * h_norm


class TestBplaneDegenerate:
    def test_bound_arrival_rejected(self):
        """束缚到达（v∞²≤0，圆轨道）→ ValueError。"""
        r = 2000.0
        v_circ = np.sqrt(_MU / r)
        st = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])
        with pytest.raises(ValueError, match="非双曲"):
            bplane_from_state(st, _MU)

    def test_non_finite_state_rejected(self):
        """非有限状态（NaN/Inf）→ ValueError（不静默产出 NaN 几何）。"""
        st = state_from_bplane(AsymptoteParams(30.0, 0.0, 1.0), 1000.0, 1000.0, _MU, 9000.0)
        bad = st.copy()
        bad[0] = float("nan")
        with pytest.raises(ValueError, match="非有限"):
            bplane_from_state(bad, _MU)

    def test_asymptote_parallel_to_z_rejected(self):
        """Ŝ∥ẑ 时 T 矢量退化 → ValueError。"""
        with pytest.raises(ValueError, match="退化"):
            state_from_bplane(AsymptoteParams(0.0, 90.0, 1.0), 1000.0, 0.0, _MU, 20000.0)

    def test_zero_b_vector_rejected(self):
        with pytest.raises(ValueError, match="B 矢量"):
            state_from_bplane(AsymptoteParams(0.0, 0.0, 1.0), 0.0, 0.0, _MU, 20000.0)

    def test_radius_inside_perilune_rejected(self):
        a = AsymptoteParams(0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="近心距"):
            state_from_bplane(a, 20000.0, 0.0, _MU, 100.0)


class TestHyperbolicTimeToPeriapsis:
    def test_propagation_reaches_perilune(self):
        """入射态传播 t_peri 后命中闭式近月点（定义级）。"""
        from e2m2e.algorithm.transfer.multi_impulse import propagate_two_body

        a = AsymptoteParams(45.0, 8.0, 1.0)
        st = state_from_bplane(a, 3000.0, 4000.0, _MU, 20000.0)
        t_peri = hyperbolic_time_to_periapsis(st, _MU)
        assert t_peri > 0.0  # 入射分支，到近月点时间为正
        peri = perilune_state_from_bplane(a, 3000.0, 4000.0, _MU)
        out = propagate_two_body(st, np.linspace(0.0, t_peri, 50), _MU)
        err = float(np.linalg.norm(out["states"][-1][:3] - peri[:3]))
        assert err < 1.0, f"传播到近月点误差 {err:.3e} km"

    def test_bound_state_rejected(self):
        r = 2000.0
        v_circ = np.sqrt(_MU / r)
        st = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])
        with pytest.raises(ValueError, match="非双曲"):
            hyperbolic_time_to_periapsis(st, _MU)
