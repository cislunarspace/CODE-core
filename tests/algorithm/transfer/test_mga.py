"""行星际 MGA 连接器闭式内核（``e2m2e.algorithm.transfer.mga``）测试。

验收按 ADR 0055 决策 5 分三类 oracle：

- ① 闭式双路径互证：转角闭式 ``δ(r_p)`` ↔ 反闭式 ``r_p(δ)`` ↔ 独立撞击参数路径；
- ② 守恒量：日心比能量差 ``= v_body·Δv∞``、日心 Tisserand 借力前后不变
  （参考圆轨道上恒等 ``3 − v∞²/v_c²``）；
- ③ 零退化：``v∞out → v∞in`` 时 ``δ→0``、``r_p→∞``、``ΔT→0``、``Δε→0``。

全部为合成矢量注入，本文件不依赖 SPICE。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.transfer.mga import (
    evaluate_flyby,
    flyby_pericenter_radius,
    flyby_turn_angle,
    heliocentric_tisserand,
)

pytestmark = pytest.mark.theory

MU_SUN = 1.32712440018e11  # km³/s²

#: 参考行星轨道：1 AU 圆轨道（日心 x-y 面），v_c 为圆速度。
A_REF = 1.495978707e8
V_CIRC = math.sqrt(MU_SUN / A_REF)
BODY_STATE = np.array([A_REF, 0.0, 0.0, 0.0, V_CIRC, 0.0])
BODY_VELOCITY = BODY_STATE[3:]

#: 地球/金星/火星/木星量级的 (r_p, v∞, 天体 GM) 组合：闭式双路径互证与
#: 具体权威口径无关，GM 取量级值。
_FLYBY_CASES = [
    (7000.0, 5.0, 3.986004418e5),  # 地球量级
    (6500.0, 8.0, 3.24859e5),  # 金星量级
    (15000.0, 3.0, 4.282837e4),  # 火星量级
    (80000.0, 12.0, 1.26686534e8),  # 木星量级
]


def _rotate_perifocal_to_j2000(
    vec: np.ndarray, raan: float, inclination: float, argp: float
) -> np.ndarray:
    """近焦点系矢量 → J2000：``Rz(Ω)·Rx(i)·Rz(ω)``。"""
    cos_w, sin_w = math.cos(argp), math.sin(argp)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    rot_w = np.array([[cos_w, -sin_w, 0.0], [sin_w, cos_w, 0.0], [0.0, 0.0, 1.0]])
    rot_i = np.array([[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]])
    rot_o = np.array([[cos_o, -sin_o, 0.0], [sin_o, cos_o, 0.0], [0.0, 0.0, 1.0]])
    return rot_o @ rot_i @ rot_w @ vec


def _state_from_elements(
    a: float, e: float, inclination: float, raan: float, argp: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """经典根数 → 日心 J2000 状态（km, km/s），定义级对照用。"""
    p = a * (1.0 - e * e)
    radius = p / (1.0 + e * math.cos(nu))
    r_perifocal = np.array([radius * math.cos(nu), radius * math.sin(nu), 0.0])
    v_perifocal = math.sqrt(MU_SUN / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    r = _rotate_perifocal_to_j2000(r_perifocal, raan, inclination, argp)
    v = _rotate_perifocal_to_j2000(v_perifocal, raan, inclination, argp)
    return r, v


def _rotate_about_axis(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋转：绕 ``axis``（自动归一化）转 ``angle`` 弧度。"""
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (
        vec * cos_a + np.cross(unit, vec) * sin_a + unit * float(np.dot(unit, vec)) * (1.0 - cos_a)
    )


def _specific_energy(r: np.ndarray, v: np.ndarray) -> float:
    """日心比能量 ``ε = v²/2 − μ/r``，km²/s²。"""
    return 0.5 * float(np.dot(v, v)) - MU_SUN / float(np.linalg.norm(r))


class TestFlybyTurnAngle:
    """① 转角闭式：双路径互证 + 反闭式往返 + 定义域。"""

    @pytest.mark.parametrize("r_p, v_inf, mu_body", _FLYBY_CASES)
    def test_delta_matches_impact_parameter_path(self, r_p, v_inf, mu_body):
        """δ(r_p) 与独立撞击参数路径 δ(b) 一致（研究级 1e-12）。"""
        delta = flyby_turn_angle(r_p, v_inf, mu_body)
        # 双曲几何：b = a_hyp·sqrt(e²−1) = r_p·sqrt(1 + 2μ/(r_p·v∞²))，
        # 且 sin(δ/2) = 1/sqrt(1 + (b·v∞²/μ)²)。
        b = r_p * math.sqrt(1.0 + 2.0 * mu_body / (r_p * v_inf**2))
        delta_impact = 2.0 * math.asin(1.0 / math.sqrt(1.0 + (b * v_inf**2 / mu_body) ** 2))
        assert abs(delta - delta_impact) < 1e-12

    @pytest.mark.parametrize("r_p, v_inf, mu_body", _FLYBY_CASES)
    def test_pericenter_radius_inverse_round_trip(self, r_p, v_inf, mu_body):
        """反闭式是闭式的逆（相对 1e-9）。"""
        delta = flyby_turn_angle(r_p, v_inf, mu_body)
        assert flyby_pericenter_radius(delta, v_inf, mu_body) == pytest.approx(r_p, rel=1e-9)

    def test_smaller_pericenter_gives_larger_turn(self):
        """r_p 越小转角越大（单调）。"""
        deltas = [flyby_turn_angle(r_p, 6.0, 3.986004418e5) for r_p in (20000.0, 10000.0, 7000.0)]
        assert deltas[0] < deltas[1] < deltas[2]

    def test_zero_turn_angle_gives_infinite_pericenter(self):
        """δ == 0 是零退化极限：近心点半径 → ∞（无几何约束）。"""
        assert flyby_pericenter_radius(0.0, 5.0, 3.986004418e5) == math.inf

    def test_turn_angle_limit_is_pi(self):
        """r_p → 0 时 δ → π（掠地极限）：π − δ ≈ 2·sqrt(2·r_p·v∞²/μ) ≈ 7.1e-7。"""
        assert flyby_turn_angle(1e-9, 5.0, 3.986004418e5) == pytest.approx(math.pi, abs=1e-6)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
    def test_non_positive_inputs_rejected(self, bad):
        with pytest.raises(ValueError, match="正的有限值"):
            flyby_turn_angle(bad, 5.0, 3.986004418e5)
        with pytest.raises(ValueError, match="正的有限值"):
            flyby_pericenter_radius(1.0, bad, 3.986004418e5)

    @pytest.mark.parametrize("delta", [-0.1, math.pi, 1.5 * math.pi, math.inf])
    def test_out_of_domain_turn_angle_rejected(self, delta):
        with pytest.raises(ValueError, match="delta_rad"):
            flyby_pericenter_radius(delta, 5.0, 3.986004418e5)


class TestHeliocentricTisserand:
    """定义级对照：经典根数手算公式。"""

    def test_matches_classical_formula(self):
        """T = a_ref/a + 2·cos i·sqrt(a(1−e²)/a_ref)（相对 1e-12）。"""
        a, e, inclination = 2.5e8, 0.3, 0.2
        r, v = _state_from_elements(a, e, inclination, 0.4, 0.9, 0.7)
        expected = A_REF / a + 2.0 * math.cos(inclination) * math.sqrt(a * (1.0 - e * e) / A_REF)
        assert heliocentric_tisserand(r, v, MU_SUN, A_REF) == pytest.approx(expected, rel=1e-12)

    def test_circular_coplanar_identity(self):
        """参考行星圆轨道上恒等 T = 3 − v∞²/v_c²（任意 v∞ 方向）。"""
        for v_inf in ([0.0, 2.0, 0.0], [3.0, 1.0, -2.0], [0.0, 0.0, 4.0]):
            state_v = BODY_VELOCITY + np.array(v_inf)
            expected = 3.0 - float(np.dot(v_inf, v_inf)) / V_CIRC**2
            assert heliocentric_tisserand(BODY_STATE[:3], state_v, MU_SUN, A_REF) == pytest.approx(
                expected, rel=1e-12
            )

    def test_zero_v_inf_on_circular_orbit_is_three(self):
        assert heliocentric_tisserand(BODY_STATE[:3], BODY_VELOCITY, MU_SUN, A_REF) == (
            pytest.approx(3.0, rel=1e-14)
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.inf])
    def test_non_positive_reference_rejected(self, bad):
        with pytest.raises(ValueError, match="a_ref_km"):
            heliocentric_tisserand(BODY_STATE[:3], BODY_VELOCITY, MU_SUN, bad)

    def test_degenerate_state_rejected(self):
        with pytest.raises(ValueError, match="角动量"):
            heliocentric_tisserand([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], MU_SUN, A_REF)


class TestEvaluateFlyby:
    """② 守恒量 + 剔除面 + ③ 零退化（经 ``evaluate_flyby`` 装配层）。"""

    @staticmethod
    def _evaluation(v_inf_in, v_inf_out, *, tol=1e-9, r_p_min=1.0, mu_body=3.986004418e5):
        return evaluate_flyby(
            v_inf_in,
            v_inf_out,
            BODY_STATE,
            "TESTBODY",
            mu_body,
            MU_SUN,
            A_REF,
            r_p_min,
            tol,
        )

    def test_equal_norm_energy_identity_and_tisserand(self):
        """② 等模旋转：|Δv∞| 只改方向 → Δε = v_body·Δv∞、T 不变。"""
        v_inf_in = np.array([0.5, -1.5, 3.0])
        v_inf_out = _rotate_about_axis(v_inf_in, np.array([1.0, 2.0, 3.0]), 0.7)
        evaluation = self._evaluation(v_inf_in, v_inf_out)
        assert evaluation is not None
        # 等模旋转不改变 V∞ 模（装配层报告 in/out 均值）
        norm_in = float(np.linalg.norm(v_inf_in))
        assert evaluation.v_inf_km_s == pytest.approx(norm_in, rel=1e-12)
        # 转角即两向量的夹角（绕非垂直轴的旋转小于旋转角，故按 arccos 独立复核）
        expected_angle = math.acos(
            float(np.dot(v_inf_in, v_inf_out)) / (norm_in * float(np.linalg.norm(v_inf_out)))
        )
        assert evaluation.turn_angle_rad == pytest.approx(expected_angle, abs=1e-12)
        # 日心比能量差恒等于 v_body·(v∞out − v∞in)（|v∞| 相等时）
        delta_energy = _specific_energy(
            BODY_STATE[:3], BODY_VELOCITY + v_inf_out
        ) - _specific_energy(BODY_STATE[:3], BODY_VELOCITY + v_inf_in)
        assert delta_energy - float(np.dot(BODY_VELOCITY, v_inf_out - v_inf_in)) < 1e-12
        # Tisserand：无动力飞越前后名义不变（诊断量，容差 1e-12）
        assert evaluation.tisserand_before == pytest.approx(evaluation.tisserand_after, abs=1e-12)

    def test_rejects_mismatched_v_inf_magnitude(self):
        evaluation = self._evaluation([4.0, 0.0, 0.0], [4.5, 0.0, 0.0], tol=0.1)
        assert evaluation is None

    def test_tolerance_boundary_is_not_rejected(self):
        """|Δv∞| == tol 含边界（``<=`` 不剔）。"""
        evaluation = self._evaluation([4.0, 0.0, 0.0], [4.5, 0.0, 0.0], tol=0.5)
        assert evaluation is not None
        assert evaluation.turn_angle_rad == 0.0

    def test_rejects_below_pericenter_floor(self):
        """r_p 低于下限即剔除；δ == 0（r_p = ∞）恒通过下限。"""
        deep = _rotate_about_axis(np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0]), 2.5)
        assert self._evaluation([0.0, 5.0, 0.0], deep, r_p_min=1e12) is None
        collinear = self._evaluation([0.0, 5.0, 0.0], [0.0, 5.0, 0.0], r_p_min=1e12)
        assert collinear is not None
        assert collinear.pericenter_radius_km == math.inf

    def test_rejects_zero_relative_velocity(self):
        """与天体同速（v∞ = 0）时转角与近心点半径无定义 → 剔除。"""
        assert self._evaluation([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) is None

    def test_zero_degeneracy_limit(self):
        """③ θ → 0：δ → 0、r_p → ∞、ΔT → 0、Δε → 0；r_p 随 θ 单调递减。

        转轴取 ⊥ v∞（Rodrigues 转角即两向量夹角），使 δ 与 θ 可直接对照。
        """
        v_inf_in = np.array([1.0, 2.0, 0.5])
        axis = np.cross(v_inf_in, np.array([0.0, 0.0, 1.0]))
        angles = [1.0, 0.3, 0.1, 0.01, 0.001]
        previous_radius = 0.0  # θ 递减扫描，r_p 应单调递增（δ 越小近心点越远）
        for angle in angles:
            v_inf_out = _rotate_about_axis(v_inf_in, axis, angle)
            evaluation = self._evaluation(v_inf_in, v_inf_out, tol=1e-12)
            assert evaluation is not None
            assert abs(evaluation.turn_angle_rad - angle) < 1e-12
            assert evaluation.pericenter_radius_km > previous_radius
            assert abs(evaluation.tisserand_after - evaluation.tisserand_before) < 1e-12
            previous_radius = evaluation.pericenter_radius_km
        degenerate = self._evaluation(v_inf_in, v_inf_in.copy(), tol=1e-12)
        assert degenerate is not None
        assert degenerate.turn_angle_rad == 0.0
        assert degenerate.pericenter_radius_km == math.inf
        assert degenerate.tisserand_after == degenerate.tisserand_before
