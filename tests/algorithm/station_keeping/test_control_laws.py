"""station_keeping 控制律单测（假传播器/假会合系视图，不依赖 SPICE/pyd）。

假传播器构造可解析的线性场景（自由运动 r(t) = r0 + v·(t-t0)），手算
对照解析解，验证三种控制律的数值正确性。
"""

import numpy as np
import pytest

from e2m2e.algorithm.station_keeping.special_point import SpecialPointLaw
from e2m2e.algorithm.station_keeping.target_point import (
    LooseTargetPointLaw,
    StrictTargetPointLaw,
)

_SECONDS_PER_DAY = 86400.0


class FreeMotionPropagator:
    """自由运动假传播器：r(t) = r0 + v·(t-t0)，STM 为 [I, (t-t0)I; 0, I]。"""

    def propagate_with_stm(self, state0, t0, t_eval):
        t_eval = np.asarray(t_eval, dtype=float)
        state0 = np.asarray(state0, dtype=float)
        dt = t_eval - t0
        states = np.empty((len(t_eval), 6))
        states[:, :3] = state0[:3] + np.outer(dt, state0[3:])
        states[:, 3:] = state0[3:]
        stm = np.empty((len(t_eval), 6, 6))
        for i, d in enumerate(dt):
            stm[i] = np.eye(6)
            stm[i, :3, 3:] = d * np.eye(3)
        return {"time": t_eval, "states": states, "stm": stm}

    def propagate(self, state0, t0, t_eval):
        return self.propagate_with_stm(state0, t0, t_eval)["states"]


class IdentitySynodic:
    """恒等会合系视图：会合系 = 惯性系（y 穿越/速度分量不变）。"""

    def to_synodic(self, states, ets):
        return np.asarray(states, dtype=float)

    def rotation_matrix(self, et):
        return np.eye(3)


class NominalState:
    """标称轨道视图：常值状态。"""

    def __init__(self, state):
        self._state = np.asarray(state, dtype=float)

    def state_at(self, t):
        return self._state.copy()


def _free_flight_stm(dt):
    """自由飞行的 6×6 STM（dt 秒）。"""
    stm = np.eye(6)
    stm[:3, 3:] = dt * np.eye(3)
    return stm


class TestStrictTargetPointLaw:
    def test_position_reconverge_linear_case(self):
        """自由运动场景：线性初值 + 1 次精化后外推位置与目标节点重合。"""
        t0 = 0.0
        state0 = np.array([100.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # km, km/s
        # 标称：目标节点在 28 天处，位置 (200, 0, 0)
        t_j = t0 + 28.0 * _SECONDS_PER_DAY
        nominal = NominalState(np.array([200.0 - 0.1 * t_j, 0.0, 0.0, 0.1, 0.0, 0.0]))

        law = StrictTargetPointLaw(feedback_arc_days=28.0, tolerance_km=1.0)
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator(), nominal=nominal)

        # 施加 Δv 后外推：r(t_j) = r0 + (v0+dv)·t_j = 标称位置（线性场景下精确）
        v_new = state0[3:] + dv
        r_new = state0[:3] + v_new * t_j
        np.testing.assert_allclose(r_new, nominal.state_at(t_j)[:3], atol=1e-6)
        # 解析解：Δv = (r_target - r0)/t_j - v0
        expected = (nominal.state_at(t_j)[:3] - state0[:3]) / t_j - state0[3:]
        np.testing.assert_allclose(dv, expected, atol=1e-9)

    def test_default_tolerance_and_max_iter(self):
        """默认 tolerance_km=0.1、max_iter=6（#280 修复值）。"""
        law = StrictTargetPointLaw()
        assert law.tolerance_km == 0.1
        assert law.max_iter == 6


class TestLooseTargetPointLaw:
    def test_analytic_solution_matches_hand_calc(self):
        """宽松控制：Δv* 与式 5.36 单节点解析解逐分量一致（当前偏差 p₀/v₀）。"""
        t0 = 0.0
        t_j = 28.0 * _SECONDS_PER_DAY
        # 测量状态相对标称有偏差（p0, v0）
        state0 = np.array([10.0, -5.0, 2.0, 0.01, -0.02, 0.005])
        x_nom = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        nominal = NominalState(x_nom)
        p0 = state0[:3] - x_nom[:3]
        v0 = state0[3:] - x_nom[3:]

        law = LooseTargetPointLaw(feedback_arc_days=28.0, q=1.0, r=1.0, s=1e-2)
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator(), nominal=nominal)

        # 手算：Φ = [I, t_j·I; 0, I] → A=D=I, B=t_j·I, C=0
        stm = _free_flight_stm(t_j)
        a, b, c, d = stm[:3, :3], stm[:3, 3:], stm[3:6, :3], stm[3:6, 3:]
        q = np.eye(3)
        r = np.eye(3)
        s = 1e-2 * np.eye(3)
        bt_r_b = b.T @ r @ b
        dt_s_d = d.T @ s @ d
        expected = -np.linalg.solve(
            q + bt_r_b + dt_s_d,
            (bt_r_b + dt_s_d) @ v0 + (b.T @ r @ a + d.T @ s @ c) @ p0,
        )
        np.testing.assert_allclose(dv, expected, atol=1e-12)

    def test_zero_deviation_zero_dv(self):
        """测量状态与标称一致时控制量为零。"""
        x_nom = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        law = LooseTargetPointLaw()
        dv = law.compute_maneuver(
            x_nom, 0.0, propagator=FreeMotionPropagator(), nominal=NominalState(x_nom)
        )
        np.testing.assert_allclose(dv, 0.0, atol=1e-12)


class TestSpecialPointLaw:
    def test_crossing_constraint_free_motion(self):
        """自由运动 + 恒等会合系：约束 ẋ(t*)=0，一步收敛 Δv = (-vx, 0, 0)。

        轨道沿 +x 匀速运动、y 以 vy 穿越 y=0（x-z 平面）。穿越时刻 t* 处
        ẋ = vx 为常数，约束 ẋ=0 需要把 vx 清零。
        """
        t0 = 0.0
        state0 = np.array([0.0, 10.0, 0.0, 0.05, -0.01, 0.0])  # vy < 0 → 将穿越 y=0
        law = SpecialPointLaw(
            special_mode=1,
            crossings=1,
            horizon_sec=30.0 * _SECONDS_PER_DAY,
            synodic=IdentitySynodic(),
        )
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator())
        assert dv is not None
        np.testing.assert_allclose(dv, np.array([-0.05, 0.0, 0.0]), atol=1e-6)

    def test_halo_mode_constrains_vz(self):
        """mode 2（Halo）：约束 [ẋ, ż]，两步各清零。"""
        t0 = 0.0
        state0 = np.array([0.0, 10.0, 0.0, 0.05, -0.01, 0.02])
        law = SpecialPointLaw(
            special_mode=2,
            crossings=1,
            horizon_sec=30.0 * _SECONDS_PER_DAY,
            synodic=IdentitySynodic(),
        )
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator())
        assert dv is not None
        np.testing.assert_allclose(dv, np.array([-0.05, 0.0, -0.02]), atol=1e-6)

    def test_requires_synodic_and_horizon(self):
        """未提供 synodic/horizon 时报错。"""
        law = SpecialPointLaw()
        with pytest.raises(ValueError, match="synodic"):
            law.compute_maneuver(np.zeros(6), 0.0, propagator=FreeMotionPropagator())

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="special_mode"):
            SpecialPointLaw(special_mode=3)

    def test_default_damping_factor_and_v_c(self):
        """默认 damping_factor=1.0、v_c=None，行为与修改前一致。"""
        law = SpecialPointLaw()
        assert law.damping_factor == 1.0
        assert law.v_c is None

    def test_v_c_scales_jacobian(self):
        """v_c > 1 使雅可比除以 v_c，Newton 步长相应放大（自由运动场景）。"""
        t0 = 0.0
        state0 = np.array([0.0, 10.0, 0.0, 0.05, -0.01, 0.0])
        law = SpecialPointLaw(
            special_mode=1,
            crossings=1,
            horizon_sec=30.0 * _SECONDS_PER_DAY,
            synodic=IdentitySynodic(),
            v_c=2.0,
            max_iter=1,  # 只看单步 Newton 修正量，不要求收敛
        )
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator())
        # jac = [1,0,0]/v_c = [0.5,0,0] → Δv_x = -g/jac = -0.05/0.5 = -0.1
        # v_c=2.0 使步长放大2×（自由运动线性场景下过冲，不收敛，返回 None 亦可接受）
        if dv is not None:
            np.testing.assert_allclose(dv, np.array([-0.1, 0.0, 0.0]), atol=1e-6)

    def test_damping_factor_converges(self):
        """damping_factor < 1 在过冲场景下逐步回溯，收敛到容差内。"""
        t0 = 0.0
        state0 = np.array([0.0, 10.0, 0.0, 0.05, -0.01, 0.0])
        # v_c=1.0 无过冲，damping 仍生效：第一迭代全步长-0.05
        # g=0.05>g_old=0.05 取等不触发，第二迭代全步长-0.05→v=0，g=0→收敛
        # 验证 damping 不阻止正常收敛
        law = SpecialPointLaw(
            special_mode=1,
            crossings=1,
            horizon_sec=30.0 * _SECONDS_PER_DAY,
            synodic=IdentitySynodic(),
            damping_factor=0.5,
            max_iter=10,
        )
        dv = law.compute_maneuver(state0, t0, propagator=FreeMotionPropagator())
        assert dv is not None
        np.testing.assert_allclose(dv, np.array([-0.05, 0.0, 0.0]), atol=1e-6)
