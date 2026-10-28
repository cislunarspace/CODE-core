"""Floquet 模态基础解集与有界性判据测试（主题 3）。

测试轨道为 2:1 地月 DRO（月心近侧穿越 −0.192 LU，会合系 x0 = 1−μ−0.192，
周期约 14.75 d）。论文数值（杨驰航等，航空学报 2023 年 44 卷 5 期 326563
的式(13) 与特征值）属结果性对照，只进 docstring，不进断言；断言一律按
物理定义（单位模、切向恒等、变分方程对拍、周期化闭合、有界性方向）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_halo, design_lyapunov
from e2m2e.algorithm.family.orbits.dro import _correct_dro
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.proximity import FloquetModeError, floquet_modes
from e2m2e.algorithm.proximity.relative_dynamics import RelativeDynamics, TargetOrbit
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration

#: 复现的 2:1 DRO 特征值（月心近侧穿越精确取 −0.192 LU；结果性对照不进断言，
#: 此处仅用于回归钉住收敛轨道本身）。
_EIGENVALUE_IN_PLANE = complex(-0.742423, 0.669932)
_EIGENVALUE_OUT_OF_PLANE = complex(0.139176, 0.990268)


@pytest.fixture(scope="module")
def dro21() -> tuple[Orbit, CR3BP_Dynamics]:
    """2:1 地月 DRO 与地月动力学。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    seed = _correct_dro(dynamics, 0.79188556619742, None)
    orbit = _correct_dro(dynamics, (1.0 - dynamics.system.mu) - 0.192, seed)
    return orbit, dynamics


@pytest.fixture(scope="module")
def modes21(dro21):
    orbit, dynamics = dro21
    return floquet_modes(orbit, dynamics)


class TestClassification:
    """单值矩阵特征值分类。"""

    def test_tangent_is_trivial_eigenvector(self, modes21) -> None:
        """切向向量是单值矩阵的平凡特征向量：M v1 = v1。"""
        modes = modes21
        residual = float(np.linalg.norm(modes.monodromy @ modes.tangent - modes.tangent))
        assert residual < 1e-8

    def test_trivial_pair_multiplicity(self, modes21) -> None:
        """特征值里恰有两个贴近 1 的平凡成员（数值分裂为 1±1e-6 量级）。"""
        near_one = np.abs(modes21.eigenvalues - 1.0) <= 1e-4
        assert int(np.count_nonzero(near_one)) == 2

    def test_in_plane_pair(self, modes21) -> None:
        """面内对：单位模共轭复对，特征向量无 z、vz 分量。

        实部 a 与虚部 b 正交但范数可以不等（复向量整体单位化），z、vz
        支撑按各自范数的占比断言。
        """
        pair = modes21.in_plane
        assert abs(abs(pair.eigenvalue) - 1.0) < 1e-6
        assert abs(pair.eigenvalue - _EIGENVALUE_IN_PLANE) < 1e-4
        for vector in (pair.a, pair.b):
            support = float(np.linalg.norm(vector[[2, 5]])) / float(np.linalg.norm(vector))
            assert support < 1e-6

    def test_out_of_plane_pair(self, modes21) -> None:
        """法向对：单位模共轭复对，特征向量集中在 z、vz 分量上。"""
        pair = modes21.out_of_plane
        assert abs(abs(pair.eigenvalue) - 1.0) < 1e-6
        assert abs(pair.eigenvalue - _EIGENVALUE_OUT_OF_PLANE) < 1e-4
        for vector in (pair.a, pair.b):
            support = float(np.linalg.norm(vector[[2, 5]])) / float(np.linalg.norm(vector))
            assert support > 0.999

    def test_alpha_is_argument(self, modes21) -> None:
        """每周期转角等于特征值辐角，落在 (0, π)。"""
        for pair in (modes21.in_plane, modes21.out_of_plane):
            assert abs(pair.alpha - float(np.angle(pair.eigenvalue))) < 1e-12
            assert 0.0 < pair.alpha < np.pi


class TestBasis:
    """周期化基础解。"""

    def test_grid_covers_one_period(self, modes21) -> None:
        """默认网格从相位原点起恰好覆盖一个周期。"""
        modes = modes21
        assert modes.times.shape[0] == 201
        assert modes.times[0] == pytest.approx(0.0, abs=1e-12)
        assert modes.times[-1] == pytest.approx(modes.t_origin + modes.period, abs=1e-9)
        assert modes.basis.shape == (modes.times.shape[0], 6, 6)

    def test_first_grid_point_is_modal_initials(self, modes21) -> None:
        """网格首点的基矩阵列为各模态初值（Φ(0) = I）。"""
        modes = modes21
        expected = np.column_stack(
            [
                modes.tangent,
                modes.in_plane.a,
                modes.in_plane.b,
                modes.out_of_plane.a,
                modes.out_of_plane.b,
                modes.drift,
            ]
        )
        np.testing.assert_allclose(modes.basis[0], expected, atol=1e-12)

    def test_periodic_modes_close_after_one_period(self, modes21) -> None:
        """切向与两对中心模态一周期后闭合；漂移模态线性增长。"""
        modes = modes21
        for column in range(5):
            closure = float(np.linalg.norm(modes.basis[-1, :, column] - modes.basis[0, :, column]))
            assert closure < 1e-9, f"第 {column} 列模态未闭合，闭合差 {closure:.3e}"
        drift_growth = float(np.linalg.norm(modes.basis[-1, :, 5] - modes.basis[0, :, 5]))
        assert drift_growth > 1.0

    def test_basis_matches_rlm_propagation(self, dro21, modes21) -> None:
        """基础解与 RLM 相对运动传播对拍（独立线性化与积分路径）。

        目标轨迹密网格线性插值后经 RelativeDynamics 逐点线性化积分，
        与单值矩阵路径的差异在登记容差（rtol 2e-4，目标插值主导）内。
        中心模态的物理解按模态旋转组合复原：Φ(t)a = cos(αt/T)u(t) −
        sin(αt/T)v(t)。
        """
        orbit, dynamics = dro21
        state0 = orbit.states[0]
        period = float(orbit.period)

        traj_eval = np.linspace(0.0, 1.05 * period, 4201)
        traj = dynamics.propagate(state0, (0.0, 1.05 * period), t_eval=traj_eval)
        # 目标轨迹多积 5%：积分器末步会轻微过冲传播区间，目标轨迹恰好止于
        # 区间末端时 TargetOrbit.state_at 越界抛错、Rust 侧中断并触发截断
        # 防御（RelativeDynamics.propagate 的既有锐边，目标轨迹须覆盖过冲）
        target = TargetOrbit(
            Orbit(
                states=np.asarray(traj["states"], dtype=float),
                times=traj_eval,
                system=dynamics.system,
            )
        )
        rel = RelativeDynamics(target, dynamics)
        _, rhos_tangent = rel.propagate(modes21.tangent, (0.0, period), max_step=0.01)
        times_r, rhos_center = rel.propagate(modes21.in_plane.a, (0.0, period), max_step=0.01)
        _, rhos_center_b = rel.propagate(modes21.in_plane.b, (0.0, period), max_step=0.01)
        _, rhos_drift = rel.propagate(modes21.drift, (0.0, period), max_step=0.01)

        modes = floquet_modes(orbit, dynamics, times=times_r)
        np.testing.assert_allclose(rhos_tangent, modes.basis[:, :, 0], rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(rhos_drift, modes.basis[:, :, 5], rtol=2e-4, atol=2e-5)

        theta = modes.in_plane.alpha * times_r / period
        predicted_a = (
            np.cos(theta)[:, None] * modes.basis[:, :, 1]
            - np.sin(theta)[:, None] * modes.basis[:, :, 2]
        )
        np.testing.assert_allclose(rhos_center, predicted_a, rtol=2e-4, atol=2e-5)
        predicted_b = (
            np.sin(theta)[:, None] * modes.basis[:, :, 1]
            + np.cos(theta)[:, None] * modes.basis[:, :, 2]
        )
        np.testing.assert_allclose(rhos_center_b, predicted_b, rtol=2e-4, atol=2e-5)


class TestBoundedNormal:
    """有界子空间法向量。"""

    def test_dual_to_drift(self, modes21) -> None:
        """n·v2 = 1，n 与五个有界模态初值正交。"""
        modes = modes21
        assert modes.bounded_normal @ modes.drift == pytest.approx(1.0, abs=1e-9)
        assert float(np.max(np.abs(modes.bounded_normal @ modes.basis[0][:, :5]))) < 1e-9

    def test_bounded_vs_drift_perturbation(self, dro21, modes21) -> None:
        """有界初值的相对状态量级保持不变，漂移初值线性发散。

        中心模态初值是有界振荡（不必回到初态），断言其相对状态在六个
        周期后仍在初始量级；漂移初值的相对状态按每周期约 8 倍 ε 线性
        增长（ε 为初始扰动幅度）。
        """
        orbit, dynamics = dro21
        modes = modes21
        state0 = orbit.states[0]
        period = float(orbit.period)
        eps = 1e-6
        samples = np.arange(1.0, 7.0) * period

        def max_relative_norm(initial: np.ndarray) -> float:
            out = dynamics.propagate(state0 + initial, (0.0, float(samples[-1])), t_eval=samples)
            states = np.asarray(out["states"], dtype=float)
            return float(np.max(np.linalg.norm(states - state0, axis=1)))

        bounded = max_relative_norm(eps * modes.basis[0][:, 1])
        drift = max_relative_norm(eps * modes.drift)
        assert bounded < 3e-6
        assert drift > 2e-5


class TestErrors:
    """错误路径。"""

    def test_period_required(self, dro21) -> None:
        """period 未设置的轨道报 FloquetModeError。"""
        orbit, dynamics = dro21
        bare = Orbit(states=orbit.states[:1].copy(), times=np.array([0.0]))
        with pytest.raises(FloquetModeError, match="period"):
            floquet_modes(bare, dynamics)

    def test_broken_period_breaks_trivial_structure(self, dro21) -> None:
        """周期偏离真值时单值矩阵失去平凡切向结构，确定性报错。"""
        orbit, dynamics = dro21
        wrong = Orbit(states=orbit.states[:1].copy(), times=np.array([0.0]))
        wrong.period = 1.05 * float(orbit.period)
        with pytest.raises(FloquetModeError, match="切向"):
            floquet_modes(wrong, dynamics)

    def test_hyperbolic_pair_rejected(self) -> None:
        """双曲实对（不稳定周期轨道，L2 Lyapunov）不在本模块形态内，确定性报错。"""
        orbit = design_lyapunov(2, 5000.0)
        dynamics = CR3BP_Dynamics(earth_moon_system())
        with pytest.raises(FloquetModeError, match="双曲"):
            floquet_modes(orbit, dynamics)

    def test_three_dimensional_orbit_rejected(self) -> None:
        """三维周期轨道（L2 halo）不在本模块形态内，按平面限制确定性报错。

        三维轨道的单值矩阵面内块与法向块耦合，两对特征向量的 z、vz 支撑
        互相混杂，按支撑排序贴面内与法向标签会静默出错，因此直接拒绝。
        """
        orbit = design_halo(2, 5000.0)
        dynamics = CR3BP_Dynamics(earth_moon_system())
        with pytest.raises(FloquetModeError, match="平面"):
            floquet_modes(orbit, dynamics)

    def test_grid_must_span_exactly_one_period(self, dro21) -> None:
        """自定义网格必须严格递增、起于相位原点、止于原点加周期。"""
        orbit, dynamics = dro21
        period = float(orbit.period)
        with pytest.raises(FloquetModeError, match="网格"):
            floquet_modes(orbit, dynamics, times=np.linspace(0.0, 0.9 * period, 11))
        with pytest.raises(FloquetModeError, match="网格"):
            floquet_modes(orbit, dynamics, times=np.linspace(0.1, period, 11))
        with pytest.raises(FloquetModeError, match="网格"):
            floquet_modes(orbit, dynamics, times=np.linspace(0.0, period, 11)[::-1])
