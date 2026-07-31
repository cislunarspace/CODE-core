"""ThreeBodyLambert 三体打靶测试。

验证：
- 周期轨道（3:1 DRO）上两相位点间转移：以真实相位差为 tof，解应收敛且
  出发/到达脉冲为小量（端点本就在同一轨道上）
- guess="orbit"（直接取出发速度为初猜）同样收敛
- Lyapunov → Halo 交会场景收敛
"""

import copy

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.halo_initial_guess import _compute_gamma, compute_halo_initial_guess
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.algorithm.transfer import StateTerminal, ThreeBodyLambert, TransferSolution
from e2m2e.data.types.orbit import Orbit

# 地月系统质量参数（与 tests/algorithms/conftest.py 一致）
MU = 1.21506683e-2
DU = 384405.0  # km

_LYAP_AX = 0.01  # L1 Lyapunov 面内振幅（无量纲）
_HALO_Z0 = 0.001  # L1 Halo z 振幅（无量纲；Richardson 初猜直接修正仅小振幅可收敛）


def _make_system():
    from e2m2e.algorithm.dynamics import CR3BP_System

    return CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()


def _make_l1_lyapunov_orbit(system) -> Orbit:
    """生成地月 L1 小振幅 Lyapunov 轨道（与 tests/algorithms/test_manifolds.py 同法）。"""
    dynamics = CR3BP_Dynamics(system)
    gamma = _compute_gamma(MU, 1)
    x_l1 = 1 - MU - gamma

    jacobian = dynamics.compute_jacobian_A([x_l1, 0, 0, 0, 0, 0])
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    idx = next(k for k, lam in enumerate(eigenvalues) if abs(lam.real) < 1e-8 and lam.imag > 1e-8)
    mode = np.real(eigenvectors[:, idx] * np.exp(-1j * np.angle(eigenvectors[0, idx])))
    mode /= mode[0]

    x0 = x_l1 + _LYAP_AX
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0)
    seed = Orbit(states=[[x0, 0, 0, 0, mode[4] * _LYAP_AX, 0]], times=[0], system=system)
    seed.period = 2 * np.pi / eigenvalues[idx].imag
    orbit = corrector.iterate_correction(seed, verbose=False)
    assert orbit is not None, "L1 Lyapunov 轨道微分修正失败"
    orbit.system = system
    return orbit


def _make_l1_halo_orbit(system) -> Orbit:
    """生成地月 L1 小振幅 Halo 轨道（Richardson 初猜 + 微分修正）。"""
    dynamics = CR3BP_Dynamics(system)
    guess = compute_halo_initial_guess(MU, _HALO_Z0, L=1, halo_class=0)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0=_HALO_Z0, libration_point=1)
    seed = Orbit(
        states=[[guess["x0"], 0.0, _HALO_Z0, 0.0, guess["vy0"], 0.0]],
        times=[0.0],
        system=system,
    )
    seed.period = guess["T_half"] * 2
    orbit = corrector.iterate_correction(initial_guess=seed, verbose=False)
    assert orbit is not None, "L1 Halo 轨道微分修正失败"
    orbit.system = system
    return orbit


@pytest.fixture(scope="session")
def _orbits_cached():
    """会话级缓存 Lyapunov 与 Halo 轨道（修正需多次 STM 传播）。"""
    system = _make_system()
    return system, _make_l1_lyapunov_orbit(system), _make_l1_halo_orbit(system)


@pytest.fixture
def orbits(_orbits_cached):
    system, lyap, halo = _orbits_cached
    return system, copy.deepcopy(lyap), copy.deepcopy(halo)


class TestPeriodicOrbitTransfer:
    """周期轨道上两相位点间转移：解须收敛且脉冲为小量"""

    def test_dro_phase_points_converge(self, earth_moon_system, earth_moon_dynamics):
        """3:1 DRO 上取两点，tof 取真实相位差，打靶收敛且 ΔV 接近零"""
        x0 = np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])
        dt = 0.5  # 无量纲相位差
        result = earth_moon_dynamics.propagate(x0, (0.0, dt), t_eval=np.linspace(0, dt, 50))
        x1 = result["states"][-1]

        tu = earth_moon_system.characteristic_time
        s0 = earth_moon_system.dimensionless_to_physical(x0)
        s1 = earth_moon_system.dimensionless_to_physical(x1)

        shooter = ThreeBodyLambert(earth_moon_dynamics)
        sol = shooter.solve(StateTerminal(s0, 0.0), StateTerminal(s1, dt * tu), dt * tu)

        assert isinstance(sol, TransferSolution)
        assert sol.converged, sol.message
        assert sol.n_iter <= shooter.max_iterations
        # 端点在同一轨道上：出发/到达脉冲为小量
        assert sol.arcs[0].delta_v < 1e-6  # km/s
        assert sol.arrival_delta_v < 1e-6
        # 末端位置命中（1e-8 无量纲 ≈ 4e-3 km）
        pos_err = np.linalg.norm(sol.arcs[0].states[-1][:3] - s1[:3])
        assert pos_err < 1e-2  # km

    def test_orbit_guess_converges(self, earth_moon_system, earth_moon_dynamics):
        """guess='orbit' 直接以出发速度为初猜，轨道内转移一次迭代即收敛"""
        x0 = np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])
        dt = 0.5
        result = earth_moon_dynamics.propagate(x0, (0.0, dt), t_eval=np.linspace(0, dt, 50))
        x1 = result["states"][-1]

        tu = earth_moon_system.characteristic_time
        s0 = earth_moon_system.dimensionless_to_physical(x0)
        s1 = earth_moon_system.dimensionless_to_physical(x1)

        shooter = ThreeBodyLambert(earth_moon_dynamics)
        sol = shooter.solve(
            StateTerminal(s0, 0.0), StateTerminal(s1, dt * tu), dt * tu, guess="orbit"
        )
        assert sol.converged, sol.message
        assert sol.arcs[0].delta_v < 1e-6

    def test_invalid_guess_raises(self, earth_moon_dynamics):
        """非法 guess 参数报错"""
        shooter = ThreeBodyLambert(earth_moon_dynamics)
        with pytest.raises(ValueError, match="guess"):
            shooter.solve(
                StateTerminal([1.0, 0, 0, 0, 1.0, 0], 0.0),
                StateTerminal([0.0, 1.0, 0, -1.0, 0, 0], 1.0),
                1000.0,
                guess="bad",
            )

    def test_negative_tof_raises(self, earth_moon_dynamics):
        """非正 tof 报错"""
        shooter = ThreeBodyLambert(earth_moon_dynamics)
        with pytest.raises(ValueError, match="tof"):
            shooter.solve(
                StateTerminal([1.0, 0, 0, 0, 1.0, 0], 0.0),
                StateTerminal([0.0, 1.0, 0, -1.0, 0, 0], 1.0),
                -1.0,
            )


class TestRendezvous:
    """Lyapunov / Halo 间交会场景"""

    def test_lyapunov_to_halo_converges(self, orbits):
        """L1 Lyapunov 上一点 → L1 Halo 上一点，打靶收敛并命中末端位置"""
        system, lyap, halo = orbits
        dynamics = CR3BP_Dynamics(system)

        # Halo 上取半周期处相位点作为目标
        dt_target = 0.5 * halo.period
        result = dynamics.propagate(
            halo.states[0], (0.0, dt_target), t_eval=np.linspace(0, dt_target, 50)
        )
        x_target = result["states"][-1]

        tof = 1.0  # 无量纲
        tu = system.characteristic_time
        s0 = system.dimensionless_to_physical(lyap.states[0])
        s1 = system.dimensionless_to_physical(x_target)

        shooter = ThreeBodyLambert(dynamics)
        sol = shooter.solve(StateTerminal(s0, 0.0), StateTerminal(s1, tof * tu), tof * tu)

        assert sol.converged, sol.message
        pos_err = np.linalg.norm(sol.arcs[0].states[-1][:3] - s1[:3])
        assert pos_err < 1e-2  # km
