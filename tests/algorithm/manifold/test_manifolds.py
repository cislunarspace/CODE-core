"""InvariantManifold 不变流形计算测试。

用地月 L1 小振幅 Lyapunov 轨道（线性化初猜 + 微分修正生成）验证：
- 种子生成（形状、只存首点的轨道、ε 线性缩放）
- 稳定流形反向积分后末端比种子更靠近主天体
"""

import copy

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.halo_initial_guess import _compute_gamma
from e2m2e.algorithm.manifold import InvariantManifold, ManifoldKind, PoincareSection
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.constants import Datum
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


# 地月系统质量参数（DE421 基准）
MU = Datum.DE421.mu
DU = 384405.0  # km

# L1 Lyapunov 面内振幅（无量纲，距 L1 的 x 偏移）
_LYAP_AX = 0.01


def _make_l1_lyapunov_orbit() -> Orbit:
    """生成地月 L1 小振幅 Lyapunov 轨道。

    初猜来自 L1 处线性化中心模态特征向量，再由微分修正闭合为精确周期轨道。
    """
    from e2m2e.algorithm.dynamics import CR3BP_System

    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system)

    gamma = _compute_gamma(MU, 1)
    x_l1 = 1 - MU - gamma

    # L1 处雅可比的面内中心模态（纯虚特征值）给出 x 轴穿越点的 (x0, vy0) 关系
    jacobian = dynamics.compute_jacobian_A([x_l1, 0, 0, 0, 0, 0])
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    idx = next(k for k, lam in enumerate(eigenvalues) if abs(lam.real) < 1e-8 and lam.imag > 1e-8)
    mode = np.real(eigenvectors[:, idx] * np.exp(-1j * np.angle(eigenvectors[0, idx])))
    mode /= mode[0]

    x0 = x_l1 + _LYAP_AX
    vy0 = mode[4] * _LYAP_AX
    period_guess = 2 * np.pi / eigenvalues[idx].imag

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0)
    seed = Orbit(states=[[x0, 0, 0, 0, vy0, 0]], times=[0], system=system)
    seed.period = period_guess
    result = corrector.iterate_correction(seed, verbose=False)
    assert result.orbit is not None, "L1 Lyapunov 轨道微分修正失败"
    orbit = result.orbit
    orbit.system = system
    return orbit


@pytest.fixture(scope="session")
def _l1_lyapunov_cached() -> Orbit:
    """会话级缓存修正后的 L1 Lyapunov 轨道（修正需多次 STM 传播）。"""
    return _make_l1_lyapunov_orbit()


@pytest.fixture
def l1_lyapunov(_l1_lyapunov_cached) -> Orbit:
    """每个测试取深拷贝，避免相互污染。"""
    return copy.deepcopy(_l1_lyapunov_cached)


@pytest.fixture
def stable_manifold(l1_lyapunov) -> InvariantManifold:
    """L1 Lyapunov 稳定流形（ε = 50 km / DU）。"""
    return InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "-", 50.0 / DU)


class TestManifoldCreation:
    """测试不变流形构造参数校验"""

    def test_requires_period(self, l1_lyapunov):
        """周期未知的轨道不能生成流形"""
        l1_lyapunov.period = None
        with pytest.raises(ValueError, match="周期未知"):
            InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "+", 1e-4)

    def test_requires_system(self, l1_lyapunov):
        """未关联 system 的轨道不能生成流形"""
        l1_lyapunov.system = None
        with pytest.raises(ValueError, match="system"):
            InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "+", 1e-4)

    def test_invalid_branch(self, l1_lyapunov):
        """非法 branch 报错"""
        with pytest.raises(ValueError, match="branch"):
            InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "x", 1e-4)

    def test_invalid_epsilon(self, l1_lyapunov):
        """非正 epsilon 报错"""
        with pytest.raises(ValueError, match="epsilon"):
            InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "+", -1e-4)


class TestSeeds:
    """测试流形种子生成"""

    def test_seeds_shape(self, stable_manifold):
        """种子形状为 (n_points, 6)"""
        seeds = stable_manifold.seeds(12)
        assert seeds.shape == (12, 6)

    def test_seeds_position_offset_equals_epsilon(self, stable_manifold):
        """首点种子的位置偏移幅度恰为 ε（位置部分归一化）

        n_points=1 时相位点即轨道首点，种子偏移可直接对照。
        """
        seeds = stable_manifold.seeds(1)
        offset = np.linalg.norm(seeds[0, :3] - stable_manifold.orbit.states[0, :3])
        assert offset == pytest.approx(stable_manifold.epsilon, rel=1e-12)

    def test_seeds_from_single_point_orbit(self, l1_lyapunov):
        """只存首点的轨道（Orbit(states=[x0], times=[0])）也能生成种子"""
        single_point = Orbit(states=[l1_lyapunov.states[0]], times=[0], system=l1_lyapunov.system)
        single_point.period = l1_lyapunov.period
        manifold = InvariantManifold(single_point, ManifoldKind.UNSTABLE, "+", 1e-4)
        seeds = manifold.seeds(6)
        assert seeds.shape == (6, 6)
        assert np.all(np.isfinite(seeds))

    def test_epsilon_halving_halves_offset(self, l1_lyapunov):
        """ε 减半时种子初始偏差减半（线性区验证）"""
        epsilon = 100.0 / DU
        m_full = InvariantManifold(l1_lyapunov, ManifoldKind.UNSTABLE, "+", epsilon)
        m_half = InvariantManifold(l1_lyapunov, ManifoldKind.UNSTABLE, "+", epsilon / 2)
        x0 = l1_lyapunov.states[0]
        off_full = np.linalg.norm(m_full.seeds(1)[0] - x0)
        off_half = np.linalg.norm(m_half.seeds(1)[0] - x0)
        assert off_full / off_half == pytest.approx(2.0, rel=1e-10)

    def test_branches_are_symmetric(self, l1_lyapunov):
        """± 分支首点种子关于轨道状态对称"""
        epsilon = 1e-4
        m_plus = InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "+", epsilon)
        m_minus = InvariantManifold(l1_lyapunov, ManifoldKind.STABLE, "-", epsilon)
        x0 = l1_lyapunov.states[0]
        np.testing.assert_allclose(
            m_plus.seeds(1)[0] - x0, -(m_minus.seeds(1)[0] - x0), rtol=1e-12, atol=1e-15
        )


class TestPropagate:
    """测试流形弧传播"""

    def test_stable_manifold_approaches_primary(self, stable_manifold):
        """稳定流形反向积分后末端比种子更靠近主天体（定性基准）

        地月 L1 Lyapunov 稳定流形 "-" 分支反向积分后走向地球；
        主天体位于会合系 x = -μ。
        """
        earth_pos = np.array([-MU, 0.0, 0.0])
        tube = stable_manifold.propagate(4.0)
        assert len(tube.trajectories) > 0

        for arc in tube.trajectories:
            d_seed = np.linalg.norm(arc.states[0, :3] - earth_pos)
            d_end = np.linalg.norm(arc.states[-1, :3] - earth_pos)
            assert d_end < d_seed, (
                f"稳定流形末端应比种子更靠近主天体: d_end={d_end:.4f}, d_seed={d_seed:.4f}"
            )

    def test_backward_integration_times_decrease(self, stable_manifold):
        """稳定流形反向积分，时间序列递减"""
        tube = stable_manifold.propagate(2.0)
        for arc in tube.trajectories:
            assert arc.times[-1] < arc.times[0]

    def test_unstable_manifold_forward(self, l1_lyapunov):
        """不稳定流形正向积分，时间序列递增"""
        manifold = InvariantManifold(l1_lyapunov, ManifoldKind.UNSTABLE, "+", 50.0 / DU)
        tube = manifold.propagate(1.0)
        for arc in tube.trajectories:
            assert arc.times[-1] > arc.times[0]

    def test_tube_metadata(self, stable_manifold):
        """流形管携带轨道引用与流形参数"""
        tube = stable_manifold.propagate(1.0)
        assert tube.orbit is stable_manifold.orbit
        assert tube.kind is ManifoldKind.STABLE
        assert tube.branch == "-"
        assert tube.epsilon == stable_manifold.epsilon

    def test_propagate_with_section_truncates(self, stable_manifold):
        """给定截面时流形弧在首次穿越处截断，末点在截面上"""
        section = PoincareSection.periapsis("earth", stable_manifold.orbit.system)
        tube = stable_manifold.propagate(4.0, section=section)
        n_truncated = sum(1 for arc in tube.trajectories if abs(section(arc.states[-1])) < 1e-8)
        assert n_truncated > 0, "应至少有一条流形弧被截面截断"
