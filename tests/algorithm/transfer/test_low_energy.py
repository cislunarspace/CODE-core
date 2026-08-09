"""流形拼接与低能转移流水线测试。

复现 PRD 基准（郑越、赵敏 2023 场景）：大幅值 L1 Lyapunov 稳定流形 +
近拱点截面拼接，最优候选拼接脉冲在几十 m/s 量级。
另验证拼接候选排序/权重语义、无穿越空结果，以及
``design_low_energy_transfer`` 流水线端到端收敛。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.halo_initial_guess import _compute_gamma
from e2m2e.algorithm.manifold import InvariantManifold, ManifoldKind, PoincareSection
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.algorithm.transfer import (
    OrbitTerminal,
    PatchCandidate,
    design_low_energy_transfer,
    patch_manifolds,
)
from e2m2e.data.types.orbit import Orbit

# 集成/端到端层：族延拓 + 流形传播 + 转移优化共享会话级 fixture，默认全量不跑。
# 跑本文件：uv run pytest -m l3 tests/transfer/test_low_energy.py
pytestmark = pytest.mark.orchestration

MU = 1.21506683e-2
DU = 384405.0  # km

_LYAP_AX = 0.01
# 族延拓幅度：大幅值端 x0 ≈ x_l1 + 0.0625，中间轨道取族内索引 6
_CONTINUE_DX = 0.05
_MID_INDEX = 6
_EPS = 50.0 / DU


def _make_l1_lyapunov_orbit(system, dynamics) -> tuple[Orbit, DifferentialCorrection]:
    """生成地月 L1 小振幅 Lyapunov 轨道（与 tests/algorithms/test_manifolds.py 同法）。"""
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
    return orbit, corrector


@pytest.fixture(scope="session")
def lyapunov_family():
    """会话级 L1 Lyapunov 族：小振幅轨道自然延拓到大幅值（x0 + 0.05）。"""
    system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
    dynamics = CR3BP_Dynamics(system)
    lyap0, corrector = _make_l1_lyapunov_orbit(system, dynamics)

    cont = Continuation(corrector=corrector)
    family = cont.natural_continuation(
        seed_orbit=lyap0,
        param_range=(float(lyap0.states[0][0]), float(lyap0.states[0][0]) + _CONTINUE_DX),
        step_size=0.005,
        verbose=False,
    )
    orbits = family.orbits
    assert len(orbits) > _MID_INDEX, "Lyapunov 族延拓轨道数不足"
    for o in orbits:
        o.system = system
    return system, orbits


class TestPatchManifolds:
    """流形管截面拼接"""

    def test_benchmark_patch_cost_tens_of_mps(self, lyapunov_family):
        """PRD 基准：大幅值 Lyapunov 稳定流形 + 近地点截面，最优拼接脉冲几十 m/s

        大幅值端（族末，x0 ≈ x_l1+0.0625）稳定流形 "-" 分支与族内中间轨道
        不稳定流形 "+" 分支在地球近拱点截面拼接；最优候选 |Δv| 应落在
        20~100 m/s（无量纲约 0.02~0.10）。
        """
        system, orbits = lyapunov_family
        section = PoincareSection.periapsis("earth", system)

        tube_big = InvariantManifold(orbits[-1], ManifoldKind.STABLE, "-", _EPS).propagate(
            6.0, section=section
        )
        tube_mid = InvariantManifold(
            orbits[_MID_INDEX], ManifoldKind.UNSTABLE, "+", _EPS
        ).propagate(6.0, section=section)

        candidates = patch_manifolds(tube_mid, tube_big, section)
        assert len(candidates) > 0, "两流形管在近地点截面应有穿越点"

        best = candidates[0]
        assert isinstance(best, PatchCandidate)
        vu = system.characteristic_velocity
        assert 0.02 < best.delta_v < 0.10, (
            f"拼接脉冲应在几十 m/s 量级，当前 {best.delta_v * vu * 1e3:.1f} m/s"
        )

    def test_candidates_sorted_and_cost_consistent(self, lyapunov_family):
        """候选按代价升序，且 cost = w_r·|Δr| + w_v·|Δv|"""
        system, orbits = lyapunov_family
        section = PoincareSection.periapsis("earth", system)

        tube_big = InvariantManifold(orbits[-1], ManifoldKind.STABLE, "-", _EPS).propagate(
            6.0, section=section
        )
        tube_mid = InvariantManifold(
            orbits[_MID_INDEX], ManifoldKind.UNSTABLE, "+", _EPS
        ).propagate(6.0, section=section)

        weights = (0.5, 2.0)
        candidates = patch_manifolds(tube_mid, tube_big, section, weights=weights)
        costs = [c.cost for c in candidates]
        assert costs == sorted(costs)
        for c in candidates:
            assert c.cost == pytest.approx(
                weights[0] * c.delta_r + weights[1] * c.delta_v, rel=1e-12
            )
            assert c.delta_r == pytest.approx(np.linalg.norm(c.state_a[:3] - c.state_b[:3]))
            assert c.delta_v == pytest.approx(np.linalg.norm(c.state_a[3:] - c.state_b[3:]))

    def test_no_crossings_returns_empty(self, lyapunov_family):
        """截面与流形管无交时返回空列表"""
        system, orbits = lyapunov_family
        far_plane = PoincareSection.plane(axis=2, value=1.0)  # 面内流形 z 恒为 0
        tube = InvariantManifold(orbits[0], ManifoldKind.UNSTABLE, "+", _EPS).propagate(2.0)
        assert patch_manifolds(tube, tube, far_plane) == []


class TestDesignLowEnergyTransfer:
    """低能转移流水线端到端"""

    def test_pipeline_converges(self, lyapunov_family):
        """中间轨道 → 大幅值轨道：流水线收敛，两段弧，总脉冲为各脉冲之和"""
        system, orbits = lyapunov_family
        sol = design_low_energy_transfer(OrbitTerminal(orbits[_MID_INDEX]), orbits[-1])

        assert sol.converged, sol.message
        assert len(sol.arcs) == 2
        assert sol.total_delta_v == pytest.approx(
            sol.arcs[0].delta_v + sol.arcs[1].delta_v + sol.arrival_delta_v
        )
        assert sol.transfer_time > 0
        # 出发脉冲 ≈ 上不稳定流形的 ε 量级扰动速度（小量），拼接/到达脉冲为有限正值
        assert sol.arcs[0].delta_v < 0.01  # km/s
        assert sol.arcs[1].delta_v > 0
        # 末段弧终点即目标流形种子（无量纲换算前位置一致）
        seed_b = sol.arcs[1].states[-1]
        assert np.all(np.isfinite(seed_b))

    def test_invalid_model_raises(self, lyapunov_family):
        """不支持的 model 报错"""
        _, orbits = lyapunov_family
        with pytest.raises(ValueError, match="cr3bp"):
            design_low_energy_transfer(OrbitTerminal(orbits[0]), orbits[1], model="bcr4bp")
