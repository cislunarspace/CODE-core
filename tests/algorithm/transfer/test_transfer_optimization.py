"""Transfer 创建与配置契约测试 + DRO→RO 全链集成（生成轨道，#627）。

轨道来源：DRO 由标准种子微分修正生成（Cui et al. 2025 种子，与
tests/algorithm/conftest.py 一致）；RO 由 ``design_ro`` 按 3:1 共振生成
（替代历史外部数据文件 output/dro|ro/*.json——文件缺失即 skip 的时代
结束，测试自给自足）。``optimize()`` 全链 NLP 测试曾按维护决策移除
（rtol=1e-12 研究级容差超 ADR 0037 预算）；本文件的集成用例用筛选级
容差（1e-9）与小网格，把搜索→优化链路压回单测预算。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.orchestration


class TestTransferCreation:
    """Test Transfer class instantiation and configuration."""

    def test_transfer_creation_with_dynamics(self, dynamics):
        """Transfer should be created with a dynamics instance."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)

        assert transfer.dynamics is not None
        assert transfer.departure_orbit is None
        assert transfer.arrival_orbit is None
        assert transfer.result is None

    def test_transfer_set_orbits(self, dynamics, dro_orbit, ro_orbit):
        """Transfer.set_orbit() should accept start (departure) and end (arrival) orbits."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        assert transfer.departure_orbit is dro_orbit
        assert transfer.arrival_orbit is ro_orbit

    def test_transfer_config_fields(self, dynamics):
        """Transfer.config should expose optimization configuration."""
        from e2m2e.algorithm.transfer import Transfer, TransferConfig

        transfer = Transfer(dynamics)

        assert hasattr(transfer, "config")
        assert isinstance(transfer.config, TransferConfig)
        assert transfer.config.nlp_alpha_min == 0.5
        assert transfer.config.nlp_alpha_max == 2.5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dynamics():
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
    from e2m2e.data.constants import Datum

    system = CR3BP_System(mu=Datum.DE421.mu, primary="earth", secondary="moon")
    dyn = CR3BP_Dynamics(system=system)
    dyn.integrator = "DOP853"
    # 筛选级容差（ADR 0037 决策 3）：搜索/优化链路在 1e-9 下秒级完成
    dyn.rtol = 1e-9
    dyn.atol = 1e-9
    dyn.max_step = 0.05
    return dyn


@pytest.fixture(scope="module")
def dro_orbit(dynamics):
    """DRO 出发轨道：标准种子微分修正生成（module 级共享，只读）。"""
    from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
    from e2m2e.data.types.orbit import Orbit

    state = np.array([0.79188556619742, 0.0, 0.0, 0.0, 0.573665890385585, 0.0])
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = 6.307498
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(state[0])
    result = corrector.iterate_correction(seed, verbose=False)
    assert result.orbit is not None, "DRO 种子修正未收敛"
    return result.orbit


@pytest.fixture(scope="module")
def ro_orbit(dynamics):
    """RO 到达轨道：``design_ro`` 生成的 3:1 共振轨道（module 级共享，只读）。"""
    from e2m2e.algorithm.family.cr3bp_orbits import design_ro

    return design_ro(3, 1, dynamics=dynamics)


class TestDroRoTransferIntegration:
    """DRO→RO 搜索→优化全链集成（生成轨道，无外部数据文件依赖）。"""

    def test_search_optimize_chain_converges(self, dynamics, dro_orbit, ro_orbit):
        """小网格搜索出可行候选，NLP 从最优候选出发收敛且约束达标。"""
        from e2m2e.algorithm.transfer import (
            DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
            Transfer,
            TransferSearch,
        )
        from e2m2e.data.templates import ConvergenceState

        searcher = TransferSearch(dynamics)
        candidates = searcher.search(
            departure_orbit=dro_orbit,
            arrival_orbit=ro_orbit,
            alpha_min=0.5,
            alpha_max=1.0,
            n_alpha=10,
            n_departure=40,
            max_transfer_time=8.0,
            integration_dt=0.02,
            intersection_threshold=1e-3,
            min_distance_threshold=DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
            collision_earth_radius=5e-4,
            collision_moon_radius=3e-4,
            verbose=False,
        )
        feasible = [
            candidate
            for candidate in candidates
            if candidate.local_minimum_found and candidate.min_distance < 1e-3
        ]
        assert feasible, "搜索网格未产出局部最小可行候选（DRO→RO 几何应存在）"
        best = min(feasible, key=lambda candidate: candidate.min_distance)

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)
        assert best.min_distance_orbit_idx is not None
        result = transfer.optimize(
            initial_guess={
                "alpha": float(best.alpha),
                "transfer_time": float(best.transfer_time),
                "t_ins": float(ro_orbit.times[best.min_distance_orbit_idx]),
            },
            alpha_range=(0.5, 1.2),
            departure_state=np.asarray(best.departure_state, dtype=float),
        )

        assert result.status is ConvergenceState.CONVERGED, result.message
        # 位置连续 + 速度平行约束的剩余违反量（无量纲）
        assert result.constraints_violation < 1e-6
        assert np.isfinite(result.delta_v1) and np.isfinite(result.delta_v2)
        assert result.total_delta_v > 0.0
