"""Transfer 创建与配置契约测试 + DRO→RO 搜索集成（生成轨道，#627）。

轨道来源：DRO 由统一周期轨道真值种子微分修正生成（与
tests/algorithm/conftest.py 一致）；RO 由 ``design_ro`` 按 3:1 共振生成
（替代历史外部数据文件 output/dro|ro/*.json——文件缺失即 skip 的时代
结束，测试自给自足）。``optimize()`` 全链 NLP 测试维持既有维护决策
（不进套件，此前因 rtol=1e-12 全链超 ADR 0037 预算移除）：筛选级容差
下从粗网格候选出发，SLSQP 在 8+ 组参数组合（松弛/非松弛 × 多候选 ×
多转移时间上限）中均报 "Positive directional derivative for
linesearch" 不可行，链路收敛研究不进 pytest。松弛速度约束的方向契约
由 ``TestRelaxedVelocityConstraintSign`` 单元级钉住。
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

    state = np.array([0.79188556619742, 0.0, 0.0, 0.0, 0.536819842572739, 0.0])
    seed = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    seed.period = 3.472535773770595
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(state[0])
    result = corrector.iterate_correction(seed, verbose=False)
    assert result.orbit is not None, "DRO 种子修正未收敛"
    return result.orbit


@pytest.fixture(scope="module")
def ro_orbit(dynamics):
    """RO 到达轨道：研究级容差下由 ``design_ro`` 生成 3:1 共振轨道。"""
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics
    from e2m2e.algorithm.family.cr3bp_orbits import design_ro

    # 精确成员的闭合守卫为 1e-6；本模块的筛选级容差在 3:1 长弧上
    # 的积分误差约为 5e-6，不能用于构造 RO。搜索阶段仍使用 dynamics。
    design_dynamics = CR3BP_Dynamics(system=dynamics.system)
    return design_ro(3, 1, dynamics=design_dynamics)


class TestDroRoTransferIntegration:
    """DRO→RO 搜索集成（生成轨道，无外部数据文件依赖）。"""

    def test_search_finds_feasible_candidates(self, dynamics, dro_orbit, ro_orbit):
        """小网格搜索产出局部最小可行候选（DRO→RO 几何存在性）。"""
        from e2m2e.algorithm.transfer import (
            DEFAULT_MIN_DISTANCE_THRESHOLD_DU,
            TransferSearch,
        )

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
        assert best.min_distance_orbit_idx is not None


class TestRelaxedVelocityConstraintSign:
    """松弛速度约束的 scipy 不等式方向契约（回归钉）。"""

    def test_relaxed_velocity_inequality_is_feasible_iff_cos_above_tolerance(
        self, dynamics, dro_orbit, ro_orbit, monkeypatch
    ):
        """ineq 约束 fun ≥ 0 与 "cos_angle ≥ cos(tol)" 同向。

        符号写反时 SLSQP 会把速度夹角推离平行且照样报“收敛”，事后
        按正确语义报告的违反量可达 ~2（5.9.6 前的真实缺陷）。
        """
        from types import SimpleNamespace

        from e2m2e.algorithm.transfer import nlp_scipy
        from e2m2e.algorithm.transfer.config import TransferConfig
        from e2m2e.algorithm.transfer.transfer_optimization import DROTRONLPOptimizer

        config = TransferConfig(nlp_use_relaxed_velocity=True, nlp_velocity_angle_tol=0.05)
        optimizer = DROTRONLPOptimizer(
            dynamics.system,
            dynamics,
            departure_orbit=dro_orbit,
            arrival_orbit=ro_orbit,
            config=config,
        )
        captured: dict = {}

        def fake_minimize(fun, y0, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=False, status=9, x=y0, message="stub")

        monkeypatch.setattr(nlp_scipy, "minimize", fake_minimize)
        nlp_scipy.solve_with_scipy(
            optimizer,
            use_relaxed_velocity_constraint=True,
            velocity_angle_constraint=0.05,
        )
        inequalities = [c for c in captured["constraints"] if c["type"] == "ineq"]
        assert len(inequalities) == 1
        constraint = inequalities[0]["fun"]
        cos_tol = float(np.cos(0.05))
        optimizer._compute_cos_angle = lambda y: cos_tol + 1e-3
        assert constraint(np.zeros(3)) > 0.0, "夹角在容差内必须落在可行侧"
        optimizer._compute_cos_angle = lambda y: cos_tol - 1e-3
        assert constraint(np.zeros(3)) < 0.0, "夹角超出容差必须落在不可行侧"
