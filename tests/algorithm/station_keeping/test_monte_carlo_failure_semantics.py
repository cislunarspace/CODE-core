"""蒙特卡洛控制律失败样本语义测试（#352）。

控制律未产出机动（``compute_maneuver`` 返回 None）时，样本必须计为
失败：此前代码把 ``failed_k`` 置 False 当成功处理，失败样本的 Δv 不进
统计、失败率被系统性压低（ADR 0020 决策 1 红线：禁止把失败藏进成功
统计）。

用确定性假组件（返回 None 的控制律、自由运动假传播器、假工厂/标称视
图）跑 ``SingleSampleSimulation.run()``，不依赖 SPICE 与 Rust 扩展。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.station_keeping.error_models import (
    BoxMullerSampler,
    NavigationErrorModel,
    SrpErrorModel,
    ThrustExecutionError,
)
from e2m2e.algorithm.station_keeping.monte_carlo import (
    MonteCarloResult,
    SingleSampleSimulation,
)
from e2m2e.data.types import SKStatistic

pytestmark = pytest.mark.orchestration

_SECONDS_PER_DAY = 86400.0


class NoManeuverLaw:
    """控制律假实现：永不产出机动（返回 None，模拟不收敛/未找到穿越点）。"""

    def compute_maneuver(self, state0, t0, *, propagator, nominal):
        return None


class FixedManeuverLaw:
    """控制律假实现：返回固定 Δv（0.01 km/s = 10 m/s，落在执行误差段内）。"""

    def compute_maneuver(self, state0, t0, *, propagator, nominal):
        return np.array([0.01, 0.0, 0.0])


class FreeMotionPropagator:
    """自由运动假传播器：r(t) = r0 + v·(t-t0)。"""

    def propagate(self, state0, t0, t_eval):
        t_eval = np.asarray(t_eval, dtype=float)
        state0 = np.asarray(state0, dtype=float)
        dt = t_eval - t0
        states = np.empty((len(t_eval), 6))
        states[:, :3] = state0[:3] + np.outer(dt, state0[3:])
        states[:, 3:] = state0[3:]
        return states


class FakeFactory:
    """传播器工厂假实现：任何力配置都返回自由运动传播器。"""

    def make(self, force_config, srp_cr_scale=1.0):
        return FreeMotionPropagator()


class NominalStub:
    """标称轨道视图假实现（t_start 固定、常值状态）。"""

    def __init__(self):
        self._state = np.zeros(6)

    @property
    def t_start(self):
        return 0.0

    def state_at(self, t):
        return self._state.copy()


def _build_sim(law) -> SingleSampleSimulation:
    """纯轨道控制（无角动量管理）单样本仿真，单个控制事件。"""
    return SingleSampleSimulation(
        nominal=NominalStub(),
        law=law,
        control_interval_sec=30.0 * _SECONDS_PER_DAY,
        num_controls=3,  # events = [t0 + 1·间隔]，一次控制
        output_step_sec=_SECONDS_PER_DAY,
        nav_error=NavigationErrorModel(),
        thrust_error=ThrustExecutionError(),
        srp_error=SrpErrorModel(),
        thrust_total_mps=1000.0,
        factory=FakeFactory(),
        force_config_ctrl={},
        force_config_true={},
        sampler=BoxMullerSampler(seed=7),
    )


class TestNoManeuverSampleIsFailure:
    def test_control_law_none_flags_sample_failed(self):
        """控制律返回 None（未产出机动）→ 样本标记失败（修复前被当作成功）。"""
        result = _build_sim(NoManeuverLaw()).run()
        assert result.failed is True

    def test_no_maneuver_records_no_maneuver_row(self):
        """失败早退：机动表只有 t0 参考行（Δv=0），无虚假机动记录。"""
        result = _build_sim(NoManeuverLaw()).run()
        assert result.maneuver_times.tolist() == [0.0]
        assert result.maneuver_dv_mps.tolist() == [0.0]

    def test_failed_sample_excluded_from_statistics(self):
        """失败样本不进 SK_STATISTIC 统计行（不把失败计成功）。"""
        sample = _build_sim(NoManeuverLaw()).run()
        assert sample.failed is True
        mc = MonteCarloResult(
            total_delta_v=np.array([sample.total_delta_v_mps]),
            max_delta_v=np.array([sample.max_delta_v_mps]),
            failed_mask=np.array([sample.failed]),
            num_failed=1,
        )
        stat = mc.sk_statistic()
        assert isinstance(stat, SKStatistic)
        assert stat.num_failed == 1
        assert stat.rows.shape[0] == 0


class TestNormalManeuverNotAffected:
    def test_control_law_returns_maneuver_keeps_success(self):
        """回归：控制律正常产出机动时样本仍为成功（不误伤 happy path）。"""
        result = _build_sim(FixedManeuverLaw()).run()
        assert result.failed is False

    def test_orbital_maneuver_delta_v_accumulated_without_momentum_management(self):
        """回归（#261 引入）：无角动量管理时机动 Δv 必须累计并记录。

        ``is_orbital`` 在 ``has_mm=False`` 时恒为 False 是 #261 的回归：
        机动 Δv 既不进统计也不施加于真实轨道（实测 10 m/s 输出统计为 0），
        统计系统性偏低。修复后机动进入机动表、累计进总 Δv。
        """
        result = _build_sim(FixedManeuverLaw()).run()
        assert result.failed is False
        assert result.total_delta_v_mps > 0.0
        assert len(result.maneuver_dv_mps) == 2  # t0 参考行 + 一次机动
        assert result.maneuver_dv_mps[1] > 0.0
