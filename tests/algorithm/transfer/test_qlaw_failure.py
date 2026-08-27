"""Q-law 失败语义测试。

两条红线：

1. **步长塌缩**：积分步长缩到机器精度地板时抛 ``PropagationFailure``，
   不重置回原步长空转、也不用未验收的中间态拼控制律返回（统计/结果
   谎报成功）。
2. **``_resolve_mu``**：中心体 μ 查询失败时抛异常，不静默替换为地球 μ
   （非地球系统会被用错动力学参数）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig
from e2m2e.algorithm.transfer.qlaw import _resolve_mu, qlaw_guess
from e2m2e.exceptions import PropagationFailure

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

MU = 398600.435507  # km³/s²，地球


class TestStepCollapseRaisesPropagationFailure:
    def test_extreme_initial_state_raises_propagation_failure(self):
        """初态贴地心（重力加速度 1/r² 发散）→ 步长塌缩 → 抛 PropagationFailure。

        步长地板若重置回原步长会空转而不推进 t，最后用未经验收的中间态
        拼控制律返回且无失败标志；正确行为是步长缩到地板立即抛
        ``PropagationFailure``。
        """
        system = SimpleNamespace(origin="EARTH")
        forces = [PointMassGravity("EARTH", mu=MU)]
        # r=1e-6 km（离地心 1 mm）、切向 7.5 km/s：二体引力加速度 ~4e17 km/s²，
        # 自适应控制器不断缩步仍无法满足容差 → 步长塌缩
        init = np.array([1e-6, 0.0, 0.0, 0.0, 7.5, 0.0])
        engine = EngineConfig(t_max=0.5, isp=3000.0)

        with pytest.raises(PropagationFailure):
            qlaw_guess(
                system,
                forces,
                engine,
                init,
                1000.0,
                (8000.0, 0.0, 0.0),
                0.0,
                5 * 86400.0,
                5,
                step=60.0,
            )


class TestResolveMuDoesNotSilentlyFallback:
    def test_no_mu_source_raises(self):
        """forces 无 PointMassGravity、system 无 gravitational_parameter → 抛异常。

        静默返回地球 μ 会让非地球系统被用错动力学参数。
        """
        with pytest.raises(RuntimeError):
            _resolve_mu(SimpleNamespace(origin="MARS"), [])

    def test_gravitational_parameter_query_failure_raises(self):
        """system.gravitational_parameter 查询失败 → 异常上抛（不静默地球 μ）。"""

        class BrokenSystem:
            origin = "MARS"

            def gravitational_parameter(self, origin):
                raise KeyError(f"no ephemeris for {origin}")

        with pytest.raises(KeyError):
            _resolve_mu(BrokenSystem(), [])

    def test_point_mass_gravity_mu_still_used(self):
        """回归：PointMassGravity 提供 μ 时正常返回（不误伤 happy path）。"""
        mu = _resolve_mu(SimpleNamespace(origin="EARTH"), [PointMassGravity("EARTH", mu=MU)])
        assert mu == MU
