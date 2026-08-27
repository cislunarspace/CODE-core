"""lowthrust_shooting 油门越界报错测试。

``_decode_segments`` 不静默 clip 油门到 [0,1]：SLSQP 受 bounds 约束输出
本不应越界，越界说明约束未生效或决策非法，应报约束违反而不是用被改过的
油门传播（传播用的油门与决策变量不一致，掩盖问题）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig, LowThrustShooting

pytestmark = [pytest.mark.orchestration, pytest.mark.low_thrust]

MU = 398600.435507  # km³/s²


@pytest.fixture
def shooter() -> LowThrustShooting:
    from types import SimpleNamespace

    system = SimpleNamespace(origin="EARTH")
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    init = np.array([7000.0, 0.0, 0.0, 0.0, np.sqrt(MU / 7000.0), 0.0])
    return LowThrustShooting(
        system,
        [PointMassGravity("EARTH", mu=MU)],
        engine,
        init,
        initial_mass=1000.0,
        target_state=init.copy(),
        t0=0.0,
        tf=1200.0,
    )


class TestDecodeSegmentsThrottleBounds:
    def test_in_range_throttle_decoded(self, shooter):
        """合法油门（[0,1] 内）正常解码。"""
        segs = shooter._decode_segments(np.array([[0.5, 0.0, 0.0], [1.0, 0.1, 0.2]]))
        assert len(segs) == 2
        assert segs[0][0] == pytest.approx(0.5)
        assert segs[1][0] == pytest.approx(1.0)

    def test_throttle_above_one_raises(self, shooter):
        """油门 > 1（超过满推）报约束违反，不静默 clip 到 1。"""
        with pytest.raises(ValueError, match="油门越出"):
            shooter._decode_segments(np.array([[1.5, 0.0, 0.0]]))

    def test_negative_throttle_raises(self, shooter):
        """负油门（反向推）报约束违反，不静默 clip 到 0。"""
        with pytest.raises(ValueError, match="油门越出"):
            shooter._decode_segments(np.array([[-0.1, 0.0, 0.0]]))
