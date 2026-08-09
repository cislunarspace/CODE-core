"""ForceModel.propagate_maneuvers 编排测试。

验证单脉冲速度/能量变化、多脉冲顺序与边界。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.thrust import BurnApplication, ImpulsiveBurn

pytestmark = pytest.mark.force


class _FakeSystem:
    """仅用于传播测试的最小 System 桩。"""

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return 398600.4415


def test_single_impulsive_burn_changes_velocity_and_energy():
    """单脉冲 @ t0、零外力：末速度 = 初速度 + Δv；比动能变化 = v·Δv + 0.5|Δv|²。"""
    system = _FakeSystem()
    fm = ForceModel(system)  # 无力模型 → 零加速度

    v0 = np.array([1.0, 0.5, -0.2])
    dv = np.array([0.3, -0.1, 0.4])
    y0 = np.array([100.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

    burn = ImpulsiveBurn(epoch=0.0, delta_v=dv)
    result = fm.propagate_maneuvers(y0, (0.0, 1.0), burns=[burn])

    final = result["states"][-1]

    # 零外力下 coast 不改速度；末速度 = 初速度 + Δv
    np.testing.assert_allclose(final[3:6], v0 + dv, rtol=1e-12)

    # 比动能变化 = v·Δv + 0.5|Δv|²
    delta_energy = 0.5 * (np.dot(final[3:6], final[3:6]) - np.dot(v0, v0))
    expected = np.dot(v0, dv) + 0.5 * np.dot(dv, dv)
    np.testing.assert_allclose(delta_energy, expected, rtol=1e-12)


def test_propagate_maneuvers_records_burn_application():
    """BurnApplication 记录 index/epoch/delta_v/velocity_before/after，index 指向 post-burn 行。"""
    system = _FakeSystem()
    fm = ForceModel(system)  # 零外力

    v0 = np.array([1.0, 0.0, 0.0])
    dv = np.array([0.0, 0.5, 0.0])
    y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

    burn = ImpulsiveBurn(epoch=0.5, delta_v=dv)
    result = fm.propagate_maneuvers(y0, (0.0, 1.0), burns=[burn])

    burns = result["burns"]
    assert len(burns) == 1
    rec = burns[0]
    assert isinstance(rec, BurnApplication)
    assert rec.epoch == pytest.approx(0.5)
    np.testing.assert_allclose(rec.delta_v, dv, rtol=1e-12)
    np.testing.assert_allclose(rec.velocity_before, v0, rtol=1e-12)
    np.testing.assert_allclose(rec.velocity_after, v0 + dv, rtol=1e-12)

    # index 指向 post-burn 行：该行时刻 == burn epoch、速度 == velocity_after
    assert result["time"][rec.index] == pytest.approx(0.5)
    np.testing.assert_allclose(result["states"][rec.index][3:6], rec.velocity_after, rtol=1e-12)


def test_propagate_maneuvers_no_duplicate_epoch_at_burn():
    """burn epoch 处输出单行（post-burn）：time 严格单调、burn epoch 恰好一次。"""
    system = _FakeSystem()
    fm = ForceModel(system)  # 零外力

    y0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    burn = ImpulsiveBurn(epoch=0.5, delta_v=np.array([0.0, 0.5, 0.0]))
    result = fm.propagate_maneuvers(y0, (0.0, 1.0), burns=[burn])

    time = result["time"]
    # time 严格单调递增 → 无重复 epoch（拼接已丢 pre-burn 行）
    assert np.all(np.diff(time) > 0.0), "burn epoch 处出现重复行"
    # burn epoch 恰好出现一次（post-burn 单行）
    assert int(np.sum(np.isclose(time, 0.5))) == 1


def test_propagate_maneuvers_rejects_burn_epoch_outside_span():
    """burn epoch 落在 t_span 之外 → ValueError（不静默丢弃）。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    y0 = np.zeros(6)

    # epoch < t0
    with pytest.raises(ValueError, match="t_span"):
        fm.propagate_maneuvers(
            y0, (0.0, 1.0), burns=[ImpulsiveBurn(epoch=-0.1, delta_v=np.zeros(3))]
        )
    # epoch > tf
    with pytest.raises(ValueError, match="t_span"):
        fm.propagate_maneuvers(
            y0, (0.0, 1.0), burns=[ImpulsiveBurn(epoch=1.5, delta_v=np.zeros(3))]
        )
