"""ImpulsiveBurn / FiniteBurn 定义与 API 契约。"""

import dataclasses

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.force_config import build_force
from e2m2e.algorithm.forces.thrust import FiniteBurn, ImpulsiveBurn
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


def test_impulsive_burn_stores_copied_delta_v_and_is_frozen():
    """ImpulsiveBurn 存 epoch + delta_v（拷贝），且 frozen 不可变。"""
    dv = np.array([0.1, 0.2, 0.3])
    burn = ImpulsiveBurn(epoch=1.0, delta_v=dv)

    assert burn.epoch == 1.0
    np.testing.assert_allclose(burn.delta_v, [0.1, 0.2, 0.3])

    # 存储为拷贝：修改原数组不影响 burn
    dv[0] = 99.0
    np.testing.assert_allclose(burn.delta_v, [0.1, 0.2, 0.3])

    # frozen：不可重新赋值
    with pytest.raises(dataclasses.FrozenInstanceError):
        burn.epoch = 2.0


def _compiled_burn(
    *,
    mass: float = 1000.0,
    thrust_profile: dict[str, float | str] | None = None,
    direction: list[float] | None = None,
    direction_frame: str | None = None,
) -> FiniteBurn:
    params: dict[str, object] = {
        "mass": mass,
        "thrust_profile": thrust_profile or {"kind": "constant", "thrust": 10.0},
        "direction": {"kind": "fixed", "vector": direction or [1.0, 0.0, 0.0]},
    }
    if direction_frame is not None:
        params["direction_frame"] = direction_frame
    return build_force("FiniteBurn", params)


def test_finite_burn_constant_and_zero_thrust_propagation():
    """恒定/零 DSL 推力通过公开传播接口给出正确的速度变化。"""
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    for thrust, expected_delta_v in ((10.0, 1e-5), (0.0, 0.0)):
        burn = _compiled_burn(thrust_profile={"kind": "constant", "thrust": thrust})
        result = ForceModel(FakeSystem(), [burn]).propagate(
            y0, (0.0, 1.0), t_eval=np.array([0.0, 1.0])
        )
        np.testing.assert_allclose(result["states"][-1, 3] - y0[3], expected_delta_v, rtol=1e-8)


def test_finite_burn_pulse_profile_switches_at_configured_epochs():
    """pulse profile 只在闭区间内施加推力。"""
    fm = ForceModel(
        FakeSystem(),
        [
            _compiled_burn(
                thrust_profile={
                    "kind": "pulse",
                    "t_start": 1.0,
                    "t_end": 2.0,
                    "thrust": 10.0,
                }
            )
        ],
    )
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = fm.propagate(y0, (0.0, 3.0), t_eval=np.array([0.0, 1.0, 2.0, 3.0]))
    # 端点包含在内，积分器在两个端点间只会对约 1 秒的有效区间累计推力。
    np.testing.assert_allclose(result["states"][-1, 3], 1e-5, rtol=1e-5, atol=1e-11)


def test_finite_burn_pulse_profile_honors_boundaries_outside_t_eval():
    """pulse 边界不是输出点时，累计冲量仍只覆盖开机区间。"""
    burn = _compiled_burn(
        thrust_profile={"kind": "pulse", "t_start": 0.75, "t_end": 1.25, "thrust": 10.0}
    )
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    sparse = ForceModel(FakeSystem(), [burn]).propagate(y0, (0.0, 3.0), t_eval=np.array([0.0, 3.0]))
    dense = ForceModel(FakeSystem(), [burn]).propagate(
        y0, (0.0, 3.0), t_eval=np.linspace(0.0, 3.0, 31)
    )
    np.testing.assert_allclose(sparse["states"][-1, 3], 5e-6, rtol=2e-5, atol=1e-11)
    np.testing.assert_allclose(sparse["states"][-1], dense["states"][-1], atol=1e-11)


def test_finite_burn_constant_force_supports_stm():
    """恒质量固定方向推力可通过公开接口传播 STM。"""
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = ForceModel(FakeSystem(), [_compiled_burn()]).propagate(
        y0, (0.0, 1.0), t_eval=np.array([0.0, 1.0]), with_stm=True
    )
    assert result["stm"].shape == (2, 6, 6)
    np.testing.assert_allclose(result["stm"][0], np.eye(6), atol=1e-12)
    np.testing.assert_allclose(result["stm"][-1, 3:, 3:], np.eye(3), atol=1e-8)


def test_finite_burn_vnb_stm_matches_initial_velocity_finite_difference():
    """VNB 推力的公开 STM 与独立初态速度有限差分一致。"""
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    t_eval = np.array([0.0, 10.0])
    fm = ForceModel(
        FakeSystem(), [_compiled_burn(direction=[1.0, 0.0, 0.0], direction_frame="VNB")]
    )
    stm_result = fm.propagate(y0, (0.0, 10.0), t_eval=t_eval, with_stm=True)
    perturbation = 1e-6
    plus = y0.copy()
    minus = y0.copy()
    plus[3] += perturbation
    minus[3] -= perturbation
    final_plus = fm.propagate(plus, (0.0, 10.0), t_eval=t_eval)["states"][-1]
    final_minus = fm.propagate(minus, (0.0, 10.0), t_eval=t_eval)["states"][-1]
    finite_difference = (final_plus - final_minus) / (2.0 * perturbation)

    np.testing.assert_allclose(stm_result["stm"][-1, :, 3], finite_difference, rtol=1e-4, atol=1e-7)


def test_finite_burn_rejects_uncompiled_callables():
    """任意 Python callable 不能在 Rust RK 内执行，传播必须明确拒绝。"""
    y0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    callable_profile = FiniteBurn(lambda t: 10.0, [1.0, 0.0, 0.0], 1000.0)
    compiled_profile = _compiled_burn().thrust_profile
    callable_direction = FiniteBurn(compiled_profile, lambda _t, _state: [1.0, 0.0, 0.0], 1000.0)

    for burn in (callable_profile, callable_direction):
        with pytest.raises(NotImplementedError, match="无 Rust 实现"):
            ForceModel(FakeSystem(), [burn]).propagate(y0, (0.0, 1.0))
