"""FiniteBurn 编译传播的方向帧契约。"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.force_config import build_force
from e2m2e.algorithm.forces.thrust import FiniteBurn
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


def _propagate_delta_v(direction, direction_frame=None, *, thrust=10.0, state=None):
    params = {
        "mass": 1000.0,
        "thrust_profile": {"kind": "constant", "thrust": thrust},
        "direction": {"kind": "fixed", "vector": direction},
    }
    if direction_frame is not None:
        params["direction_frame"] = direction_frame
    burn = build_force("FiniteBurn", params)
    y0 = (
        np.asarray(state, dtype=float)
        if state is not None
        else np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    )
    result = ForceModel(FakeSystem(), [burn]).propagate(y0, (0.0, 1.0), t_eval=[0.0, 1.0])
    return result["states"][-1, 3:6] - y0[3:6]


@pytest.mark.parametrize("direction_frame", ["VNB", "LVLH"])
def test_finite_burn_direction_frame_is_validated(direction_frame):
    """VNB/LVLH 是允许的编译方向帧。"""
    burn = FiniteBurn(lambda _t: 10.0, [1.0, 0.0, 0.0], 1000.0, direction_frame)
    assert burn.direction_frame == direction_frame


def test_finite_burn_invalid_direction_frame_raises():
    """非法方向帧在构造时拒绝。"""
    with pytest.raises(ValueError, match="direction_frame"):
        FiniteBurn(lambda _t: 10.0, [1.0, 0.0, 0.0], 1000.0, "INVALID")


def test_finite_burn_lvlh_along_track_uses_orbital_axis():
    """LVLH 沿迹轴垂直于径向，即使速度含径向分量也不随速度偏斜。"""
    delta_v = _propagate_delta_v(
        [0.0, 1.0, 0.0],
        "LVLH",
        state=[7000.0, 0.0, 0.0, 1.0, 7.5, 0.0],
    )
    assert abs(delta_v[0]) < 1e-8
    assert delta_v[1] > 0.99e-5


@pytest.mark.parametrize(
    ("direction_frame", "direction", "expected"),
    [
        (None, [2.0, 0.0, 0.0], [1e-5, 0.0, 0.0]),
        ("VNB", [1.0, 0.0, 0.0], [0.0, 1e-5, 0.0]),
        ("VNB", [0.0, 1.0, 0.0], [0.0, 0.0, 1e-5]),
        ("VNB", [0.0, 0.0, 1.0], [1e-5, 0.0, 0.0]),
        ("LVLH", [1.0, 0.0, 0.0], [1e-5, 0.0, 0.0]),
        ("LVLH", [0.0, 1.0, 0.0], [0.0, 1e-5, 0.0]),
        ("LVLH", [0.0, 0.0, 1.0], [0.0, 0.0, 1e-5]),
    ],
)
def test_finite_burn_compiled_direction_frames(direction_frame, direction, expected):
    """固定推力方向在 Rust RHS 中按惯性/VNB/LVLH 正确解析。"""
    np.testing.assert_allclose(_propagate_delta_v(direction, direction_frame), expected, atol=1e-8)


def test_finite_burn_lvlh_collinear_state_normalizes_direction():
    """LVLH 共线退化时，径向/沿迹分量仍归一化为单位方向。"""
    delta_v = _propagate_delta_v(
        [2.0, 0.0, 0.0],
        "LVLH",
        state=[7000.0, 0.0, 0.0, 7.5, 0.0, 0.0],
    )
    np.testing.assert_allclose(delta_v, [1e-5, 0.0, 0.0], atol=1e-11)


def test_finite_burn_zero_thrust_skips_degenerate_direction_frame():
    """关机时不需要定义 VNB 方向。"""
    delta_v = _propagate_delta_v(
        [1.0, 0.0, 0.0], "VNB", thrust=0.0, state=[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(delta_v, np.zeros(3))


@pytest.mark.parametrize(
    ("direction_frame", "state", "error"),
    [
        ("VNB", [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0], "non-zero velocity"),
        ("LVLH", [0.0, 0.0, 0.0, 0.0, 7.5, 0.0], "non-zero position"),
    ],
)
def test_finite_burn_degenerate_direction_frame_raises(direction_frame, state, error):
    """开机时动态方向帧的退化状态明确失败。"""
    with pytest.raises(RuntimeError, match=error):
        _propagate_delta_v([1.0, 0.0, 0.0], direction_frame, state=state)


def test_finite_burn_callable_direction_is_not_compilable():
    """可调用方向不能跨 Python/Rust 边界，传播入口明确拒绝。"""
    profile = build_force(
        "FiniteBurn",
        {
            "mass": 1000.0,
            "thrust_profile": {"kind": "constant", "thrust": 10.0},
            "direction": {"kind": "fixed", "vector": [1.0, 0.0, 0.0]},
        },
    ).thrust_profile
    burn = FiniteBurn(profile, lambda _t, _state: [1.0, 0.0, 0.0], 1000.0)
    with pytest.raises(NotImplementedError, match="无 Rust 实现"):
        ForceModel(FakeSystem(), [burn]).propagate([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], (0.0, 1.0))
