"""完整作用量角度局部根数的约定与可逆性测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.normal_form.coord_trans import (
    action_angle_to_canonical,
    canonical_to_action_angle,
    shortest_angle_error,
)

pytestmark = pytest.mark.theory


def test_qiao_action_angle_round_trip_preserves_canonical_state():
    """正作用量和双曲分量符号均在往返中保留。"""
    canonical = np.array([0.03, 0.01, -0.02, 0.015, -0.025, 0.04])
    elements = canonical_to_action_angle(canonical)
    expected = np.array(
        [
            canonical[0] * canonical[3],
            0.5 * np.log(abs(canonical[0] / canonical[3])),
            0.5 * (canonical[1] ** 2 + canonical[4] ** 2),
            np.arctan2(canonical[1], canonical[4]) % (2 * np.pi),
            0.5 * (canonical[2] ** 2 + canonical[5] ** 2),
            np.arctan2(canonical[2], canonical[5]) % (2 * np.pi),
        ]
    )
    np.testing.assert_allclose(elements, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(action_angle_to_canonical(elements), canonical, atol=2e-15)


def test_negative_saddle_action_round_trip_preserves_opposite_signs():
    """双曲作用量为负时保留 q1 与 p1 的异号关系。"""
    canonical = np.array([-0.03, 0.01, -0.02, 0.015, -0.025, 0.04])
    elements = canonical_to_action_angle(canonical)
    assert elements[0] < 0.0
    np.testing.assert_allclose(action_angle_to_canonical(elements), canonical, atol=2e-15)


def test_negative_hyperbolic_momentum_outside_documented_branch_is_rejected():
    """p1<0 的状态不在 p1>0 往返分支内，显式拒绝而不是返回镜像状态。"""
    canonical = np.array([0.03, 0.01, -0.02, -0.015, -0.025, 0.04])
    with pytest.raises(ValueError, match="p1>0"):
        canonical_to_action_angle(canonical)


def test_zero_saddle_coordinate_rejects_undefined_logarithmic_angle():
    """双曲坐标为零时不对数奇异性返回伪造有限角。"""
    with pytest.raises(ValueError, match="theta1"):
        canonical_to_action_angle(np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1]))


def test_angle_difference_takes_shortest_periodic_branch():
    """跨零的角度误差不产生接近 2π 的伪大残差。"""
    assert shortest_angle_error(0.01, 2 * np.pi - 0.01) == pytest.approx(0.02)
    assert shortest_angle_error(2 * np.pi - 0.01, 0.01) == pytest.approx(-0.02)
