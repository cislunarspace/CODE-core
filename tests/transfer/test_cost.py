"""转移成本计算测试。

验证速度匹配时成本为零、非零和与手动公式等价。
"""

import numpy as np
import pytest

from e2m2e.transfer.cost import compute_transfer_cost


def test_transfer_cost_zero_when_velocities_match():
    """出发/注入速度相同、末端/插入速度相同 → 成本为 0"""
    vel = np.array([1.0, 0.0, 0.0])
    departure_state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])

    result = compute_transfer_cost(
        departure_state=departure_state,
        initial_velocity=vel,
        final_velocity=vel,
        insertion_velocity=vel,
    )

    assert result.total == pytest.approx(0.0, abs=1e-12)
    assert result.dv1 == pytest.approx(0.0, abs=1e-12)
    assert result.dv2 == pytest.approx(0.0, abs=1e-12)


def test_transfer_cost_nonzero_sum_of_dv1_and_dv2():
    """dv1=0.3 (切向), dv2=0.4 (径向) → total=0.7"""
    departure_state = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
    initial_velocity = np.array([1.3, 0.0, 0.0])
    final_velocity = np.array([0.0, 0.6, 0.0])
    insertion_velocity = np.array([0.0, 0.2, 0.0])

    result = compute_transfer_cost(
        departure_state=departure_state,
        initial_velocity=initial_velocity,
        final_velocity=final_velocity,
        insertion_velocity=insertion_velocity,
    )

    assert result.dv1 == pytest.approx(0.3, abs=1e-12)
    assert result.dv2 == pytest.approx(0.4, abs=1e-12)
    assert result.total == pytest.approx(0.7, abs=1e-12)


def test_transfer_cost_matches_manual_formula():
    """与 compute_delta_v1 + compute_delta_v2 公式逐一等价"""
    rng = np.random.default_rng(42)

    departure_vel = rng.standard_normal(3)
    departure_state = np.concatenate([rng.standard_normal(3), departure_vel])
    initial_velocity = rng.standard_normal(3)
    final_velocity = rng.standard_normal(3)
    insertion_velocity = rng.standard_normal(3)

    expected_dv1 = float(np.linalg.norm(initial_velocity - departure_vel))
    expected_dv2 = float(np.linalg.norm(final_velocity - insertion_velocity))

    result = compute_transfer_cost(
        departure_state=departure_state,
        initial_velocity=initial_velocity,
        final_velocity=final_velocity,
        insertion_velocity=insertion_velocity,
    )

    assert result.dv1 == pytest.approx(expected_dv1, abs=1e-12)
    assert result.dv2 == pytest.approx(expected_dv2, abs=1e-12)
    assert result.total == pytest.approx(expected_dv1 + expected_dv2, abs=1e-12)
