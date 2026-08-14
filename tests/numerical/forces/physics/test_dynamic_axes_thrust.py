"""FiniteBurn 动态方向帧的传播集成测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PointMassGravity
from e2m2e.algorithm.forces.force_config import build_force
from tests.numerical.forces.conftest import EARTH_RE, keplerian_to_cartesian, semi_major_axis

pytestmark = pytest.mark.force


def _propagate_dynamic_axes(
    system,
    direction,
    direction_frame,
    duration_s,
    n_points,
    *,
    eccentricity: float = 0.0,
    thrust: float = 1.0,
):
    """传播配置 DSL 表达的 VNB/LVLH 恒质量推力。"""
    mu = system.gravitational_parameter("EARTH")
    y0 = keplerian_to_cartesian(EARTH_RE + 400.0, eccentricity, 0.0, 0.0, 0.0, 0.0, mu)
    burn = build_force(
        "FiniteBurn",
        {
            "mass": 1000.0,
            "thrust_profile": {"kind": "constant", "thrust": thrust},
            "direction": {"kind": "fixed", "vector": direction},
            "direction_frame": direction_frame,
        },
    )
    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    result = ForceModel(system, [PointMassGravity(body="EARTH"), burn]).propagate(
        y0,
        (et0, et0 + duration_s),
        t_eval=np.linspace(et0, et0 + duration_s, n_points),
        max_steps=200_000,
    )
    return result, mu


def _eccentricity(state, mu):
    """从惯性状态计算偏心率。"""
    r_vec = state[:3]
    v_vec = state[3:6]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    return float(np.linalg.norm(e_vec))


@pytest.mark.spice
def test_vnb_velocity_direction_thrust_increases_orbital_energy(earth_icrf_system):
    """VNB 的速度轴推力持续提高半长轴和轨道能量。"""
    result, mu = _propagate_dynamic_axes(earth_icrf_system, [1.0, 0.0, 0.0], "VNB", 3600.0, 50)
    initial, final = result["states"][0], result["states"][-1]
    assert semi_major_axis(final, mu) > semi_major_axis(initial, mu)
    initial_energy = np.dot(initial[3:6], initial[3:6]) / 2.0 - mu / np.linalg.norm(initial[:3])
    final_energy = np.dot(final[3:6], final[3:6]) / 2.0 - mu / np.linalg.norm(final[:3])
    assert final_energy > initial_energy


@pytest.mark.spice
def test_lvlh_radial_direction_thrust_increases_eccentricity(earth_icrf_system):
    """LVLH 的径向推力使近圆轨道偏心率增加。"""
    result, mu = _propagate_dynamic_axes(
        earth_icrf_system, [1.0, 0.0, 0.0], "LVLH", 3600.0, 50, eccentricity=0.001
    )
    assert _eccentricity(result["states"][-1], mu) > _eccentricity(result["states"][0], mu)


@pytest.mark.spice
def test_dynamic_direction_frames_normalize_mixed_direction(earth_icrf_system):
    """VNB/LVLH 混合方向都可通过 Rust 路径传播。"""
    for direction_frame, direction in (("VNB", [1.0, 1.0, 1.0]), ("LVLH", [1.0, 1.0, 0.0])):
        result, _ = _propagate_dynamic_axes(
            earth_icrf_system, direction, direction_frame, 600.0, 10
        )
        assert np.isfinite(result["states"]).all()


@pytest.mark.spice
def test_vnb_zero_thrust_matches_gravity_only(earth_icrf_system):
    """零推力时不会尝试解析动态方向，轨迹与纯引力相同。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")
    y0 = keplerian_to_cartesian(EARTH_RE + 400.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu)
    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_eval = np.linspace(et0, et0 + 600.0, 20)
    zero_burn = build_force(
        "FiniteBurn",
        {
            "mass": 1000.0,
            "thrust_profile": {"kind": "constant", "thrust": 0.0},
            "direction": {"kind": "fixed", "vector": [1.0, 0.0, 0.0]},
            "direction_frame": "VNB",
        },
    )
    gravity = PointMassGravity(body="EARTH")
    with_burn = ForceModel(system, [gravity, zero_burn]).propagate(
        y0, (et0, et0 + 600.0), t_eval=t_eval, max_steps=200_000
    )
    gravity_only = ForceModel(system, [gravity]).propagate(
        y0, (et0, et0 + 600.0), t_eval=t_eval, max_steps=200_000
    )
    np.testing.assert_allclose(with_burn["states"], gravity_only["states"], atol=1e-12)
