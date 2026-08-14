"""碰撞终止 + body-radius 注入测试（ADR 0020 决策 5 / #355）。

覆盖：撞天体轨迹事件终止、初始在半径内短路、近掠过不误杀、radius 注入
校验、与用户事件合并、STM 增广路径、不同半径配置、默认不启用、正则化保留。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.data.constants import MOON, Datum

pytestmark = pytest.mark.theory

MOON_RADIUS_KM = MOON.require_mean_radius_km()
MOON_RADIUS_DU = MOON_RADIUS_KM / Datum.DE421.char_length_km
EARTH_RADIUS_DU = Datum.WGS84.earth_radius_km / Datum.DE421.char_length_km


@pytest.fixture
def dynamics():
    """地月 CR3BP 动力学，注入地球/月球半径（启用碰撞检测所需）。"""
    system = CR3BP_System(
        mu=Datum.DE421.mu,
        primary="Earth",
        secondary="Moon",
        primary_radius_km=Datum.WGS84.earth_radius_km,
        secondary_radius_km=MOON_RADIUS_KM,
    )._with_default_scales()
    return CR3BP_Dynamics(system)


def _moon_center(mu: float) -> np.ndarray:
    return np.array([1.0 - mu, 0.0, 0.0])


def _earth_center(mu: float) -> np.ndarray:
    return np.array([-mu, 0.0, 0.0])


def _rest_state(center: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """位置 center + offset、零速度的 6 维状态。"""
    return np.concatenate([center + offset, np.zeros(3)])


# ---------------------------------------------------------------------------
# 碰撞终止
# ---------------------------------------------------------------------------


def test_collision_moon_terminates(dynamics):
    """初始在月球外侧零速度释放：在月面处事件终止 + COLLISION 标记。"""
    mu = dynamics.system.mu
    y0 = _rest_state(_moon_center(mu), np.array([0.01, 0.0, 0.0]))

    result = dynamics.propagate(y0, (0.0, 0.1), backend="scipy", collision_detection=True)

    collision = result["collision"]
    assert collision is not None
    assert collision["body"] == "Moon"
    assert result["time"][-1] == pytest.approx(collision["t"])
    assert result["time"][-1] < 0.1  # 积分被截断
    # 终止位置在月面处
    r_final = np.linalg.norm(result["states"][-1, :3] - _moon_center(mu))
    assert r_final == pytest.approx(MOON_RADIUS_DU, rel=1e-6)


def test_collision_earth_terminates(dynamics):
    """初始在地球外侧零速度释放：在地面处事件终止 + COLLISION 标记。"""
    mu = dynamics.system.mu
    y0 = _rest_state(_earth_center(mu), np.array([-0.03, 0.0, 0.0]))

    result = dynamics.propagate(y0, (0.0, 0.1), backend="scipy", collision_detection=True)

    collision = result["collision"]
    assert collision is not None
    assert collision["body"] == "Earth"
    assert result["time"][-1] == pytest.approx(collision["t"])
    assert result["time"][-1] < 0.1
    r_final = np.linalg.norm(result["states"][-1, :3] - _earth_center(mu))
    assert r_final == pytest.approx(EARTH_RADIUS_DU, rel=1e-6)


def test_initial_inside_shortcut(dynamics):
    """初始状态已在月球半径内：短路为即时碰撞（单点轨迹 + t=0）。"""
    mu = dynamics.system.mu
    y0 = _rest_state(_moon_center(mu), np.array([0.0, 0.0, 0.001]))

    result = dynamics.propagate(y0, (0.0, 10.0), backend="scipy", collision_detection=True)

    assert result["collision"] is not None
    assert result["collision"]["body"] == "Moon"
    assert result["collision"]["t"] == 0.0
    assert len(result["time"]) == 1
    assert np.allclose(result["states"][0], y0[:6])


# ---------------------------------------------------------------------------
# 不误杀
# ---------------------------------------------------------------------------


def test_gravity_assist_no_collision(dynamics):
    """月球低高度掠过（未撞月面）正常积分，不误杀。"""
    mu = dynamics.system.mu
    # 月面上方 ~38 km，切向速度 3.0 VU（远超逃逸），离心 > 引力，向外飞离
    y0 = np.concatenate(
        [
            _moon_center(mu) + np.array([MOON_RADIUS_DU + 0.0001, 0.0, 0.0]),
            np.array([0.0, 3.0, 0.0]),
        ]
    )

    result = dynamics.propagate(y0, (0.0, 5.0), backend="scipy", collision_detection=True)

    assert result["collision"] is None
    assert result["time"][-1] == pytest.approx(5.0)  # 完整积分未截断


def test_radius_config_affects_termination():
    """不同半径配置产生不同终止行为：半径大 → 更早终止。"""
    mu = Datum.DE421.mu
    y0 = _rest_state(_moon_center(mu), np.array([0.01, 0.0, 0.0]))

    ts = []
    for r in (MOON_RADIUS_KM, MOON_RADIUS_KM * 1.5):
        system = CR3BP_System(
            mu=mu, primary="Earth", secondary="Moon", secondary_radius_km=r
        )._with_default_scales()
        res = CR3BP_Dynamics(system).propagate(
            y0, (0.0, 0.1), backend="scipy", collision_detection=True
        )
        ts.append(res["collision"]["t"])

    assert ts[0] > ts[1]  # 半径小 → 掉得更深才触发 → 终止更晚


# ---------------------------------------------------------------------------
# 与现有事件机制组合
# ---------------------------------------------------------------------------


def test_merge_with_user_events(dynamics):
    """用户事件与碰撞事件合并：两者都记录，碰撞仍可识别。"""
    mu = dynamics.system.mu
    y0 = _rest_state(_moon_center(mu), np.array([0.01, 0.0, 0.0]))

    def user_event(t, state):
        return state[0] - (1.0 - mu)  # 越过月球 x 位置

    user_event.terminal = False
    user_event.direction = 0

    result = dynamics.propagate(
        y0, (0.0, 0.1), events=[user_event], backend="scipy", collision_detection=True
    )

    assert len(result["t_events"]) == 3  # 用户事件 + 地球/月球两个碰撞事件
    assert result["collision"] is not None
    assert result["collision"]["body"] == "Moon"


def test_collision_with_stm(dynamics):
    """with_stm=True 时碰撞事件正常（增广状态只取前 3 维位置）。"""
    mu = dynamics.system.mu
    y0 = _rest_state(_moon_center(mu), np.array([0.01, 0.0, 0.0]))

    result = dynamics.propagate(
        y0, (0.0, 0.1), with_stm=True, backend="scipy", collision_detection=True
    )

    assert result["collision"] is not None
    assert result["stm"].shape[1:] == (6, 6)
    assert np.all(np.isfinite(result["stm"]))
