"""PoincareSection 庞加莱截面工具测试。

验证平面截面与近拱点截面的事后穿越检测精度：
- 平面截面穿越坐标残差 < 1e-10
- 近拱点穿越态 r·v 残差 < 1e-8
"""

import numpy as np
import pytest

from e2m2e.algorithm.manifold import ManifoldKind, ManifoldTube, PoincareSection, SectionCrossings
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


# 3:1 DRO 种子（与 tests/conftest.py 一致），用于绕月弧的近拱点检测
DRO_X0 = 0.79188556619742
DRO_VY0 = 0.573665890385585
DRO_PERIOD = 6.307498


def _make_tube(states, times, system) -> ManifoldTube:
    """把单条轨迹包装成 ManifoldTube 以便调用 crossings()。"""
    arc = Orbit(states=states, times=times, system=system)
    return ManifoldTube(
        orbit=arc, kind=ManifoldKind.UNSTABLE, branch="+", epsilon=0.0, trajectories=[arc]
    )


class TestSectionConstruction:
    """测试截面构造"""

    def test_plane_invalid_axis(self):
        """非法轴索引报错"""
        with pytest.raises(ValueError, match="axis"):
            PoincareSection.plane(axis=6, value=0.0)

    def test_periapsis_unknown_center(self, cr3bp_system):
        """无法识别的中心天体报错"""
        with pytest.raises(ValueError, match="中心天体"):
            PoincareSection.periapsis("mars", cr3bp_system)

    def test_periapsis_body_names(self, cr3bp_system):
        """主/次天体名称（不区分大小写）均可构造"""
        PoincareSection.periapsis("Earth", cr3bp_system)
        PoincareSection.periapsis("MOON", cr3bp_system)


class TestPlaneCrossings:
    """测试平面截面穿越检测"""

    def test_plane_crossing_residual(self, cr3bp_system, cr3bp_dynamics):
        """平面截面穿越态坐标残差 < 1e-10

        用 3:1 DRO 传播一个周期，检测 y=0 平面穿越。
        """
        x0 = np.array([DRO_X0, 0, 0, 0, DRO_VY0, 0])
        t_eval = np.linspace(0, DRO_PERIOD, 4000)
        result = cr3bp_dynamics.propagate(x0, (0, DRO_PERIOD), t_eval=t_eval)

        section = PoincareSection.plane(axis=1, value=0.0)
        tube = _make_tube(result["states"], result["time"], cr3bp_system)
        crossings = section.crossings(tube)

        assert isinstance(crossings, SectionCrossings)
        assert len(crossings.times) >= 2, "一个周期内应至少穿越 y=0 两次"
        assert np.all(np.abs(crossings.states[:, 1]) < 1e-10)
        assert np.all(crossings.trajectory_index == 0)

    def test_plane_crossing_at_nonzero_value(self, cr3bp_system, cr3bp_dynamics):
        """非零取值的平面截面穿越残差同样 < 1e-10"""
        x0 = np.array([DRO_X0, 0, 0, 0, DRO_VY0, 0])
        t_eval = np.linspace(0, DRO_PERIOD, 4000)
        result = cr3bp_dynamics.propagate(x0, (0, DRO_PERIOD), t_eval=t_eval)

        section = PoincareSection.plane(axis=0, value=1.0)
        tube = _make_tube(result["states"], result["time"], cr3bp_system)
        crossings = section.crossings(tube)

        assert len(crossings.times) >= 1
        assert np.all(np.abs(crossings.states[:, 0] - 1.0) < 1e-10)

    def test_no_crossing_returns_empty(self, cr3bp_system, cr3bp_dynamics):
        """无穿越时返回空结果（形状保持 (0, 6)）"""
        x0 = np.array([DRO_X0, 0, 0, 0, DRO_VY0, 0])
        result = cr3bp_dynamics.propagate(x0, (0, 0.5), t_eval=np.linspace(0, 0.5, 500))

        section = PoincareSection.plane(axis=2, value=1.0)  # DRO 在平面内，z 恒为 0
        tube = _make_tube(result["states"], result["time"], cr3bp_system)
        crossings = section.crossings(tube)

        assert crossings.states.shape == (0, 6)
        assert len(crossings.times) == 0


class TestPeriapsisCrossings:
    """测试近拱点截面穿越检测"""

    def test_periapsis_crossing_residual(self, cr3bp_system, cr3bp_dynamics):
        """近拱点穿越态 r·v 残差 < 1e-8

        3:1 DRO 绕月一周有近月点与远月点，r·v = 0 穿越至少两次。
        """
        x0 = np.array([DRO_X0, 0, 0, 0, DRO_VY0, 0])
        t_eval = np.linspace(0, DRO_PERIOD, 4000)
        result = cr3bp_dynamics.propagate(x0, (0, DRO_PERIOD), t_eval=t_eval)

        section = PoincareSection.periapsis("moon", cr3bp_system)
        tube = _make_tube(result["states"], result["time"], cr3bp_system)
        crossings = section.crossings(tube)

        assert len(crossings.times) >= 2
        moon_pos = np.array([1.0 - cr3bp_system.mu, 0.0, 0.0])
        rv = np.array([np.dot(state[:3] - moon_pos, state[3:]) for state in crossings.states])
        assert np.all(np.abs(rv) < 1e-8)

    def test_periapsis_detects_perilune_and_apolune(self, cr3bp_system, cr3bp_dynamics):
        """近拱点截面同时检出近月点与远月点（相对距离一极小一极大）"""
        x0 = np.array([DRO_X0, 0, 0, 0, DRO_VY0, 0])
        t_eval = np.linspace(0, DRO_PERIOD, 4000)
        result = cr3bp_dynamics.propagate(x0, (0, DRO_PERIOD), t_eval=t_eval)

        section = PoincareSection.periapsis("moon", cr3bp_system)
        tube = _make_tube(result["states"], result["time"], cr3bp_system)
        crossings = section.crossings(tube)

        moon_pos = np.array([1.0 - cr3bp_system.mu, 0.0, 0.0])
        distances = np.linalg.norm(crossings.states[:, :3] - moon_pos, axis=1)
        assert distances.max() / distances.min() > 1.5, "应同时检出近月点与远月点"
