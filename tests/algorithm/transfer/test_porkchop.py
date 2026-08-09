"""porkchop 扫描测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.transfer.porkchop import PorkchopData, porkchop
from e2m2e.algorithm.transfer.terminal import TerminalCondition

pytestmark = pytest.mark.orchestration


MU_EARTH = 398600.4418  # km³/s²


class CircularOrbitTerminal(TerminalCondition):
    """解析圆轨道终端：半径 r、初始相位 phase0（rad），二体开普勒运动。"""

    def __init__(self, radius: float, phase0: float = 0.0, mu: float = MU_EARTH):
        self.radius = float(radius)
        self.phase0 = float(phase0)
        self.mu = float(mu)
        self.n = np.sqrt(self.mu / self.radius**3)  # 平均运动
        self.v = np.sqrt(self.mu / self.radius)

    def _state_at(self, t: float) -> np.ndarray:
        th = self.phase0 + self.n * t
        return np.array(
            [
                self.radius * np.cos(th),
                self.radius * np.sin(th),
                0.0,
                -self.v * np.sin(th),
                self.v * np.cos(th),
                0.0,
            ]
        )

    def get_initial_state(self) -> np.ndarray:
        return self._state_at(0.0)

    def get_arrival_state(self, t_ins: float, dynamics: object) -> tuple[np.ndarray, np.ndarray]:
        state = self._state_at(float(t_ins))
        return state[:3], state[3:6]


@pytest.fixture
def leo_geo():
    """LEO (7000 km) -> GEO (42164 km) 共面圆轨道场景。"""
    dep = CircularOrbitTerminal(7000.0)
    arr = CircularOrbitTerminal(42164.0)
    return dep, arr


class TestPorkchopGrid:
    def test_grid_shapes(self, leo_geo):
        dep, arr = leo_geo
        t_dep = np.linspace(0.0, 3600.0, 5)
        tof = np.linspace(900.0, 3600.0, 7)
        data = porkchop(dep, arr, t_dep, tof, mu=MU_EARTH, dynamics=None)

        assert isinstance(data, PorkchopData)
        for field in (data.dv1, data.dv2, data.total):
            assert field.shape == (5, 7)
        np.testing.assert_allclose(data.total, data.dv1 + data.dv2)

    def test_nonnegative_and_has_valley(self, leo_geo):
        """LEO→GEO 网格：ΔV 非负，且存在明显的低 ΔV 谷区（霍曼附近）。"""
        dep, arr = leo_geo
        t_dep = np.linspace(0.0, 3600.0, 8)
        tof = np.linspace(3600.0, 5 * 3600.0, 20)  # 霍曼转移约 5.26 h 附近放宽
        data = porkchop(dep, arr, t_dep, tof, mu=MU_EARTH, dynamics=None)

        valid = data.total[~np.isnan(data.total)]
        assert np.all(valid >= 0.0)
        # 谷区与峰值差异显著（等值线形态存在）
        assert valid.min() < 0.5 * valid.max()

    def test_hohmann_valley_value(self, leo_geo):
        """最优网格点 ΔV 应接近霍曼转移理论值（约 3.94 km/s）。"""
        dep, arr = leo_geo
        r1, r2 = 7000.0, 42164.0
        a_t = (r1 + r2) / 2
        t_h = np.pi * np.sqrt(a_t**3 / MU_EARTH)
        dv_h = abs(np.sqrt(MU_EARTH / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)) + abs(
            np.sqrt(MU_EARTH / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
        )

        # 相位对齐的出发时刻：让到达端在 tof 后位于出发端对面
        phase_arr0 = np.pi - dep.n * 0.0 - arr.n * t_h
        arr_aligned = CircularOrbitTerminal(r2, phase0=phase_arr0)
        data = porkchop(dep, arr_aligned, [0.0], [t_h], mu=MU_EARTH, dynamics=None)
        assert data.total[0, 0] == pytest.approx(dv_h, rel=1e-3)
