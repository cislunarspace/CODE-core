"""IAS15 积分器端到端测试（编译力模型路径）。

IAS15（15 阶 Gauss-Radau 预测-校正 + 补偿求和，Rein & Spiegel 2015 /
Holman et al. 2023 ASSIST）经 ``ForceModel.propagate(integrator="ias15")``
暴露。断言全部来自解析解与守恒量：

- 圆轨道二体：解析旋转解；
- e=0.9 椭圆：整圈闭合（近拱点步长自适应）；
- 二体 Hamilton 流的 STM 是辛矩阵：det Φ = 1；
- 长弧段能量守恒（补偿求和抑制舍入积累）。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.forces import ForceModel, PointMassGravity
from tests.numerical.forces.conftest import EARTH_MU, FakeSystem

pytestmark = pytest.mark.integrator


def _make_fm() -> ForceModel:
    """纯二体点质量 ForceModel（无需 SPICE 求值）。"""
    fm = ForceModel(FakeSystem(), forces=[PointMassGravity("EARTH", mu=EARTH_MU)])
    fm.rtol = 1e-13
    fm.max_step = 600.0
    return fm


def _circular_y0(r: float = 7000.0) -> np.ndarray:
    v = np.sqrt(EARTH_MU / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def _circular_exact(y0: np.ndarray, t: float) -> np.ndarray:
    """圆轨道解析解：轨道面内匀速旋转。"""
    r = np.linalg.norm(y0[:3])
    n = np.sqrt(EARTH_MU / r**3)
    theta = n * t
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.concatenate([rot @ y0[:3], rot @ y0[3:]])


class TestIas15Kepler:
    def test_circular_orbit_ten_periods(self):
        """圆轨道 10 圈，末态对照解析解。"""
        y0 = _circular_y0()
        period = 2.0 * np.pi * np.sqrt(y0[0] ** 3 / EARTH_MU)
        t1 = 10.0 * period
        result = _make_fm().propagate(y0, (0.0, t1), t_eval=[t1], integrator="ias15")
        exact = _circular_exact(y0, t1)
        err = np.linalg.norm(result["states"][-1] - exact)
        assert err < 1e-7, f"10 圈末态误差 {err:.3e} km"

    def test_eccentric_orbit_closure(self):
        """e=0.9 椭圆整圈闭合：近拱点强加速靠步长自适应解析。"""
        a = 7000.0
        e = 0.9
        r_a = a * (1.0 + e)
        v_a = np.sqrt(EARTH_MU * (1.0 - e) / (a * (1.0 + e)))
        y0 = np.array([r_a, 0.0, 0.0, 0.0, v_a, 0.0])  # 远拱点出发，xy 平面
        period = 2.0 * np.pi * np.sqrt(a**3 / EARTH_MU)
        result = _make_fm().propagate(y0, (0.0, period), t_eval=[period], integrator="ias15")
        err = np.linalg.norm(result["states"][-1] - y0)
        assert err < 1e-5, f"整圈闭合误差 {err:.3e} km"

    def test_matches_rk_backend(self):
        """与 PD78 后端交叉对照：两积分器在各自容差内一致。"""
        y0 = _circular_y0()
        period = 2.0 * np.pi * np.sqrt(y0[0] ** 3 / EARTH_MU)
        t1 = 3.0 * period
        fm = _make_fm()
        res_ias15 = fm.propagate(y0, (0.0, t1), t_eval=[t1], integrator="ias15")
        res_rk = fm.propagate(y0, (0.0, t1), t_eval=[t1], integrator="rk")
        diff = np.linalg.norm(res_ias15["states"][-1] - res_rk["states"][-1])
        assert diff < 1e-6, f"两后端末态差异 {diff:.3e} km"

    def test_stm_det_is_one(self):
        """二体 Hamilton 流的变分矩阵是辛矩阵：det Φ = 1。"""
        y0 = _circular_y0()
        period = 2.0 * np.pi * np.sqrt(y0[0] ** 3 / EARTH_MU)
        result = _make_fm().propagate(
            y0, (0.0, period), t_eval=[period], integrator="ias15", with_stm=True
        )
        det = np.linalg.det(result["stm"][-1])
        assert abs(det - 1.0) < 1e-8, f"det Φ = {det}"

    def test_long_run_energy_drift(self):
        """100 圈能量漂移：补偿求和使舍入按 Brouwer 律 n^(1/2) 积累。

        上界取宽容的 1e-9（比能比），只防回归，不做精确定标。
        """
        y0 = _circular_y0()
        period = 2.0 * np.pi * np.sqrt(y0[0] ** 3 / EARTH_MU)
        t1 = 100.0 * period
        result = _make_fm().propagate(y0, (0.0, t1), t_eval=[t1], integrator="ias15")

        def energy(s: np.ndarray) -> float:
            return 0.5 * float(s[3:] @ s[3:]) - EARTH_MU / float(np.linalg.norm(s[:3]))

        drift = abs(energy(result["states"][-1]) - energy(y0)) / abs(energy(y0))
        assert drift < 1e-9, f"100 圈相对能量漂移 {drift:.3e}"


class TestIas15Interface:
    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValueError, match="integrator"):
            _make_fm().propagate(_circular_y0(), (0.0, 60.0), t_eval=[60.0], integrator="rk45")

    def test_ias15_state_only_matches_stm_states(self):
        """with_stm 开关不改变状态轨迹（容差内一致）。"""
        y0 = _circular_y0()
        period = 2.0 * np.pi * np.sqrt(y0[0] ** 3 / EARTH_MU)
        fm = _make_fm()
        plain = fm.propagate(y0, (0.0, period), t_eval=[period], integrator="ias15")
        with_stm = fm.propagate(
            y0, (0.0, period), t_eval=[period], integrator="ias15", with_stm=True
        )
        assert_allclose(with_stm["states"][-1], plain["states"][-1], atol=1e-8, rtol=0)
