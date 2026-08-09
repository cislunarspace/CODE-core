"""station_keeping 误差模型单测（纯 numpy，不依赖 SPICE/pyd）。"""

import numpy as np
import pytest

from e2m2e.algorithm.station_keeping.error_models import (
    BoxMullerSampler,
    NavigationErrorModel,
    SrpErrorModel,
    ThrustExecutionError,
)

pytestmark = pytest.mark.orchestration


class TestBoxMullerSampler:
    def test_statistics(self):
        """大样本均值≈0、标准差≈1（Marsaglia 极坐标 Box-Muller）。"""
        s = BoxMullerSampler(seed=42)
        z = s.standard_normal(200_000)
        assert abs(z.mean()) < 5e-3
        assert abs(z.std() - 1.0) < 5e-3

    def test_reproducible(self):
        """同种子同结果（蒙特卡洛回归要求）。"""
        a = BoxMullerSampler(seed=7).standard_normal(100)
        b = BoxMullerSampler(seed=7).standard_normal(100)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        a = BoxMullerSampler(seed=1).standard_normal(50)
        b = BoxMullerSampler(seed=2).standard_normal(50)
        assert not np.allclose(a, b)


class TestNavigationErrorModel:
    def test_perturb_statistics(self):
        """扰动各分量标准差 = 对应 1-sigma（位置 m、速度 m/s 换算 km）。"""
        model = NavigationErrorModel(position_sigma_m=1500.0, velocity_sigma_mps=0.002)
        s = BoxMullerSampler(seed=3)
        state = np.zeros(6)
        dz = np.stack([model.perturb(state, s) for _ in range(20_000)])
        assert abs(dz[:, :3].std() - 1.5) < 0.05  # km
        assert abs(dz[:, 3:].std() - 2e-6) < 1e-7  # km/s

    def test_reproducible(self):
        model = NavigationErrorModel()
        s1 = BoxMullerSampler(seed=5)
        s2 = BoxMullerSampler(seed=5)
        np.testing.assert_array_equal(
            model.perturb(np.zeros(6), s1), model.perturb(np.zeros(6), s2)
        )


class TestThrustExecutionError:
    def _apply(self, mag, **kw):
        """构造沿 +x 的理论控制量并施加误差。"""
        model = ThrustExecutionError(**kw)
        dv_c = np.array([mag, 0.0, 0.0])
        return model.apply(dv_c, BoxMullerSampler(seed=11))

    def test_below_min_no_burn(self):
        """|Δv| < dv_min：完全不开机（返回 None，非失败）。"""
        dv, failed = self._apply(0.05, dv_min=0.1)
        assert dv is None and not failed

    def test_above_max_failed(self):
        """|Δv| > dv_max：判控制失败。"""
        dv, failed = self._apply(150.0, dv_max=100.0)
        assert dv is None and failed

    def test_abs_branch_magnitude(self):
        """小量段（min ≤ |Δv| < mid）：大小 = Δv + N(0, abs_sigma)。"""
        dv, failed = self._apply(1.0, dv_min=0.1, dv_mid=10.0, abs_sigma_mps=0.033)
        assert not failed
        mag = np.linalg.norm(dv)  # m/s
        # 期望：大小 ≈ 1.0 + 0.033·z m/s；3σ 范围 ≈ ±0.099 m/s
        # 方向误差 0.333° 的 3σ ≈ 1°，横向分量 ≤ sin(1°)·1.0 ≈ 0.0175 m/s
        assert 0.9 < mag < 1.1
        assert abs(dv[1]) < 0.02 and abs(dv[2]) < 0.02
        assert dv[0] > 0

    def test_abs_branch_direction_error(self):
        """小量段含方向角误差：角度 1-sigma 生效（多样本统计）。"""
        model = ThrustExecutionError(angle_sigma_deg=0.333)
        angs = []
        for _ in range(3000):
            dv, failed = model.apply(np.array([1.0, 0.0, 0.0]), BoxMullerSampler(seed=1 + _))
            angs.append(np.arctan2(dv[1], dv[0]))
        angs = np.array(angs)
        assert abs(np.degrees(angs.std()) - 0.333) < 0.03

    def test_rel_branch_magnitude(self):
        """大量段（mid ≤ |Δv|）：大小 = Δv·(1 + N(0, rel_sigma))。"""
        dv, failed = self._apply(20.0, dv_mid=10.0, rel_sigma=0.003)
        assert not failed
        mag = np.linalg.norm(dv)
        assert 19.9 < mag < 20.1

    def test_mid_boundary_uses_abs_branch(self):
        """|Δv| = dv_mid 落在小量段（绝对误差）。"""
        dv, failed = self._apply(10.0, dv_mid=10.0, abs_sigma_mps=0.033, rel_sigma=0.003)
        assert not failed
        mag = np.linalg.norm(dv)
        # 绝对误差 0.033 m/s（apply 返回 m/s）；若误用 rel 分支则偏差 ≈ 10×0.003 = 0.03
        # 阈值 0.1 m/s 远大于 3σ(0.033) ≈ 0.1 m/s，远小于 rel 分支 0.03 m/s
        assert abs(mag - 10.0) < 0.1


class TestSrpErrorModel:
    def test_scale_distribution(self):
        """光压乘子 = 1 + level·z（z 标准正态）。"""
        model = SrpErrorModel(error_level=0.10)
        s = BoxMullerSampler(seed=9)
        scales = np.array([model.sample_cr_scale(s) for _ in range(10_000)])
        assert abs(scales.mean() - 1.0) < 5e-3
        assert abs(scales.std() - 0.10) < 5e-3

    def test_reproducible(self):
        model = SrpErrorModel()
        a = model.sample_cr_scale(BoxMullerSampler(seed=4))
        b = model.sample_cr_scale(BoxMullerSampler(seed=4))
        assert a == b
