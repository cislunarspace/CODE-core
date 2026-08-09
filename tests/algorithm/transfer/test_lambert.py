"""二体 Lambert 求解器（``e2m2e.algorithm.transfer.lambert``）测试。"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from e2m2e.algorithm.transfer import LambertSolution, solve_lambert, solve_lambert_batch

MU = 398600.4418

# Vallado 经典算例（poliastro / Orekit 同款回归基准）
R0 = [5000.0, 10000.0, 2100.0]
RF = [-14600.0, 2500.0, 7000.0]
TOF = 3600.0
EXP_V0 = [-5.99249503, 1.92536671, 3.24563805]
EXP_VF = [-3.31245851, -4.19661901, -0.38529044]


def _propagate(r0, v0, tof, mu=MU):
    """数值积分二体轨道，作为独立交叉验证。"""

    def rhs(_t, y):
        return np.concatenate([y[3:], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3])

    sol = solve_ivp(rhs, (0.0, tof), np.concatenate([r0, v0]), rtol=1e-11, atol=1e-11)
    return sol.y[:3, -1]


class TestSolveLambert:
    def test_vallado_benchmark_regression(self):
        """基准值回归：短程解与 Vallado 文献值一致（文献值 8 位有效数字）。"""
        sol = solve_lambert(R0, RF, TOF, MU)
        assert isinstance(sol, LambertSolution)
        assert sol.converged
        assert sol.n_iter <= 5
        np.testing.assert_allclose(sol.v0, EXP_V0, atol=2e-6)
        np.testing.assert_allclose(sol.vf, EXP_VF, atol=2e-6)

    def test_76min_short_way(self):
        """任务指定算例：r0=[15945.34,0,0]，tof=76 min 短程解。"""
        r0 = [15945.34, 0.0, 0.0]
        rf = [12214.83899, 10249.46731, 0.0]
        sol = solve_lambert(r0, rf, 76.0 * 60.0, MU)
        np.testing.assert_allclose(sol.v0, [2.058913, 2.915965, 0.0], atol=1e-5)

    def test_long_way_reaches_target(self):
        """长程解：数值传播交叉验证落点。"""
        tof = TOF * 4.0
        sol = solve_lambert(R0, RF, tof, MU, direction="long")
        r_end = _propagate(R0, sol.v0, tof)
        np.testing.assert_allclose(r_end, RF, rtol=1e-7)

    @pytest.mark.parametrize("revs", [1, 2])
    def test_multi_rev_reaches_target(self, revs):
        """多圈解（右分支低能解）：数值传播交叉验证落点。"""
        tof = TOF * 30.0
        sol = solve_lambert(R0, RF, tof, MU, revs=revs)
        assert sol.revs == revs
        r_end = _propagate(R0, sol.v0, tof)
        np.testing.assert_allclose(r_end, RF, rtol=1e-7)

    def test_below_tmin_raises(self):
        """tof 低于该圈数最小转移时间时抛 ValueError。"""
        with pytest.raises(ValueError, match="最小时间"):
            solve_lambert(R0, RF, TOF, MU, revs=1)

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            solve_lambert(R0, RF, TOF, MU, direction="sideways")

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="长度 3"):
            solve_lambert([1.0, 2.0], RF, TOF, MU)


class TestSolveLambertBatch:
    def test_shape(self):
        r0_list = np.tile(R0, (3, 1))
        rf_list = np.tile(RF, (3, 1))
        out = solve_lambert_batch(r0_list, rf_list, [TOF, 2 * TOF], MU)
        assert out.shape == (3, 2, 2, 3)

    def test_batch_matches_single(self):
        """批量结果与逐条 solve_lambert 一致。"""
        r0_list = np.tile(R0, (4, 1))
        rf_list = np.tile(RF, (4, 1))
        tofs = [TOF, 2 * TOF, 3 * TOF]
        out = solve_lambert_batch(r0_list, rf_list, tofs, MU)
        for j, tof in enumerate(tofs):
            sol = solve_lambert(R0, RF, tof, MU)
            for i in range(4):
                np.testing.assert_allclose(out[i, j, 0, :], sol.v0, atol=1e-12)
                np.testing.assert_allclose(out[i, j, 1, :], sol.vf, atol=1e-12)

    def test_failed_combination_is_nan(self):
        """无解组合（弦长为零）填 NaN，不影响其余组合。"""
        r0_list = [R0, R0]
        rf_list = [RF, R0]  # 第二组 r0 == rf，弦长为零
        out = solve_lambert_batch(r0_list, rf_list, [TOF], MU)
        assert np.all(np.isfinite(out[0]))
        assert np.all(np.isnan(out[1]))
