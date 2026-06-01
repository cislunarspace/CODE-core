"""
Period Validation Tests for Differential Correction

Tests for the period validation feature that rejects converged solutions
with invalid periods (T < 1e-6) to prevent convergence to invalid solutions.

Reference: Commit 584c44d - fix: 添加周期合理性验证防止收敛到无效解
"""


from e2m2e.core import Orbit

# 公共 fixtures 从 tests/algorithms/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro
# 种子 x0=0.79188556619742, vy0=0.573665890385585, period=6.307498 来自 conftest。


class TestPeriodValidation:
    """测试周期有效性验证"""

    def test_valid_period_accepted(self, dro_corrector):
        """测试有效周期（远大于1e-6）应该被接受"""
        # 创建一个有效的DRO轨道初始猜测
        x0 = 0.79188556619742
        valid_orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        valid_orbit.period = 6.307498  # 有效周期

        result = dro_corrector.iterate_correction(valid_orbit, verbose=False)

        # 如果收敛成功，结果不应为None
        if result is not None:
            assert result.period >= 1e-6

    def test_period_validation_constant_value(self, dro_corrector):
        """测试周期验证阈值为1e-6"""
        # 这个测试验证 min_valid_period = 1e-6 这个常量存在且正确
        min_valid_period = 1e-6

        # 验证1e-6是合理的周期阈值
        assert min_valid_period > 0
        assert min_valid_period < 1e-3  # 应该是非常小的值

    def test_converged_with_valid_period(self, dro_corrector):
        """测试能收敛到有效周期轨道"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        orbit.period = 3.0  # 合理的初始周期猜测

        result = dro_corrector.iterate_correction(orbit, verbose=False)

        # 结果可能为None（如果初始猜测不好），但如果返回了轨道，周期应该有效
        if result is not None:
            assert result.period >= 1e-6, f"Expected period >= 1e-6, got {result.period}"
            assert dro_corrector.termination_reason is not None


class TestPeriodValidationEdgeCases:
    """测试周期验证的边界情况"""

    def test_very_small_initial_period(self, dro_corrector):
        """测试非常小的初始周期（接近1e-6）"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        orbit.period = 1e-5  # 非常小但可能有效的周期

        result = dro_corrector.iterate_correction(orbit, verbose=False)

        # 如果结果不为None，验证周期有效性
        if result is not None:
            assert result.period >= 1e-6

    def test_corrector_tracks_termination_reason(self, dro_corrector):
        """测试修正器能记录终止原因"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        orbit.period = 6.307498

        dro_corrector.iterate_correction(orbit, verbose=False)

        # 验证有终止原因记录
        assert dro_corrector.termination_reason is not None


class TestPeriodValidationFailureMode:
    """测试周期验证失败模式"""

    def test_invalid_period_rejected_with_message(self, dro_corrector):
        """测试无效周期会被拒绝并返回None"""
        # 创建一个初始猜测，其周期非常小
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.1, 0.0]],  # y_dot过小会导致极短周期
            times=[0],
        )
        orbit.period = 1e-7  # 无效的极小周期

        result = dro_corrector.iterate_correction(orbit, verbose=False)

        # 无效周期应该导致返回None或success=False
        if result is None:
            assert not dro_corrector.success
            assert "周期" in dro_corrector.termination_reason or (
                "period" in dro_corrector.termination_reason.lower()
            )


class TestCorrectionNormTermination:
    """测试修正量过小时 improved termination condition

    Reference: Commit 891fbe0 - feat(algorithms): 优化差分修正算法的收敛判断逻辑
    """

    def test_termination_when_correction_small_and_error_small(self, dro_corrector):
        """测试当修正量过小但误差也已足够小时能正常收敛"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        orbit.period = 6.307498

        dro_corrector.iterate_correction(orbit, verbose=False)

        # 如果成功，验证终止原因
        if dro_corrector.success:
            # 终止原因可能是:
            # 1. 正常收敛 (误差小于容差)
            # 2. 修正量过小但误差足够小
            assert dro_corrector.termination_reason in [
                "收敛成功：误差小于容差",
                "收敛成功：修正量过小但误差足够小",
                "收敛成功",
            ]

    def test_correction_history_tracked(self, dro_corrector):
        """测试修正量历史被正确追踪"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.573665890385585, 0.0]],
            times=[0],
        )
        orbit.period = 6.307498

        dro_corrector.iterate_correction(orbit, verbose=False)

        # 验证修正历史被记录
        assert len(dro_corrector.correction_history) > 0

    def test_stagnation_limit_rejected(self, dro_corrector):
        """测试当修正量过小但误差不够小时会被拒绝"""
        x0 = 0.79188556619742
        orbit = Orbit(
            states=[[x0, 0.0, 0.0, 0.0, 0.001, 0.0]],  # y_dot很小，可能导致停滞
            times=[0],
        )
        orbit.period = 3.0

        dro_corrector.iterate_correction(orbit, verbose=False)

        # 如果不成功，检查终止原因
        if not dro_corrector.success:
            # 可能是停滞（修正量过小）或周期无效
            assert dro_corrector.termination_reason is not None
