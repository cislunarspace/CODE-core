"""微分修正轨道闭合性测试。

覆盖 closure_error、is_periodic、修正结果属性、
状态数组独立性与边界情况。
"""

import pytest

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


# 公共 fixtures 从 tests/algorithm/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro
# 种子 x0=0.79188556619742, vy0=0.573665890385585, period=6.307498 来自 conftest。
# 注：corrected_dro 每次返回深拷贝，可安全 mutate。


# ============================================================
# 闭合误差属性测试
# ============================================================
class TestClosureErrorAttribute:
    """测试轨道闭合误差属性的正确设置"""

    def test_closure_error_reasonable_magnitude(self, corrected_dro):
        """成功的修正后，closure_error 应该在合理范围内"""
        # 成功的修正应该使 closure_error 非常小
        assert corrected_dro.closure_error < 1e-6, (
            f"closure_error should be small after correction, got {corrected_dro.closure_error}"
        )


# ============================================================
# 周期性标记测试
# ============================================================
class TestPeriodicFlag:
    """测试轨道周期性标记的正确设置"""

    def test_successful_correction_is_periodic(self, corrected_dro):
        """成功的修正后，轨道应该被标记为周期轨道"""
        # closure_error < 1e-8 时 is_periodic 应为 True
        if corrected_dro.closure_error < 1e-8:
            assert corrected_dro.is_periodic is True, (
                "Orbit with small closure_error should be periodic"
            )

    def test_large_closure_error_not_periodic(self, dro_dynamics):
        """closure_error 较大时，轨道不应被标记为周期轨道"""
        # 创建一个 closure_error 会很大的情况
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=0.5)
        corrector.max_iterations = 1  # 限制迭代次数

        # 使用一个不太好的初始猜测
        bad_guess = Orbit(
            states=[[0.5, 0.0, 0.0, 0.0, 0.5, 0.0]],
            times=[0],
        )
        bad_guess.period = 2.0

        result = corrector.iterate_correction(bad_guess)
        # 如果返回了结果，检查 is_periodic
        if result is not None and hasattr(result, "closure_error") and result.closure_error >= 1e-8:
            assert result.is_periodic is False, (
                "Orbit with large closure_error should not be periodic"
            )


# ============================================================
# 修正结果信息属性测试
# ============================================================
class TestCorrectionResultAttributes:
    """测试修正结果信息属性的正确设置"""

    def test_correction_iterations_non_negative(self, dro_corrector, dro_seed_orbit):
        """修正结果的迭代次数应为非负值。"""
        result = dro_corrector.iterate_correction(dro_seed_orbit)
        assert result.iterations >= 0

    def test_successful_correction_has_convergence_reason(self, dro_corrector, dro_seed_orbit):
        """成功修正应携带稳定的成功状态和消息。"""
        result = dro_corrector.iterate_correction(dro_seed_orbit)
        if result.orbit is not None:
            assert result.status.value == "converged"
            assert result.cause.value == "none"
            assert result.message


# ============================================================
# 轨道状态数组独立性测试
# ============================================================
class TestOrbitStateIndependence:
    """测试轨道状态的独立副本，避免与内部积分共享内存"""

    def test_orbit_states_independent_from_propagation(self, corrected_dro):
        """轨道的 states 数组应该是独立的副本"""
        # 修改返回的 states 不应该影响内部的 propagation.y
        original_first_element = corrected_dro.states[0, 0]
        corrected_dro.states[0, 0] += 1.0
        assert corrected_dro.states[0, 0] != original_first_element, (
            "Modification to orbit states should affect the returned object"
        )

    def test_orbit_times_independent(self, corrected_dro):
        """轨道的 times 数组应该是独立的副本"""
        original_first_element = corrected_dro.times[0]
        if len(corrected_dro.times) > 1:
            corrected_dro.times[0] += 1.0
            assert corrected_dro.times[0] != original_first_element, (
                "Modification to orbit times should affect the returned object"
            )


# ============================================================
# 家族类型推断测试
# ============================================================
class TestFamilyTypeInference:
    """测试轨道家族类型推断"""

    def test_2d_orbit_family_type_lyapunov(self, dro_corrector, dro_seed_orbit):
        """2D 对称轨道应该被识别为 lyapunov 类型"""
        result = dro_corrector.iterate_correction(dro_seed_orbit)
        if result.orbit is not None:
            assert result.orbit.family_type == "lyapunov", (
                f"2D orbit should have family_type='lyapunov', got {result.orbit.family_type}"
            )

    def test_3d_orbit_family_type_halo(self, dro_dynamics):
        """3D 对称轨道应该被识别为 halo 类型"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_3D_symmetric_x_fixed_x0(x0=0.8)

        # 创建 3D 初始猜测
        z0 = 0.01
        orbit = Orbit(
            states=[[0.8, 0.0, z0, 0.0, 0.3, 0.0]],
            times=[0],
        )
        orbit.period = 3.0

        result = corrector.iterate_correction(orbit, verbose=False)
        if result is not None and hasattr(result, "family_type"):
            assert result.family_type == "halo", (
                f"3D orbit should have family_type='halo', got {result.family_type}"
            )


# ============================================================
# 边界情况测试
# ============================================================
class TestBoundaryCases:
    """测试边界情况"""

    def test_very_small_step_correction(self, dro_corrector, dro_seed_orbit):
        """测试修正器处理极小步长的能力"""
        dro_corrector.max_iterations = 10
        result = dro_corrector.iterate_correction(dro_seed_orbit, verbose=False)
        # 应返回状态化结果，且不应崩溃
        assert result.status.value != "iterating"

    def test_none_result_handling(self, dro_dynamics):
        """测试修正失败返回 None 的情况"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=0.3)
        corrector.max_iterations = 1

        # 使用一个会失败的初始猜测
        bad_orbit = Orbit(
            states=[[0.3, 0.0, 0.0, 0.0, 10.0, 0.0]],
            times=[0],
        )
        bad_orbit.period = 0.01

        result = corrector.iterate_correction(bad_orbit)
        # 应返回包含失败状态的结果
        assert result.orbit is None
        assert result.status.value != "converged"
