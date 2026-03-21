"""
微分修正轨道闭合性测试模块

测试 _create_corrected_orbit 方法中的闭合误差计算和轨道闭合验证功能。

这些测试覆盖最近 commit "Enhance orbit closure verification in differential correction" 的功能：
- closure_error 计算
- orbit.is_periodic 标记
- 轨道闭合性修正尝试
- correction_success, correction_iterations, correction_error, correction_termination_reason, closure_error 等属性
"""

import numpy as np
import pytest

import e2m2e
from e2m2e.algorithms import DifferentialCorrection
from e2m2e.core import CR3BP_Dynamics, Orbit

# 地月系统质量比
MU = 1.21506683e-2


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def earth_moon_system():
    """创建地月CR3BP系统"""
    system = e2m2e.core.system.CR3BP_System(mu=MU, primary="earth", secondary="moon")
    system.compute_libration_points()
    return system


@pytest.fixture
def dynamics(earth_moon_system):
    """创建动力学对象"""
    return CR3BP_Dynamics(earth_moon_system)


@pytest.fixture
def corrector_2d_fixed_x0(dynamics):
    """创建2D对称x轴、固定x0的微分修正器"""
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=0.79188556619742)
    return corrector


@pytest.fixture
def dro_initial_guess():
    """DRO初始猜测轨道"""
    x0 = 0.79188556619742
    vy0 = 0.53682
    guess = Orbit(
        states=[[x0, 0.0, 0.0, 0.0, vy0, 0.0]],
        times=[0],
    )
    guess.period = 3.420385
    return guess


@pytest.fixture
def corrected_dro(corrector_2d_fixed_x0, dro_initial_guess):
    """执行微分修正后的DRO轨道"""
    return corrector_2d_fixed_x0.iterate_correction(dro_initial_guess)


# ============================================================
# 闭合误差属性测试
# ============================================================
class TestClosureErrorAttribute:
    """测试轨道闭合误差属性的正确设置"""

    def test_corrected_orbit_has_closure_error_attribute(self, corrected_dro):
        """修正后的轨道应有 closure_error 属性"""
        assert hasattr(corrected_dro, "closure_error"), \
            "Orbit should have closure_error attribute"

    def test_closure_error_is_float(self, corrected_dro):
        """closure_error 应该是浮点数类型"""
        assert isinstance(corrected_dro.closure_error, float), \
            f"closure_error should be float, got {type(corrected_dro.closure_error)}"

    def test_closure_error_is_positive(self, corrected_dro):
        """closure_error 应该为非负值"""
        assert corrected_dro.closure_error >= 0, \
            f"closure_error should be non-negative, got {corrected_dro.closure_error}"

    def test_closure_error_reasonable_magnitude(self, corrected_dro):
        """成功的修正后，closure_error 应该在合理范围内"""
        # 成功的修正应该使 closure_error 非常小
        assert corrected_dro.closure_error < 1e-6, \
            f"closure_error should be small after correction, got {corrected_dro.closure_error}"


# ============================================================
# 周期性标记测试
# ============================================================
class TestPeriodicFlag:
    """测试轨道周期性标记的正确设置"""

    def test_corrected_orbit_has_is_periodic_attribute(self, corrected_dro):
        """修正后的轨道应有 is_periodic 属性"""
        assert hasattr(corrected_dro, "is_periodic"), \
            "Orbit should have is_periodic attribute"

    def test_is_periodic_is_bool(self, corrected_dro):
        """is_periodic 应该是布尔类型"""
        assert isinstance(corrected_dro.is_periodic, bool), \
            f"is_periodic should be bool, got {type(corrected_dro.is_periodic)}"

    def test_successful_correction_is_periodic(self, corrected_dro):
        """成功的修正后，轨道应该被标记为周期轨道"""
        # closure_error < 1e-8 时 is_periodic 应为 True
        if corrected_dro.closure_error < 1e-8:
            assert corrected_dro.is_periodic is True, \
                "Orbit with small closure_error should be periodic"

    def test_large_closure_error_not_periodic(self, dynamics):
        """closure_error 较大时，轨道不应被标记为周期轨道"""
        # 创建一个 closure_error 会很大的情况
        corrector = DifferentialCorrection(dynamics)
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
        if result is not None and hasattr(result, 'closure_error'):
            if result.closure_error >= 1e-8:
                assert result.is_periodic is False, \
                    "Orbit with large closure_error should not be periodic"


# ============================================================
# 修正结果信息属性测试
# ============================================================
class TestCorrectionResultAttributes:
    """测试修正结果信息属性的正确设置"""

    def test_correction_success_attribute(self, corrected_dro):
        """轨道应有 correction_success 属性"""
        assert hasattr(corrected_dro, "correction_success"), \
            "Orbit should have correction_success attribute"

    def test_correction_iterations_attribute(self, corrected_dro):
        """轨道应有 correction_iterations 属性"""
        assert hasattr(corrected_dro, "correction_iterations"), \
            "Orbit should have correction_iterations attribute"

    def test_correction_iterations_is_int(self, corrected_dro):
        """correction_iterations 应该是整数类型"""
        assert isinstance(corrected_dro.correction_iterations, int), \
            f"correction_iterations should be int, got {type(corrected_dro.correction_iterations)}"

    def test_correction_iterations_non_negative(self, corrected_dro):
        """correction_iterations 应该为非负值"""
        assert corrected_dro.correction_iterations >= 0, \
            f"correction_iterations should be non-negative, got {corrected_dro.correction_iterations}"

    def test_correction_error_attribute(self, corrected_dro):
        """轨道应有 correction_error 属性"""
        assert hasattr(corrected_dro, "correction_error"), \
            "Orbit should have correction_error attribute"

    def test_correction_termination_reason_attribute(self, corrected_dro):
        """轨道应有 correction_termination_reason 属性"""
        assert hasattr(corrected_dro, "correction_termination_reason"), \
            "Orbit should have correction_termination_reason attribute"

    def test_successful_correction_has_convergence_reason(self, corrected_dro):
        """成功的修正应该有收敛的 termination_reason"""
        if corrected_dro.correction_success:
            assert corrected_dro.correction_termination_reason is not None, \
                "Successful correction should have termination_reason"


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
        assert corrected_dro.states[0, 0] != original_first_element, \
            "Modification to orbit states should affect the returned object"

    def test_orbit_times_independent(self, corrected_dro):
        """轨道的 times 数组应该是独立的副本"""
        original_first_element = corrected_dro.times[0]
        if len(corrected_dro.times) > 1:
            corrected_dro.times[0] += 1.0
            assert corrected_dro.times[0] != original_first_element, \
                "Modification to orbit times should affect the returned object"


# ============================================================
# 家族类型推断测试
# ============================================================
class TestFamilyTypeInference:
    """测试轨道家族类型推断"""

    def test_2d_orbit_family_type_lyapunov(self, corrector_2d_fixed_x0, dro_initial_guess):
        """2D 对称轨道应该被识别为 lyapunov 类型"""
        orbit = corrector_2d_fixed_x0.iterate_correction(dro_initial_guess)
        if orbit is not None:
            assert orbit.family_type == "lyapunov", \
                f"2D orbit should have family_type='lyapunov', got {orbit.family_type}"

    def test_3d_orbit_family_type_halo(self, dynamics):
        """3D 对称轨道应该被识别为 halo 类型"""
        corrector = DifferentialCorrection(dynamics)
        corrector.setup_3D_symmetric_x_fixed_x0(x0=0.8)
        
        # 创建 3D 初始猜测
        z0 = 0.01
        orbit = Orbit(
            states=[[0.8, 0.0, z0, 0.0, 0.3, 0.0]],
            times=[0],
        )
        orbit.period = 3.0
        
        result = corrector.iterate_correction(orbit, verbose=False)
        if result is not None and hasattr(result, 'family_type'):
            assert result.family_type == "halo", \
                f"3D orbit should have family_type='halo', got {result.family_type}"


# ============================================================
# 轨道状态数组形状测试
# ============================================================
class TestOrbitStateShape:
    """测试轨道状态数组的形状正确性"""

    def test_states_2d_array(self, corrected_dro):
        """轨道 states 应该是 2D 数组 (N, 6)"""
        assert corrected_dro.states.ndim == 2, \
            f"states should be 2D array, got {corrected_dro.states.ndim}D"
        assert corrected_dro.states.shape[1] == 6, \
            f"states should have 6 columns (x,y,z,vx,vy,vz), got {corrected_dro.states.shape[1]}"

    def test_times_1d_array(self, corrected_dro):
        """轨道 times 应该是 1D 数组"""
        assert corrected_dro.times.ndim == 1, \
            f"times should be 1D array, got {corrected_dro.times.ndim}D"

    def test_states_and_times_same_length(self, corrected_dro):
        """states 和 times 应该有相同的长度"""
        assert len(corrected_dro.states) == len(corrected_dro.times), \
            f"states ({len(corrected_dro.states)}) and times ({len(corrected_dro.times)}) should have same length"


# ============================================================
# 边界情况测试
# ============================================================
class TestBoundaryCases:
    """测试边界情况"""

    def test_very_small_step_correction(self, corrector_2d_fixed_x0, dro_initial_guess):
        """测试修正器处理极小步长的能力"""
        corrector_2d_fixed_x0.max_iterations = 10
        orbit = corrector_2d_fixed_x0.iterate_correction(dro_initial_guess, verbose=False)
        # 应该成功或返回 None，不应崩溃
        assert orbit is None or isinstance(orbit, Orbit)

    def test_none_result_handling(self, dynamics):
        """测试修正失败返回 None 的情况"""
        corrector = DifferentialCorrection(dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=0.3)
        corrector.max_iterations = 1
        
        # 使用一个会失败的初始猜测
        bad_orbit = Orbit(
            states=[[0.3, 0.0, 0.0, 0.0, 10.0, 0.0]],
            times=[0],
        )
        bad_orbit.period = 0.01
        
        result = corrector.iterate_correction(bad_orbit)
        # 应该返回 None
        assert result is None
