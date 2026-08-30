"""Multiple Shooting 多段打靶法测试（Layer 2）——签名与输入校验契约。

只保留不驱动真实 SPICE 多重打靶计算的轻量覆盖：初始化配置、patch
points 输入校验（fail fast）与 verbose 参数签名契约。真实修正、Jacobian、
固定/可变时间选项等重度 SPICE 计算测试已按 ADR 0037 预算决策移出
默认套件。
"""

import inspect

import numpy as np
import pytest

from e2m2e.algorithm.solver.multiple_shooting import MultipleShooting

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
]


# =============================================================================
# Fixtures
# =============================================================================
# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, spice_syn_j2000,
#   reference_epoch, spice_kernel_path


@pytest.fixture
def ms_corrector(spice_eph_dynamics):
    """创建 MultipleShooting 实例"""
    return MultipleShooting(dynamics=spice_eph_dynamics)


@pytest.fixture
def simple_patch_points(reference_et):
    """
    创建简单的 patch points 用于测试修正。
    模拟一个在月球附近近似圆形轨道上的 3 个 patch points。
    """
    r0 = 384400.0
    v0 = 1.0
    dt = 3600.0 * 6  # 6 小时间隔

    t_patch = np.array(
        [
            reference_et,
            reference_et + dt,
            reference_et + 2 * dt,
        ]
    )
    state_patch = np.array(
        [
            [r0, 0, 0, 0, v0, 0],
            [r0 * 0.99, 0, 500, 0, v0 * 1.01, 0.01],
            [r0 * 0.98, 0, -300, 0, v0 * 0.99, -0.01],
        ]
    )
    return t_patch, state_patch


# =============================================================================
# Test MultipleShooting 初始化
# =============================================================================
class TestMultipleShootingInit:
    """测试 MultipleShooting 的创建和配置"""

    def test_create_instance(self, spice_eph_dynamics):
        """应能创建 MultipleShooting 实例"""
        ms = MultipleShooting(dynamics=spice_eph_dynamics)
        assert ms is not None

    def test_requires_dynamics(self):
        """MultipleShooting 需要 Dynamics 实例"""
        with pytest.raises(TypeError):
            MultipleShooting(dynamics=None)

    def test_has_correct_method(self, ms_corrector):
        """应有 correct 方法"""
        assert hasattr(ms_corrector, "correct")
        assert callable(ms_corrector.correct)

    def test_dynamics_reference(self, ms_corrector, spice_eph_dynamics):
        """应持有 Dynamics 引用"""
        assert ms_corrector.dynamics is spice_eph_dynamics

    def test_default_parameters(self, ms_corrector):
        """应有合理的默认参数"""
        assert hasattr(ms_corrector, "max_iter")
        assert hasattr(ms_corrector, "tolerance")
        assert ms_corrector.max_iter > 0
        assert ms_corrector.tolerance > 0


# =============================================================================
# Test patch points 输入验证
# =============================================================================
class TestPatchPointsValidation:
    """测试 patch points 输入格式验证"""

    def test_accepts_valid_patch_points(self, ms_corrector, simple_patch_points):
        """应接受正确格式的 patch points"""
        t_patch, state_patch = simple_patch_points
        # 不执行修正，只验证输入
        assert t_patch.shape[0] == state_patch.shape[0]
        assert state_patch.shape[1] == 6

    def test_rejects_mismatched_dimensions(self, ms_corrector, reference_et):
        """时间与状态数量不匹配应报错"""
        t_patch = np.array([reference_et, reference_et + 3600])
        state_patch = np.array([[384400, 0, 0, 0, 1, 0]])
        with pytest.raises(ValueError):
            ms_corrector.correct(t_patch, state_patch)

    def test_rejects_empty_patch_points(self, ms_corrector):
        """空 patch points 应报错"""
        with pytest.raises(ValueError):
            ms_corrector.correct(np.array([]), np.array([]).reshape(0, 6))


# =============================================================================
# 需求: MultipleShooting.correct() 添加 verbose 参数支持 tqdm 进度条
# =============================================================================
class TestMultipleShootingVerbose:
    """verbose 参数的签名契约（轻量，不驱动真实修正）。

    功能契约：correct() 接受 verbose 参数 (bool, 默认 False)；
    verbose=True 的 tqdm 行为不在此覆盖。
    """

    def test_verbose_parameter_exists(self, ms_corrector):
        """correct() 方法应接受 verbose 参数"""
        sig = inspect.signature(ms_corrector.correct)
        assert "verbose" in sig.parameters

    def test_verbose_parameter_default_false(self, ms_corrector):
        """verbose 参数默认值应为 False"""
        sig = inspect.signature(ms_corrector.correct)
        assert sig.parameters["verbose"].default is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
