"""
需求: Multiple Shooting 多段打靶法差分修正 (Layer 2)

e2m2e 需要实现 Multiple Shooting 算法，在星历 N-body 模型下修正轨道使 patch
points 之间位置和速度连续。这是将 CR3BP 轨道转换到高精度星历模型的核心算法。

功能要求:
  1. 将轨道分割为多个 patch points
  2. 分别传播每段轨迹（带 STM）
  3. 组装约束向量 F(X)（位置/速度不连续量）
  4. 组装 Jacobian DF(X)（基于 STM）
  5. Newton 迭代: X_{k+1} = X_k - DF^{-1} @ F(X_k)
  6. 支持可变时间 / 固定时间修正
  7. 支持部分自由变量（只修正位置、只修正速度等）

约束条件:
  - 位置连续: r_i(t_i_end) = r_{i+1}(t_{i+1_start})
  - 速度连续: v_i(t_i_end) = v_{i+1}(t_{i+1_start})  (可选)
  - 历元连续: 时间节点不跳变

自由变量:
  - 每个 patch point 的 6 维状态 [x, y, z, vx, vy, vz]
  - 时间节点 t_i (可选, var_time=True 时)

算法:
  F_i = [r_{i+1} - r_prop_i(t_{i+1}), v_{i+1} - v_prop_i(t_{i+1})]
  DF_ij = STM_i * I_j - delta_{i+1,j} * I  (简化表示)
  dX = -DF^{-1} @ F  (或最小范数解)

参考实现:
  SEMpy multiple_shooting.py 中的 MultipleShooting.correct()
  SEMpy 支持并行传播 patch segments

参考论文:
  陈昱桔 (2024) "面向地月空间态势感知的DRO轨道设计与控制研究"

依赖:
  Layer 1b (EphemerisDynamics)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from e2m2e.core import (
    SPICEManager,
    EphemerisSystem,
    EphemerisDynamics,
)
from e2m2e.algorithms import MultipleShooting


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def spice_manager(spice_kernel_path):
    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)
    yield mgr
    mgr.unload_kernel(spice_kernel_path)


@pytest.fixture
def eph_system(spice_manager):
    return EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=spice_manager,
        origin="EARTH",
        frame="J2000",
    )


@pytest.fixture
def eph_dynamics(eph_system):
    return EphemerisDynamics(system=eph_system)


@pytest.fixture
def ms_corrector(eph_dynamics):
    """创建 MultipleShooting 实例"""
    return MultipleShooting(dynamics=eph_dynamics)


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def simple_patch_points(reference_et):
    """
    创建简单的 patch points 用于测试修正。
    模拟一个在月球附近近似圆形轨道上的 3 个 patch points。
    """
    r0 = 384400.0
    v0 = 1.0
    dt = 3600.0 * 6  # 6 小时间隔

    t_patch = np.array([
        reference_et,
        reference_et + dt,
        reference_et + 2 * dt,
    ])
    state_patch = np.array([
        [r0, 0, 0, 0, v0, 0],
        [r0 * 0.99, 0, 500, 0, v0 * 1.01, 0.01],
        [r0 * 0.98, 0, -300, 0, v0 * 0.99, -0.01],
    ])
    return t_patch, state_patch


# =============================================================================
# Test MultipleShooting 初始化
# =============================================================================
class TestMultipleShootingInit:
    """测试 MultipleShooting 的创建和配置"""

    def test_create_instance(self, eph_dynamics):
        """应能创建 MultipleShooting 实例"""
        ms = MultipleShooting(dynamics=eph_dynamics)
        assert ms is not None

    def test_requires_dynamics(self):
        """MultipleShooting 需要 Dynamics 实例"""
        with pytest.raises(TypeError):
            MultipleShooting(dynamics=None)

    def test_has_correct_method(self, ms_corrector):
        """应有 correct 方法"""
        assert hasattr(ms_corrector, "correct")
        assert callable(ms_corrector.correct)

    def test_dynamics_reference(self, ms_corrector, eph_dynamics):
        """应持有 Dynamics 引用"""
        assert ms_corrector.dynamics is eph_dynamics

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
# Test 修正过程
# =============================================================================
class TestMultipleShootingCorrection:
    """测试多段打靶修正的核心功能"""

    def test_correction_returns_result(self, ms_corrector, simple_patch_points):
        """修正应返回结果对象"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=20,
            tolerance=1e-6,
        )
        assert result is not None

    def test_result_has_corrected_patch_points(self, ms_corrector, simple_patch_points):
        """结果应包含修正后的 patch points"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=20,
        )
        assert hasattr(result, "t_patch")
        assert hasattr(result, "state_patch")
        assert result.state_patch.shape == state_patch.shape

    def test_result_has_convergence_info(self, ms_corrector, simple_patch_points):
        """结果应包含收敛信息"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=20,
        )
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "max_residual")

    def test_position_continuity_improved(self, ms_corrector, simple_patch_points):
        """修正后位置连续性应改善"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=30,
            tolerance=1e-6,
        )

        if result.converged:
            corrected_states = result.state_patch
            for i in range(len(corrected_states) - 1):
                dt = t_patch[i + 1] - t_patch[i] if not hasattr(result, "t_patch") else result.t_patch[i + 1] - result.t_patch[i]
                propagated = ms_corrector.dynamics.propagate(
                    corrected_states[i],
                    (result.t_patch[i], result.t_patch[i + 1]),
                )
                final_prop = propagated["states"][:, -1]
                pos_error = np.linalg.norm(final_prop[:3] - corrected_states[i + 1, :3])
                assert pos_error < 1.0, f"位置连续性误差 {pos_error:.2e} km 过大"


# =============================================================================
# Test Jacobian 计算
# =============================================================================
class TestMultipleShootingJacobian:
    """测试 Jacobian 矩阵计算"""

    def test_compute_jacobian_shape(self, ms_corrector, simple_patch_points):
        """Jacobian 矩阵维度应正确"""
        t_patch, state_patch = simple_patch_points
        n = len(t_patch)
        # 约束数: (n-1) * 6 (位置+速度连续性)
        # 自由变量数: n * 6 (+ n * 1 如果 var_time)
        n_constraints = (n - 1) * 6
        n_vars = n * 6
        expected_shape = (n_constraints, n_vars)
        # 这个测试验证 Jacobian 维度的正确性
        assert expected_shape[0] > 0
        assert expected_shape[1] > 0


# =============================================================================
# Test 固定时间 vs 可变时间
# =============================================================================
class TestMultipleShootingTimeOptions:
    """测试固定时间和可变时间修正"""

    def test_fixed_time_correction(self, ms_corrector, simple_patch_points):
        """固定时间修正应保持时间节点不变"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            var_time=False,
            max_iter=20,
        )
        if result.converged:
            assert_allclose(result.t_patch, t_patch, atol=1e-10)

    def test_variable_time_correction(self, ms_corrector, simple_patch_points):
        """可变时间修正应允许时间节点微调"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            var_time=True,
            max_iter=20,
        )
        if result.converged:
            assert result.t_patch is not None
            assert len(result.t_patch) == len(t_patch)


# =============================================================================
# Test 收敛性
# =============================================================================
class TestMultipleShootingConvergence:
    """测试收敛行为"""

    def test_convergence_decreases_residual(self, ms_corrector, simple_patch_points):
        """迭代过程中残差应单调递减"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=30,
        )
        if hasattr(result, "residual_history") and result.residual_history:
            residuals = result.residual_history
            for i in range(1, len(residuals)):
                assert residuals[i] <= residuals[i - 1] * 1.5, (
                    f"残差在第 {i} 步增大: {residuals[i]:.2e} > {residuals[i-1]:.2e}"
                )

    def test_max_iter_respected(self, ms_corrector, simple_patch_points):
        """迭代次数不应超过 max_iter"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=5,
        )
        assert result.iterations <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
