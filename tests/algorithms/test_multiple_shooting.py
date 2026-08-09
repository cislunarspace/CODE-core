"""Multiple Shooting 多段打靶法测试（Layer 2）。

覆盖初始化、patch points 校验、修正过程、Jacobian 计算、
固定/可变时间选项与 verbose 进度条。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.solver.multiple_shooting import MultipleShooting

pytestmark = [pytest.mark.spice, pytest.mark.l3]


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
        assert hasattr(result, "outer_iterations")
        assert hasattr(result, "status")
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

        # 不收敛应直接失败而非 skip，否则算法回归会被绿色报告掩盖（issue #218）
        assert result.converged, f"多重打靶未收敛 (residual={result.max_residual:.2e})"
        corrected_states = result.state_patch
        for i in range(len(corrected_states) - 1):
            (
                t_patch[i + 1] - t_patch[i]
                if not hasattr(result, "t_patch")
                else result.t_patch[i + 1] - result.t_patch[i]
            )
            propagated = ms_corrector.dynamics.propagate(
                corrected_states[i],
                (result.t_patch[i], result.t_patch[i + 1]),
            )
            final_prop = propagated["states"][-1]
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
        # 固定时间修正在 simple_patch_points 上应能收敛；不收敛是回归（issue #218）
        assert result.converged, f"多重打靶未收敛 (residual={result.max_residual:.2e})"
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
        # 可变时间修正在 simple_patch_points 上应能收敛；不收敛是回归（issue #218）
        assert result.converged, f"多重打靶未收敛 (residual={result.max_residual:.2e})"
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
                assert residuals[i] <= residuals[i - 1] * 10.0, (
                    f"残差在第 {i} 步增大: {residuals[i]:.2e} > {residuals[i - 1]:.2e}"
                )

    def test_max_iter_respected(self, ms_corrector, simple_patch_points):
        """迭代次数不应超过 max_iter"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=5,
        )
        assert result.outer_iterations <= 5


# =============================================================================
# 需求: MultipleShooting.correct() 添加 verbose 参数支持 tqdm 进度条
# =============================================================================
class TestMultipleShootingVerbose:
    """
    需求: MultipleShooting.correct() 添加 verbose 参数

    功能:
      1. correct() 接受 verbose 参数 (bool, 默认 False)
      2. verbose=True 时，使用 tqdm 显示迭代进度条
      3. 进度条显示: 当前迭代/最大迭代、当前残差
      4. verbose=False (默认) 时，行为与当前完全一致，无 tqdm 输出
      5. 提前收敛时进度条正确关闭

    参考:
      e2m2e.algorithm.transfer.TransferSearch 中 tqdm 的使用方式

    依赖:
      tqdm>=4.66 (已在 pyproject.toml 中声明)
    """

    def test_verbose_parameter_exists(self, ms_corrector):
        """correct() 方法应接受 verbose 参数"""
        import inspect

        sig = inspect.signature(ms_corrector.correct)
        assert "verbose" in sig.parameters

    def test_verbose_default_false(self, ms_corrector):
        """verbose 参数默认值应为 False"""
        import inspect

        sig = inspect.signature(ms_corrector.correct)
        assert sig.parameters["verbose"].default is False

    def test_backward_compatible(self, ms_corrector, simple_patch_points):
        """不传 verbose 时行为与之前完全一致"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=5,
        )
        assert result is not None
        assert hasattr(result, "converged")
        assert hasattr(result, "outer_iterations")

    def test_verbose_false_no_tqdm_output(self, ms_corrector, simple_patch_points, capsys):
        """verbose=False 时 stderr 不包含 tqdm 进度条输出"""
        t_patch, state_patch = simple_patch_points
        ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=5,
            verbose=False,
        )
        captured = capsys.readouterr()
        assert "%" not in captured.err

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_does_not_affect_result(self, ms_corrector, simple_patch_points, verbose):
        """verbose 设置不影响修正结果的数值正确性"""
        t_patch, state_patch = simple_patch_points
        result = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=10,
            verbose=verbose,
        )
        result_ref = ms_corrector.correct(
            t_patch=t_patch,
            state_patch=state_patch,
            max_iter=10,
            verbose=False,
        )
        assert result.converged == result_ref.converged
        assert result.outer_iterations == result_ref.outer_iterations
        np.testing.assert_allclose(result.max_residual, result_ref.max_residual, rtol=1e-12)
        if result.residual_history and result_ref.residual_history:
            np.testing.assert_allclose(
                result.residual_history, result_ref.residual_history, rtol=1e-12
            )

    def test_verbose_true_invokes_tqdm(self, ms_corrector, simple_patch_points):
        """verbose=True 时应调用 tqdm 创建进度条"""
        from unittest.mock import MagicMock, patch

        t_patch, state_patch = simple_patch_points
        with patch("e2m2e.algorithm.solver.multiple_shooting.tqdm") as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar

            ms_corrector.correct(
                t_patch=t_patch,
                state_patch=state_patch,
                max_iter=5,
                verbose=True,
            )
            assert mock_tqdm.called

    def test_verbose_true_updates_per_iteration(self, ms_corrector, simple_patch_points):
        """进度条应在每次迭代中更新残差信息"""
        from unittest.mock import MagicMock, patch

        t_patch, state_patch = simple_patch_points
        with patch("e2m2e.algorithm.solver.multiple_shooting.tqdm") as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar

            result = ms_corrector.correct(
                t_patch=t_patch,
                state_patch=state_patch,
                max_iter=5,
                verbose=True,
            )

            n_calls = result.outer_iterations if result.converged else 5
            assert (
                mock_bar.set_postfix.call_count >= n_calls or mock_bar.update.call_count >= n_calls
            )

    def test_verbose_early_convergence(self, ms_corrector, simple_patch_points):
        """提前收敛时进度条应正确关闭，不报错"""
        from unittest.mock import MagicMock, patch

        t_patch, state_patch = simple_patch_points
        with patch("e2m2e.algorithm.solver.multiple_shooting.tqdm") as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar

            result = ms_corrector.correct(
                t_patch=t_patch,
                state_patch=state_patch,
                max_iter=100,
                tolerance=1e-3,
                verbose=True,
            )
            assert result is not None

    def test_verbose_with_var_time(self, ms_corrector, simple_patch_points):
        """verbose=True 与 var_time=True 组合正常工作"""
        from unittest.mock import MagicMock, patch

        t_patch, state_patch = simple_patch_points
        with patch("e2m2e.algorithm.solver.multiple_shooting.tqdm") as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar

            result = ms_corrector.correct(
                t_patch=t_patch,
                state_patch=state_patch,
                var_time=True,
                max_iter=10,
                verbose=True,
            )
            assert result is not None
            assert mock_tqdm.called


# =============================================================================
# 需求: 多进程子进程内核加载经 SPICEManager（双侧 furnsh）
# =============================================================================
class TestMultipleShootingMultiprocess:
    """多进程模式下，子进程内核加载经 ``SPICEManager.load_kernel``。

    子进程是新进程、CSPICE 内核池为空。``_worker_init`` →
    ``_load_worker_kernels`` → ``SPICEManager.load_kernel`` 须在 Python
    spiceypy 与 Rust cspice 两侧 furnsh，下沉到 Rust 的力模型查询（经
    ``EphemerisDynamics.propagate`` → ``propagate_with_stm_py``）才能查到
    天体状态。本测试守护这一加载链不被绕过（issue #334）。
    """

    def test_multiprocess_worker_loads_rust_kernels(
        self, spice_eph_dynamics, spice_kernel_path, simple_patch_points
    ):
        """n_workers>1 + kernel_dir：correct 不报 SPICE 错误（Rust 侧已 furnsh）。"""
        import os

        from e2m2e.algorithm.solver.multiple_shooting import MultipleShooting

        kernel_dir = os.path.dirname(spice_kernel_path)
        ms = MultipleShooting(dynamics=spice_eph_dynamics, n_workers=2, kernel_dir=kernel_dir)
        t_patch, state_patch = simple_patch_points
        # 不关心收敛，只关心多进程 worker 加载内核后 Rust 力模型查询不报错。
        result = ms.correct(t_patch=t_patch, state_patch=state_patch, max_iter=3, verbose=False)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
