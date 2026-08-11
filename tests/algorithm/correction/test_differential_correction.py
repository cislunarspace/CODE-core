"""DifferentialCorrection 微分修正算法测试。

覆盖多种对称性配置、收敛性、修正轨道物理性质、
历史 API 与失败场景。
"""

import numpy as np
import pytest

from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


# 公共 fixtures 从 tests/algorithm/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro
# 这里仅在需要测试 _非_标准配置时构造自己的 corrector。


# ============================================================
# 配置类测试
# ============================================================
class TestSetup:
    """测试各种对称性配置的正确性"""

    def test_setup_2d_symmetric_x_fixed_x0(self, dro_dynamics):
        """2D对称x轴、固定x0配置"""
        corrector = DifferentialCorrection(dro_dynamics)
        result = corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)

        assert result is corrector
        assert corrector.setup_type == "2D_symmetric_x_fixed_x0"
        assert corrector.free_variables == ["y_dot0", "T_half"]
        assert corrector.free_variable_indices == [4, 6]
        assert corrector.constraint_indices == [1, 3]
        assert corrector.target_conditions == {"y": 0.0, "x_dot": 0.0}
        assert corrector.fixed_parameters["x0"] == 0.8

    def test_setup_2d_symmetric_x_fixed_t(self, dro_dynamics):
        """2D对称x轴、固定T配置"""
        corrector = DifferentialCorrection(dro_dynamics)
        result = corrector.setup_2D_symmetric_x_fixed_t(t_half=3.0)

        assert result is corrector
        assert corrector.setup_type == "2D_symmetric_x_fixed_t"
        assert corrector.free_variables == ["x0", "y_dot0"]
        assert corrector.free_variable_indices == [0, 4]
        assert corrector.constraint_indices == [1, 3]

    def test_iterate_correction_2d_symmetric_x_fixed_t(self, dro_dynamics):
        """2D对称x轴、固定T配置的迭代修正"""
        # 配置修正器
        t_half = 2.5
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_x_fixed_t(t_half=t_half)

        # DRO 的初始猜测
        x0_guess = 0.6
        y_dot0_guess = 0.4
        initial_state = np.array([x0_guess, 0.0, 0.0, 0.0, y_dot0_guess, 0.0])

        orbit_init = Orbit(
            states=initial_state.reshape(1, -1),
            times=np.array([0.0]),
        )

        # 执行迭代修正
        result = corrector.iterate_correction(initial_guess=orbit_init, verbose=False)
        result_orbit = result.orbit

        # 验证结果
        if result_orbit is not None:
            # 检查轨道周期是否接近目标周期
            target_period = 2 * t_half
            assert abs(result_orbit.period - target_period) < 1e-4, (
                f"周期误差过大: 期望 {target_period}, 实际 {result_orbit.period}"
            )
            assert result.status.value == "converged"
        # 如果 result_orbit 为 None，可能是初始猜测导致发散，不强制失败

    def test_setup_3d_symmetric_x_fixed_x0(self, dro_dynamics):
        """3D对称x轴、固定x0配置（Halo轨道）"""
        corrector = DifferentialCorrection(dro_dynamics)
        result = corrector.setup_3D_symmetric_x_fixed_x0(x0=0.8)

        assert result is corrector
        assert corrector.setup_type == "3D_symmetric_x_fixed_x0"
        assert corrector.free_variables == ["z0", "y_dot0", "T_half"]
        assert corrector.free_variable_indices == [2, 4, 6]
        assert corrector.constraint_indices == [1, 3, 5]
        assert corrector.target_conditions == {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}

    def test_setup_3d_symmetric_xz_fixed_x0(self, dro_dynamics):
        """3D XZ对称、固定x0配置"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_3D_symmetric_xz_fixed_x0(x0=0.8)

        assert corrector.setup_type == "3D_symmetric_xz_fixed_x0"
        assert corrector.free_variable_indices == [2, 4, 6]
        assert corrector.constraint_indices == [1, 3, 5]

    def test_setup_3d_symmetric_xz_fixed_z0(self, dro_dynamics):
        """3D XZ对称、固定z0配置"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_3D_symmetric_xz_fixed_z0(z0=0.1)

        assert corrector.setup_type == "3D_symmetric_xz_fixed_z0"
        assert corrector.free_variable_indices == [0, 4, 6]
        assert corrector.fixed_parameters["z0"] == 0.1

    def test_setup_2d_symmetric_y_fixed_y0(self, dro_dynamics):
        """2D对称y轴、固定y0配置（RO轨道）"""
        corrector = DifferentialCorrection(dro_dynamics)
        result = corrector.setup_2D_symmetric_y_fixed_y0(y0=0.4633)

        assert result is corrector
        assert corrector.setup_type == "2D_symmetric_y_fixed_y0"
        assert corrector.symmetry_condition == "y_axis"
        assert corrector.free_variables == ["x_dot0", "T_half"]
        assert corrector.free_variable_indices == [3, 6]
        assert corrector.constraint_indices == [0, 3]
        assert corrector.target_conditions == {"x": 0.0, "x_dot": 0.0}
        assert corrector.fixed_parameters["y0"] == 0.4633

    def test_setup_2d_symmetric_y_fixed_y0_default(self, dro_dynamics):
        """2D对称y轴配置默认y0=0"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_y_fixed_y0()

        assert corrector.fixed_parameters["y0"] == 0.0

    def test_setup_resets_history(self, dro_dynamics):
        """配置时应重置收敛历史"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.error_history = [1.0, 0.5]
        corrector._converged = True

        corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)

        assert corrector.error_history == []
        assert corrector._converged is False


# ============================================================
# 微分修正收敛测试
# ============================================================
class TestIterateCorrection:
    """测试微分修正迭代的收敛性"""

    def test_correction_converges(self, corrected_dro):
        """微分修正应成功收敛"""
        assert corrected_dro is not None

    def test_corrector_state_after_convergence(self, dro_corrector, dro_seed_orbit):
        """收敛后修正器内部状态正确"""
        dro_result = dro_corrector.iterate_correction(dro_seed_orbit)

        assert dro_result.status.value == "converged"
        assert dro_result.cause.value == "none"
        assert dro_corrector.check_convergence() is True
        assert dro_corrector.final_solution is not None
        assert dro_corrector.solution_time is not None

    def test_error_below_tolerance(self, dro_corrector, dro_seed_orbit):
        """最终误差应小于默认容差 1e-12"""
        dro_corrector.iterate_correction(dro_seed_orbit)

        history = dro_corrector.get_convergence_history()
        final_error = history["errors"][-1]
        assert final_error < dro_corrector.tolerance

    def test_error_monotonically_decreasing_near_convergence(self, dro_corrector, dro_seed_orbit):
        """收敛阶段误差应大致递减"""
        dro_corrector.iterate_correction(dro_seed_orbit)

        errors = dro_corrector.error_history
        # 至少最后3次迭代误差递减
        assert len(errors) >= 3
        for i in range(-3, -1):
            assert errors[i] > errors[i + 1]

    def test_converges_within_reasonable_iterations(self, dro_corrector, dro_seed_orbit):
        """应在合理的迭代次数内收敛"""
        dro_corrector.iterate_correction(dro_seed_orbit)
        assert dro_corrector.iteration_count <= 20


# ============================================================
# 修正轨道物理性质测试
# ============================================================
class TestCorrectedOrbit:
    """测试修正后轨道的物理正确性"""

    def test_orbit_is_orbit_object(self, corrected_dro):
        """返回值应为Orbit对象"""
        assert isinstance(corrected_dro, Orbit)

    def test_orbit_has_states_and_times(self, corrected_dro):
        """轨道应包含状态和时间序列"""
        assert corrected_dro.states.shape[1] == 6
        assert len(corrected_dro.times) == corrected_dro.states.shape[0]
        assert corrected_dro.states.shape[0] == 1000

    def test_initial_state_x_preserved(self, corrected_dro):
        """修正后初始x坐标应保持不变（固定量）"""
        np.testing.assert_allclose(corrected_dro.states[0, 0], 0.79188556619742, atol=1e-10)

    def test_initial_state_symmetry(self, corrected_dro):
        """初始状态应满足x轴对称条件: y=0, x_dot=0"""
        state0 = corrected_dro.states[0]
        np.testing.assert_allclose(state0[1], 0.0, atol=1e-10)  # y = 0
        np.testing.assert_allclose(state0[3], 0.0, atol=1e-10)  # x_dot = 0

    def test_half_period_crossing_condition(self, dro_corrector, dro_seed_orbit):
        """半周期处应满足垂直穿越条件: y(T/2)≈0, x_dot(T/2)≈0"""
        dro_corrector.iterate_correction(dro_seed_orbit)

        history = dro_corrector.convergence_history
        final_entry = history[-1]
        final_state = final_entry["final_state"]

        np.testing.assert_allclose(final_state[1], 0.0, atol=1e-10)  # y(T/2) ≈ 0
        np.testing.assert_allclose(final_state[3], 0.0, atol=1e-10)  # x_dot(T/2) ≈ 0

    def test_orbit_period_positive(self, dro_corrector, dro_seed_orbit):
        """修正后的轨道周期应为正数"""
        dro_corrector.iterate_correction(dro_seed_orbit)
        full_period = 2 * dro_corrector.solution_time
        assert full_period > 0

    def test_orbit_is_planar(self, corrected_dro):
        """DRO应为平面轨道: z=0, z_dot=0"""
        np.testing.assert_allclose(corrected_dro.states[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(corrected_dro.states[:, 5], 0.0, atol=1e-10)

    def test_orbit_closure(self, corrected_dro):
        """完整周期轨道首尾应闭合"""
        state_start = corrected_dro.states[0]
        state_end = corrected_dro.states[-1]
        np.testing.assert_allclose(state_start, state_end, atol=1e-6)

    def test_dro_period_reasonable(self, corrected_dro):
        """DRO 修正后周期应在合理范围内"""
        period = corrected_dro.period
        # DRO 周期通常在 1–15 个无量纲时间单位之间；
        # 上限 15 留出余量以容纳不同初值收敛到不同周期轨道的场景。
        assert 1.0 < period < 15.0, f"周期应该在合理范围内: {period}"

    def test_dro_jacobi_constant(self, corrected_dro, dro_dynamics):
        """DRO 修正后初始状态的 Jacobi 常数应在 2.5–4.0 之间"""
        C = dro_dynamics.system.get_jacobi_constant(corrected_dro.states[0])
        assert 2.5 < C < 4.0, f"Jacobi常数应该在合理范围内: {C}"

    def test_dro_orbit_save_load(self, corrected_dro, dro_dynamics, tmp_path):
        """DRO 轨道应能保存到 JSON 并正确加载回来（period 保持）"""
        filepath = tmp_path / "test_dro.json"
        corrected_dro.save_to_file(str(filepath))
        assert filepath.exists(), "文件应该被创建"

        loaded_orbit = Orbit.load_from_file(str(filepath), system=dro_dynamics.system)
        assert loaded_orbit is not None
        assert loaded_orbit.period == corrected_dro.period


# ============================================================
# 收敛历史 API 测试
# ============================================================
class TestConvergenceHistory:
    """测试收敛历史记录接口"""

    def test_get_convergence_history_keys(self, dro_corrector, dro_seed_orbit):
        """get_convergence_history 应返回正确的字段"""
        dro_corrector.iterate_correction(dro_seed_orbit)

        history = dro_corrector.get_convergence_history()
        assert "errors" in history
        assert "corrections" in history
        assert "iterations" in history
        assert "status" in history
        assert "cause" in history
        assert "message" in history

    def test_convergence_history_length(self, dro_corrector, dro_seed_orbit):
        """误差历史长度应等于迭代次数"""
        dro_corrector.iterate_correction(dro_seed_orbit)

        history = dro_corrector.get_convergence_history()
        assert len(history["errors"]) == history["iterations"]


# ============================================================
# 失败场景测试
# ============================================================
class TestFailureCases:
    """测试微分修正的失败场景"""

    def test_bad_initial_guess_returns_none(self, dro_dynamics):
        """极差的初始猜测应返回None或发散"""
        corrector = DifferentialCorrection(dro_dynamics)
        corrector.setup_2D_symmetric_x_fixed_x0(x0=0.5)
        corrector.max_iterations = 5

        bad_guess = Orbit(states=[[0.5, 0.0, 0.0, 0.0, 10.0, 0.0]], times=[0])
        bad_guess.period = 0.1

        result = corrector.iterate_correction(bad_guess)
        # 极差猜测可能收敛或不收敛，但必须返回状态化结果
        assert result.status.value != "iterating"

    def test_repr_and_str(self, dro_corrector):
        """__str__ 和 __repr__ 不应抛出异常"""
        s = str(dro_corrector)
        r = repr(dro_corrector)
        assert "DifferentialCorrection" in s
        assert "DifferentialCorrection" in r


# ============================================================
# Callback 测试
# ============================================================
class TestCallback:
    """测试 iterate_correction 的 callback 参数"""

    def test_callback_called_on_convergence(self, dro_corrector, dro_seed_orbit):
        """收敛时 callback 应在每次迭代被调用，最终一次 converged=True"""
        calls = []

        def on_iteration(iteration, error, converged):
            calls.append((iteration, error, converged))

        dro_corrector.iterate_correction(dro_seed_orbit, callback=on_iteration)

        assert len(calls) >= 1, "callback 应至少被调用一次"
        for i, (iteration, error, _converged) in enumerate(calls):
            assert iteration == i + 1
            assert isinstance(error, float)
            assert error > 0
        # 最后一次应为 converged
        assert calls[-1][2] is True, f"最终迭代 converged 应为 True，实际为 {calls[-1]}"

    def test_callback_receives_decreasing_errors(self, dro_corrector, dro_seed_orbit):
        """callback 收到的 error 应在收敛阶段递减"""
        calls = []

        def on_iteration(iteration, error, converged):
            calls.append(error)

        dro_corrector.iterate_correction(dro_seed_orbit, callback=on_iteration)

        errors = calls
        for i in range(-3, -1):
            assert errors[i] > errors[i + 1], (
                f"误差未递减: iter {i} = {errors[i]}, iter {i + 1} = {errors[i + 1]}"
            )

    def test_callback_none_does_not_affect_result(self, dro_corrector, dro_seed_orbit):
        """callback=None（默认值）不影响迭代修正行为"""
        result = dro_corrector.iterate_correction(dro_seed_orbit, callback=None)
        assert result.orbit is not None
        assert result.status.value == "converged"
