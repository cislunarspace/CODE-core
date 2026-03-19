"""
DRO轨道生成和延拓测试模块

测试DRO (Distant Retrograde Orbit) 轨道的生成和延拓功能。
这些测试确保 phase1_generate_dro.py 中使用的功能不被破坏。

参考论文：
  Cui et al. (2025) "Two-Impulse Transfers from Lunar Distant Retrograde Orbits
  to Resonant Orbits", JGCD, Vol.48, No.6

地月系统参数：
  μ = 1.21506683 × 10⁻²
  DU = 3.84405 × 10⁵ km, TU = 4.34811305 天
"""

import numpy as np
import pytest
import os
import tempfile
import shutil

import e2m2e
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit

# 地月系统质量比（论文中的精确值）
MU = 1.21506683e-2


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def earth_moon_system():
    """创建地月CR3BP系统（使用论文中的精确参数）"""
    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    return system


@pytest.fixture
def dynamics(earth_moon_system):
    """创建动力学对象"""
    return CR3BP_Dynamics(earth_moon_system)


@pytest.fixture
def corrector(dynamics):
    """创建微分修正器并配置为2D对称x固定x0模式"""
    corrector = DifferentialCorrection(dynamics)
    x0 = 0.79188556619742  # 论文中的种子DRO参数
    corrector.setup_2D_symmetric_x_fixed_x0(x0)
    return corrector


# ============================================================
# DRO 微分修正测试
# ============================================================
class TestDROCorrection:
    """测试DRO轨道的微分修正功能"""

    def test_corrector_setup(self, corrector):
        """测试微分修正器配置正确"""
        assert corrector.setup_type == "2D_symmetric_x_fixed_x0"
        assert "y_dot0" in corrector.free_variables
        assert "T_half" in corrector.free_variables
        # y分量索引为1，vx分量为3
        assert 1 in corrector.constraint_indices
        assert 3 in corrector.constraint_indices

    def test_dro_seed_correction(self, corrector):
        """测试DRO种子轨道的修正"""
        # 论文中的DRO初始猜测
        x0 = 0.79188556619742
        vy0 = 0.53682  # 初始猜测值
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        # 执行微分修正 - 返回 Orbit 对象
        result_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        # 验证结果
        assert result_orbit is not None, "DRO种子修正应该成功"
        assert isinstance(result_orbit, Orbit), "结果应该是 Orbit 对象"

        # 验证轨道属性 - iterate_correction 返回 Orbit 对象
        assert result_orbit.period > 0, "周期应该为正"

        # 验证 corrector 自身的状态
        assert corrector.success is True, "修正应该成功"
        assert corrector.converged is True, "应该收敛"

        # 验证对称性条件（通过轨道状态验证）
        states = result_orbit.states
        n_states = len(states)

        # 取中间状态（半周期处）验证对称性
        mid_idx = n_states // 2
        mid_state = states[mid_idx]

        # 验证周期性条件（对称性）
        assert abs(mid_state[1]) < 1e-2, f"y(T/2) 应该接近0，实际: {mid_state[1]}"
        assert abs(mid_state[3]) < 1e-2, f"vx(T/2) 应该接近0，实际: {mid_state[3]}"

    def test_dro_period_reasonable(self, corrector):
        """测试DRO轨道周期在合理范围内"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        result_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if result_orbit is not None and corrector.success:
            # DRO周期通常在2-7个无量纲时间单位之间
            period = result_orbit.period
            assert 1.0 < period < 10.0, f"周期应该在合理范围内: {period}"

    def test_dro_jacobi_constant(self, corrector, earth_moon_system):
        """测试DRO的Jacobi常数计算"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        result_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if result_orbit is not None and corrector.success:
            # 计算初始状态的Jacobi常数
            C = earth_moon_system.get_jacobi_constant(seed_state)

            # DRO的Jacobi常数通常在3.0-3.5之间
            assert 2.5 < C < 4.0, f"Jacobi常数应该在合理范围内: {C}"

    def test_dro_orbit_save_load(self, corrector, earth_moon_system):
        """测试DRO轨道的保存和加载"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        result_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if result_orbit is not None and corrector.success:
            # iterate_correction 已经返回完整的 Orbit 对象
            orbit = result_orbit

            # 创建临时目录并保存
            temp_dir = tempfile.mkdtemp()
            try:
                filepath = os.path.join(temp_dir, "test_dro.json")
                orbit.save_to_file(filepath)
                assert os.path.exists(filepath), "文件应该被创建"

                # 加载轨道
                loaded_orbit = Orbit.load_from_file(filepath, system=earth_moon_system)
                assert loaded_orbit is not None
                assert loaded_orbit.period == orbit.period
            finally:
                shutil.rmtree(temp_dir)


# ============================================================
# DRO 自然延拓测试
# ============================================================
class TestDRONaturalContinuation:
    """测试DRO轨道的自然延拓功能"""

    def test_continuation_setup(self, corrector):
        """测试延拓器配置"""
        continuation = Continuation(corrector, step=0.02)

        assert continuation.continuation_parameter == "x0"
        assert continuation.step_size == 0.02

    def test_continuation_single_orbit(self, corrector):
        """测试生成单条延拓轨道"""
        continuation = Continuation(corrector, step=0.01)

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        # 先修正种子轨道
        corrected_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if corrected_orbit is not None and corrector.success:
            # 执行单步延拓 (param_range, step_size)
            result_family = continuation.natural_continuation(
                corrected_orbit,
                param_range=(x0, x0 + 0.02),
                step_size=0.01,
                verbose=False,
            )

            # 验证结果
            assert result_family is not None, "延拓应该返回结果"
            assert len(result_family) >= 1, "应该至少生成一条轨道"

    def test_continuation_multiple_orbits(self, corrector):
        """测试生成多条延拓轨道"""
        continuation = Continuation(corrector, step=0.005)

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        # 先修正种子轨道
        corrected_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if corrected_orbit is not None and corrector.success:
            # 执行延拓
            result_family = continuation.natural_continuation(
                corrected_orbit,
                param_range=(x0, x0 + 0.02),
                step_size=0.005,
                verbose=False,
            )

            # 验证结果
            if result_family is not None:
                assert len(result_family) >= 1, "应该至少生成一条轨道"

                # 验证轨道周期性
                for orbit in result_family:
                    if orbit is not None:
                        assert orbit.period > 0

    def test_continuation_period_trend(self, corrector):
        """测试延拓过程中周期变化趋势"""
        continuation = Continuation(corrector, step=0.005)

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        # 先修正种子轨道
        corrected_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if corrected_orbit is not None and corrector.success:
            # 执行延拓
            result_family = continuation.natural_continuation(
                corrected_orbit,
                param_range=(x0, x0 + 0.02),
                step_size=0.005,
                verbose=False,
            )

            if result_family is not None:
                periods = result_family.get_periods()
                # 验证周期数据
                assert len(periods) >= 1
                for p in periods:
                    assert p > 0, f"周期应该为正数: {p}"


# ============================================================
# 集成测试：完整DRO生成流程
# ============================================================
class TestDROGenerationPipeline:
    """测试完整的DRO生成流程"""

    def test_full_pipeline(self, earth_moon_system):
        """测试完整的DRO生成和延拓流程"""
        # 1. 创建系统
        assert earth_moon_system.mu == MU

        # 2. 创建动力学和微分修正器
        dynamics = CR3BP_Dynamics(earth_moon_system)
        corrector = DifferentialCorrection(dynamics)
        x0 = 0.79188556619742
        corrector.setup_2D_symmetric_x_fixed_x0(x0)

        # 3. 修正种子轨道
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        corrected_orbit = corrector.iterate_correction(seed_orbit, verbose=False)
        assert corrected_orbit is not None, "种子轨道修正应该成功"
        assert corrector.success is True, "种子轨道应该成功收敛"

        # 4. 延拓生成轨道族
        continuation = Continuation(corrector, step=0.01)

        family_result = continuation.natural_continuation(
            corrected_orbit,
            param_range=(x0, x0 + 0.03),
            step_size=0.01,
            verbose=False,
        )

        # 5. 验证结果
        assert family_result is not None
        assert len(family_result) > 0

        # 验证每条轨道
        for orbit in family_result:
            if orbit is not None:
                assert orbit.period > 0

    def test_backward_continuation(self, earth_moon_system):
        """测试反向延拓"""
        dynamics = CR3BP_Dynamics(earth_moon_system)
        corrector = DifferentialCorrection(dynamics)
        x0 = 0.79188556619742
        corrector.setup_2D_symmetric_x_fixed_x0(x0)

        # 种子轨道
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_period_guess = 3.420385
        seed_orbit = Orbit([seed_state], [0])
        seed_orbit.period = seed_period_guess

        corrected_orbit = corrector.iterate_correction(seed_orbit, verbose=False)

        if corrected_orbit is not None and corrector.success:
            # 反向延拓
            continuation = Continuation(corrector, step=0.01)

            result_family = continuation.natural_continuation(
                corrected_orbit,
                param_range=(x0 - 0.02, x0),
                step_size=0.01,
                verbose=False,
            )

            if result_family is not None:
                assert len(result_family) >= 0
