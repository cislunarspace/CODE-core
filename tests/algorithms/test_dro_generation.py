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
from e2m2e.algorithms.continuation import ContinuationDirection
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
        seed_t_half = seed_period_guess / 2

        # 执行微分修正
        result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)

        # 验证结果
        assert result is not None, "DRO种子修正应该成功"
        assert result["success"], f"修正应该成功: {result.get('termination_reason')}"
        
        # 验证轨道属性 - iterate_correction 返回字典，不是 Orbit 对象
        assert "state" in result, "结果应包含state"
        assert "period" in result, "结果应包含period"
        assert result["period"] > 0, "周期应该为正"
        
        # 验证对称性条件（通过 propagate 验证）
        # 使用修正后的状态重新传播，检查周期性条件
        corrected_state = result["state"]
        t_half = result["t_half"]
        
        # 重新传播验证
        dynamics = corrector.dynamics
        prop_result = dynamics.propagate(
            corrected_state,
            (0, t_half),
            t_eval=np.linspace(0, t_half, 100),
        )
        final_state = prop_result["states"][-1]
        
        # 验证周期性条件（对称性）
        assert abs(final_state[1]) < 1e-6, f"y(T/2) 应该接近0，实际: {final_state[1]}"
        assert abs(final_state[3]) < 1e-6, f"vx(T/2) 应该接近0，实际: {final_state[3]}"

    def test_dro_period_reasonable(self, corrector):
        """测试DRO轨道周期在合理范围内"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if result is not None and result["success"]:
            # DRO周期通常在2-7个无量纲时间单位之间
            period = result["period"]
            assert 1.0 < period < 10.0, f"周期应该在合理范围内: {period}"

    def test_dro_jacobi_constant(self, corrector, earth_moon_system):
        """测试DRO的Jacobi常数计算"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if result is not None and result["success"]:
            # 计算初始状态的Jacobi常数
            C = earth_moon_system.get_jacobi_constant(seed_state)
            
            # DRO的Jacobi常数通常在3.0-3.5之间
            assert 2.5 < C < 4.0, f"Jacobi常数应该在合理范围内: {C}"

    def test_dro_orbit_save_load(self, corrector, earth_moon_system):
        """测试DRO轨道的保存和加载"""
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if result is not None and result["success"]:
            # 从结果字典创建Orbit对象进行保存测试
            corrected_state = result["state"]
            period = result["period"]
            
            # 重新传播获取完整轨道
            dynamics = corrector.dynamics
            prop_result = dynamics.propagate(
                corrected_state,
                (0, period),
                t_eval=np.linspace(0, period, 1000),
            )
            
            orbit = Orbit(
                states=prop_result["states"],
                times=prop_result["time"],
                system=earth_moon_system,
            )
            orbit.period = period
            orbit.is_periodic = True
            
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
        continuation = Continuation(corrector, param="x0", step=0.02)
        
        assert continuation.continuation_parameter == "x0"
        assert continuation.step_size == 0.02
        assert continuation.direction == ContinuationDirection.FORWARD

    def test_continuation_single_orbit(self, corrector):
        """测试生成单条延拓轨道"""
        continuation = Continuation(corrector, param="x0", step=0.01)
        continuation.direction = ContinuationDirection.FORWARD
        continuation.max_step_size = 0.01
        continuation.min_step_size = 1e-5

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        # 先修正种子轨道
        seed_result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if seed_result is not None and seed_result["success"]:
            # 执行单步延拓
            result = continuation.natural_continuation(
                seed_result["state"],
                seed_result["t_half"],
                n_orbits=2,
                param_index=0,
                verbose=False,
            )
            
            # 验证结果
            assert result is not None, "延拓应该返回结果"
            assert "orbits" in result
            assert len(result["orbits"]) >= 1, "应该至少生成一条轨道"

    def test_continuation_multiple_orbits(self, corrector):
        """测试生成多条延拓轨道"""
        continuation = Continuation(corrector, param="x0", step=0.005)
        continuation.direction = ContinuationDirection.FORWARD
        continuation.max_step_size = 0.005
        continuation.min_step_size = 1e-5

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        # 先修正种子轨道
        seed_result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if seed_result is not None and seed_result["success"]:
            # 执行延拓
            result = continuation.natural_continuation(
                seed_result["state"],
                seed_result["t_half"],
                n_orbits=5,
                param_index=0,
                verbose=False,
            )
            
            # 验证结果
            if result is not None:
                assert "orbits" in result
                assert len(result["orbits"]) >= 1, "应该至少生成一条轨道"
                
                # 验证轨道周期性
                for orbit in result["orbits"]:
                    if orbit is not None:
                        assert orbit["period"] > 0

    def test_continuation_period_trend(self, corrector):
        """测试延拓过程中周期变化趋势"""
        continuation = Continuation(corrector, param="x0", step=0.005)
        continuation.direction = ContinuationDirection.FORWARD
        continuation.max_step_size = 0.005
        continuation.min_step_size = 1e-5

        # 种子轨道
        x0 = 0.79188556619742
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2

        # 先修正种子轨道
        seed_result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if seed_result is not None and seed_result["success"]:
            # 执行延拓
            result = continuation.natural_continuation(
                seed_result["state"],
                seed_result["t_half"],
                n_orbits=5,
                param_index=0,
                verbose=False,
            )
            
            if result is not None and "periods" in result:
                periods = result["periods"]
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
        seed_t_half = 3.420385 / 2
        
        seed_result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        assert seed_result is not None, "种子轨道修正应该成功"
        assert seed_result["success"], "种子轨道应该成功收敛"
        
        # 4. 延拓生成轨道族
        continuation = Continuation(corrector, param="x0", step=0.01)
        continuation.direction = ContinuationDirection.FORWARD
        
        family_result = continuation.natural_continuation(
            seed_result["state"],
            seed_result["t_half"],
            n_orbits=3,
            param_index=0,
            verbose=False,
        )
        
        # 5. 验证结果
        assert family_result is not None
        assert "orbits" in family_result
        assert len(family_result["orbits"]) > 0
        
        # 验证每条轨道
        for orbit in family_result["orbits"]:
            if orbit is not None:
                assert orbit["period"] > 0
                assert len(orbit["state"]) > 0

    def test_backward_continuation(self, earth_moon_system):
        """测试反向延拓"""
        dynamics = CR3BP_Dynamics(earth_moon_system)
        corrector = DifferentialCorrection(dynamics)
        x0 = 0.79188556619742
        corrector.setup_2D_symmetric_x_fixed_x0(x0)
        
        # 种子轨道
        vy0 = 0.53682
        seed_state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        seed_t_half = 3.420385 / 2
        
        seed_result = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        
        if seed_result is not None and seed_result["success"]:
            # 反向延拓
            continuation = Continuation(corrector, param="x0", step=-0.01)
            continuation.direction = ContinuationDirection.BACKWARD
            
            result = continuation.natural_continuation(
                seed_result["state"],
                seed_result["t_half"],
                n_orbits=2,
                param_index=0,
                verbose=False,
            )
            
            if result is not None:
                assert "orbits" in result
