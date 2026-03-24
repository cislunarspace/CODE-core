"""测试DRO到RO转移NLP优化模块"""

import numpy as np
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.orbit import Orbit
from e2m2e.transfer.dro_transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    NLPOptimizationResult,
    TransferType,
    optimize_transfer,
)


def load_orbit_data(json_path: str) -> dict:
    """加载轨道数据"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_orbit_from_data(data: dict, system: CR3BP_System) -> Orbit:
    """从数据创建Orbit对象"""
    states = np.array(data['states'])
    times = np.array(data['times'])
    period = data.get('period', times[-1] - times[0])
    
    return Orbit(
        states=states,
        times=times,
        period=period,
        system=system
    )


def test_nlp_optimizer_initialization():
    """测试NLP优化器初始化"""
    print("\n=== 测试NLP优化器初始化 ===")
    
    # 创建系统
    system = CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system=system)
    
    # 创建虚拟轨道
    dummy_orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=system
    )
    dummy_orbit.period = 10.0
    
    # 出发点状态
    departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
    
    # 创建优化器
    optimizer = DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=dummy_orbit,
        arrival_orbit=dummy_orbit,
        departure_state=departure_state
    )
    
    assert optimizer is not None
    assert optimizer.mu == 0.012150585
    assert optimizer.departure_state is not None
    
    print("✓ NLP优化器初始化成功")


def test_optimization_variables():
    """测试优化变量类"""
    print("\n=== 测试优化变量类 ===")
    
    vars = NLPOptimizationVariables(alpha=1.2, transfer_time=15.0, t_ins=3.0)
    
    # 测试to_array
    arr = vars.to_array()
    assert arr.shape == (3,)
    assert np.allclose(arr, [1.2, 15.0, 3.0])
    
    # 测试from_array
    vars2 = NLPOptimizationVariables.from_array(arr)
    assert np.isclose(vars2.alpha, vars.alpha)
    assert np.isclose(vars2.transfer_time, vars.transfer_time)
    assert np.isclose(vars2.t_ins, vars.t_ins)
    
    print("✓ 优化变量类测试通过")


def test_departure_velocity_computation():
    """测试出发速度计算"""
    print("\n=== 测试出发速度计算 ===")
    
    system = CR3BP_System(mu=0.012150585, primary="earth", secondary="moon")
    dynamics = CR3BP_Dynamics(system=system)
    
    # 创建虚拟轨道
    dummy_orbit = Orbit(
        states=np.zeros((10, 6)),
        times=np.linspace(0, 10, 10),
        system=system
    )
    dummy_orbit.period = 10.0
    
    departure_state = np.array([0.8, 0.0, 0.0, 0.0, 0.8, 0.0])
    
    optimizer = DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=dummy_orbit,
        arrival_orbit=dummy_orbit,
        departure_state=departure_state
    )
    
    # 计算不同alpha的出发速度
    for alpha in [0.8, 1.0, 1.2, 1.5]:
        v_injection = optimizer.compute_departure_velocity(departure_state, alpha)
        
        # 验证速度方向与原始速度方向一致
        original_vel = departure_state[3:]
        original_dir = original_vel / np.linalg.norm(original_vel)
        new_dir = v_injection / np.linalg.norm(v_injection)
        
        # 方向应该相同(可能符号相反取决于轨道类型)
        dot = np.dot(original_dir, new_dir)
        assert abs(abs(dot) - 1.0) < 1e-6, f"速度方向不一致: dot={dot}"
    
    print("✓ 出发速度计算测试通过")


def test_nlp_result_structure():
    """测试NLP结果结构"""
    print("\n=== 测试NLP结果结构 ===")
    
    vars = NLPOptimizationVariables(alpha=1.2, transfer_time=15.0, t_ins=3.0)
    
    result = NLPOptimizationResult(
        variables=vars,
        objective_value=0.5,
        delta_v1=0.2,
        delta_v2=0.3,
        success=True,
        message="Test"
    )
    
    assert result.variables.alpha == 1.2
    assert result.objective_value == 0.5
    assert result.delta_v1 == 0.2
    assert result.delta_v2 == 0.3
    assert result.success == True
    
    print("✓ NLP结果结构测试通过")


def test_transfer_type_enum():
    """测试转移类型枚举"""
    print("\n=== 测试转移类型枚举 ===")
    
    assert TransferType.DIRECT.value == "direct"
    assert TransferType.LGA.value == "lga"
    assert TransferType.EXTERNAL.value == "external"
    
    print("✓ 转移类型枚举测试通过")


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    
    # 从主模块导入
    from e2m2e.transfer import (
        DROTRONLPOptimizer,
        NLPOptimizationVariables,
        NLPOptimizationResult,
        TransferType,
        optimize_transfer,
    )
    
    assert DROTRONLPOptimizer is not None
    assert NLPOptimizationVariables is not None
    assert NLPOptimizationResult is not None
    assert TransferType is not None
    assert optimize_transfer is not None
    
    print("✓ 模块导入测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("DRO-RO NLP模块测试")
    print("=" * 50)
    
    test_nlp_optimizer_initialization()
    test_optimization_variables()
    test_departure_velocity_computation()
    test_nlp_result_structure()
    test_transfer_type_enum()
    test_module_import()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
