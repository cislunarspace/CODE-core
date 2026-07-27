#!/usr/bin/env python3
"""
固定周期轨道设计示例

展示如何使用 setup_2D_symmetric_x_fixed_t 方法设计具有指定周期的平面周期轨道。
该方法通过固定轨道半周期，调整初始条件 x0 和 y_dot0 使轨道满足周期性约束。

适用于：
- DRO (Distant Retrograde Orbit) 远距离逆行轨道
- 共振轨道 (Resonant Orbits)
- 需要精确周期控制的平面对称周期轨道
"""

import numpy as np

from e2m2e.algorithms import DifferentialCorrection
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


def design_fixed_period_dro(target_period):
    """设计具有目标周期的 DRO 轨道

    参数:
        target_period: 目标轨道周期（无量纲时间单位）

    返回:
        orbit: 修正后的轨道对象，如果失败则返回 None
    """
    # 1. 创建系统
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)

    # 2. 创建动力学对象
    dynamics = CR3BP_Dynamics(system=system)

    # 3. 配置固定周期微分修正器
    t_half = target_period / 2  # 半周期
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_t(t_half=t_half)

    print(f"目标周期: {target_period:.4f} (半周期: {t_half:.4f})")

    # 4. 初始猜测 (从已有轨道族或近似估计)
    # 对于 DRO，典型的初始猜测在 x ≈ 0.5-0.9 范围
    x0_guess = 0.6  # x 坐标初值
    y_dot0_guess = 0.5  # y 方向速度初值

    initial_state = np.array([x0_guess, 0.0, 0.0, 0.0, y_dot0_guess, 0.0])
    print(f"初始猜测: x0={x0_guess}, y_dot0={y_dot0_guess}")

    # 5. 执行迭代修正
    seed = Orbit(states=initial_state.reshape(1, -1), times=np.array([0.0]), system=system)
    seed.period = target_period
    orbit = corrector.iterate_correction(
        initial_guess=seed,
        verbose=True,
    )

    if orbit is not None:
        print("\n✓ 成功找到周期轨道!")
        print(f"  实际周期: {orbit.period:.6f}")
        print(f"  误差: {abs(orbit.period - target_period):.6e}")
        return orbit
    else:
        print(f"\n✗ 修正失败: {corrector.termination_reason}")
        return None


def example_workflow():
    """完整工作流示例：从轨道族选择初值到精确周期轨道"""
    print("=" * 60)
    print("固定周期微分校正法工作流示例")
    print("=" * 60)

    # 1. 系统设置
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    dynamics = CR3BP_Dynamics(system=system)

    # 2. 假设已从轨道族数据中选择了周期接近目标的轨道
    # 这里演示如何加载已有的轨道数据作为初值
    print("\n步骤 1: 从轨道族中选择初始猜测")
    print("  - 通常选择周期最接近目标周期的轨道")
    print("  - 提取其 x0 和 y_dot0 作为初值")

    # 3. 配置修正器
    print("\n步骤 2: 配置固定周期微分修正器")
    target_T = 3.0  # 目标周期
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_t(t_half=target_T / 2)

    print(f"  - 目标半周期: {target_T / 2:.4f}")
    print(f"  - 自由变量: {corrector.free_variables}")
    print(f"  - 约束条件: {list(corrector.target_conditions.keys())}")

    # 4. 初始猜测
    print("\n步骤 3: 提供初始猜测")
    initial_state = np.array([0.6, 0.0, 0.0, 0.0, 0.5, 0.0])
    orbit_init = Orbit(states=initial_state.reshape(1, -1), times=np.array([0.0]), system=system)
    orbit_init.period = target_T
    print(f"  - 初始状态: {initial_state}")

    # 5. 执行修正
    print("\n步骤 4: 执行迭代修正")
    orbit_result = corrector.iterate_correction(initial_guess=orbit_init, verbose=False)

    if orbit_result is not None:
        print("\n步骤 5: 验证结果")
        print(f"  - 收敛状态: {corrector.converged}")
        print(f"  - 最终周期: {orbit_result.period:.6f}")
        print(f"  - 最终误差: {corrector.current_error:.2e}")
        print(f"  - 迭代次数: {corrector.iteration_count}")
    else:
        print(f"\n修正失败: {corrector.termination_reason}")

    return corrector, orbit_result


if __name__ == "__main__":
    print("示例 1: 直接设计固定周期 DRO")
    print("-" * 40)
    orbit = design_fixed_period_dro(target_period=2.8)

    print("\n" + "=" * 60)
    print("示例 2: 完整工作流")
    print("-" * 40)
    corrector, result_orbit = example_workflow()
