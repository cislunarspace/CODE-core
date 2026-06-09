#!/usr/bin/env python3
"""
e2m2e 基础使用示例

这个示例展示了如何使用 e2m2e 库的基本功能。
"""

import numpy as np

from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.system import CR3BP_System


def main():
    """主函数"""
    print("=" * 60)
    print("e2m2e 基础使用示例")
    print("=" * 60)

    # 示例 1: 创建系统并查看信息
    print("\n1. 创建地月系统")
    print("-" * 40)

    # 方法 1: 从已知系统创建
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()

    # 设置特征尺度
    system.set_characteristic_scales(
        distance=384400,  # 地月距离 (km)
        period=27.32 * 86400,  # 月球轨道周期 (s)
    )

    # 计算平动点
    system.compute_libration_points()

    # 使用 info() 方法查看系统信息
    system.info()

    # 示例 2: 计算 Jacobi 常数
    print("\n2. 计算 Jacobi 常数")
    print("-" * 40)

    # 创建一个测试状态
    test_state = np.array([0.8, 0.1, 0.0, 0.0, 0.2, 0.0])

    # 计算 Jacobi 常数
    C = system.get_jacobi_constant(test_state)
    print(f"测试状态: {test_state}")
    print(f"Jacobi 常数 C = {C:.6f}")

    # 示例 3: 坐标转换
    print("\n3. 坐标转换示例")
    print("-" * 40)

    # 无量纲坐标
    dimensionless_state = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])

    # 转换为物理坐标
    physical_state = system.dimensionless_to_physical(dimensionless_state)
    print(f"无量纲状态: {dimensionless_state}")
    print(f"物理状态: {physical_state}")

    # 转换回无量纲坐标
    back_to_dimensionless = system.physical_to_dimensionless(physical_state)
    print(f"转换回无量纲: {back_to_dimensionless}")

    # 示例 4: 创建动力学对象
    print("\n4. 创建动力学对象")
    print("-" * 40)

    dynamics = CR3BP_Dynamics(system)
    print("动力学对象创建成功")
    print(f"系统质量参数 μ = {dynamics.system.mu}")

    # 示例 5: 计算平动点稳定性
    print("\n5. 平动点稳定性分析")
    print("-" * 40)

    from e2m2e.core.system import LibrationPoint

    # 分析 L1 点的稳定性
    stability_info = system.compute_stability_index(LibrationPoint.L1)

    print("L1 点稳定性分析:")
    print(f"  是否稳定: {stability_info['is_stable']}")
    print(f"  最大实部: {stability_info['max_real_part']:.6f}")
    print(f"  最大虚部: {stability_info['max_imag_part']:.6f}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
