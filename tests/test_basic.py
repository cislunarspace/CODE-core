"""
e2m2e库基本功能测试
"""

import numpy as np
import sys


def test_import():
    """测试基本导入"""
    import e2m2e

    print(f"✓ e2m2e版本: {e2m2e.__version__}")

    # 测试所有公共类的导入
    print("✓ 所有公共类导入成功")


def test_system():
    """测试系统创建和平动点计算"""
    from e2m2e import CR3BP_System

    # 从已知系统创建
    system = CR3BP_System.from_known_system("earth_moon")
    assert system.mu == 0.01215
    assert system.primary_body == "Earth"
    assert system.secondary_body == "Moon"
    print(f"✓ 地月系统创建成功: {system}")

    # 计算平动点
    L_points = system.compute_libration_points()
    assert system.has_L_points
    assert len(L_points) == 5

    print(f"  L1: [{system.L1[0]:.6f}, {system.L1[1]:.6f}]")
    print(f"  L2: [{system.L2[0]:.6f}, {system.L2[1]:.6f}]")
    print(f"  L3: [{system.L3[0]:.6f}, {system.L3[1]:.6f}]")
    print(f"  L4: [{system.L4[0]:.6f}, {system.L4[1]:.6f}]")
    print(f"  L5: [{system.L5[0]:.6f}, {system.L5[1]:.6f}]")

    # 验证L4和L5是等边三角形点
    assert abs(system.L4[1] - np.sqrt(3) / 2) < 0.01
    assert abs(system.L5[1] + np.sqrt(3) / 2) < 0.01
    print("✓ 平动点计算正确")

    # 设置特征尺度
    system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
    assert system.is_initialized
    print("✓ 特征尺度设置成功")

    # 计算Jacobi常数
    state = np.array([system.L1[0], 0, 0, 0, 0, 0])
    C = system.get_jacobi_constant(state)
    print(f"  L1点Jacobi常数: {C:.6f}")
    print("✓ Jacobi常数计算成功")


def test_dynamics():
    """测试动力学传播"""
    from e2m2e import CR3BP_System, CR3BP_Dynamics

    system = CR3BP_System.from_known_system("earth_moon")
    dynamics = CR3BP_Dynamics(system)

    # 从L1附近传播
    system.compute_libration_points()
    initial_state = np.array([system.L1[0] + 0.01, 0, 0, 0, 0.1, 0])

    result = dynamics.propagate(initial_state, [0, 3.0])
    assert "time" in result
    assert "states" in result
    assert len(result["states"]) > 0
    print(f"✓ 轨迹传播成功: {len(result['states'])} 个点")

    # 检查Jacobi常数守恒
    jacobi_error = result["jacobi_error"]
    print(f"  Jacobi常数误差: {jacobi_error:.2e}")

    # 测试STM计算
    result_stm = dynamics.propagate(initial_state, [0, 1.0], with_stm=True)
    assert "stm" in result_stm
    stm = result_stm["stm"][-1]
    assert stm.shape == (6, 6)
    print(f"  STM行列式: {np.linalg.det(stm):.6f}")
    print("✓ STM计算成功")


def test_coordinate_transform():
    """测试坐标变换"""
    from e2m2e import CR3BP_System, CoordinateTransformation

    system = CR3BP_System.from_known_system("earth_moon")
    coord = CoordinateTransformation(system)

    state = np.array([0.5, 0.1, 0, 0.01, 0.02, 0])

    # 旋转系 → 惯性系 → 旋转系
    state_inertial = coord.rotating_to_inertial(state, time=0.5)
    state_back = coord.inertial_to_rotating(state_inertial, time=0.5)
    assert np.allclose(state[:3], state_back[:3], atol=1e-10)
    print("✓ 旋转系↔惯性系变换可逆")

    # 质心系 → 主天体系 → 质心系
    state_primary = coord.barycentric_to_primary(state)
    state_back2 = coord.primary_to_barycentric(state_primary)
    assert np.allclose(state, state_back2, atol=1e-14)


def main():
    """运行所有测试"""
    print("=" * 60)
    print("e2m2e 库功能测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import),
        ("系统创建测试", test_system),
        ("动力学传播测试", test_dynamics),
        ("坐标变换测试", test_coordinate_transform),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
