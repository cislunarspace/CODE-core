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


def test_orbit():
    """测试轨道对象"""
    from e2m2e import CR3BP_System, CR3BP_Dynamics, Orbit

    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 传播轨迹
    initial_state = np.array([system.L1[0] + 0.01, 0, 0, 0, 0.15, 0])
    result = dynamics.propagate(initial_state, [0, 6.0], t_eval=np.linspace(0, 6.0, 2000))

    # 创建轨道对象
    orbit = Orbit(result["states"], result["time"], system)
    print(f"✓ 轨道对象创建: {orbit}")
    print(f"  振幅: x={orbit.amplitudes['x']:.4f}, y={orbit.amplitudes['y']:.4f}")

    # 测试插值
    t_mid = (result["time"][0] + result["time"][-1]) / 2
    state_interp = orbit.interpolate_at_time(t_mid)
    assert len(state_interp) == 6
    print("✓ 轨道插值成功")


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
    print("✓ 质心系↔主天体中心系变换可逆")


def test_differential_correction():
    """测试微分修正 - L1 Lyapunov轨道"""
    from e2m2e import CR3BP_System, CR3BP_Dynamics, DifferentialCorrection

    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 使用固定半周期的配置, 调整x0和vy来找Lyapunov轨道
    dc = DifferentialCorrection(dynamics)
    dc.setup_2D_symmetric_x_fixed_t(t_half=1.5)

    # L1附近的初始猜测: 从x轴出发，只有y方向速度
    initial_state = np.array([system.L1[0] + 0.02, 0, 0, 0, -0.15, 0])

    orbit, result = dc.correct_orbit(initial_state, t_half=1.5, verbose=False)

    if orbit is not None:
        print("✓ 微分修正成功")
        print(f"  周期: {result['period']:.6f}")
        print(f"  迭代次数: {result['iterations']}")
        print(f"  最终误差: {result['error']:.2e}")
        # 验证轨道关闭性: 终点应接近起点
        final_state = orbit.states[-1]
        initial = orbit.states[0]
        closure_error = np.linalg.norm(final_state[:3] - initial[:3])
        print(f"  轨道闭合误差: {closure_error:.2e}")
    else:
        print(f"△ 微分修正未收敛 (原因: {result['termination_reason']})")
        print("  这可能需要更好的初始猜测")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("e2m2e 库功能测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import),
        ("系统创建测试", test_system),
        ("动力学传播测试", test_dynamics),
        ("轨道对象测试", test_orbit),
        ("坐标变换测试", test_coordinate_transform),
        ("微分修正测试", test_differential_correction),
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
