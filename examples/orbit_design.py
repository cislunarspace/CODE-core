#!/usr/bin/env python3
"""
轨道设计示例

这个示例展示了如何使用 e2m2e 设计平动点轨道。
"""

import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection


def design_lyapunov_orbit():
    """设计 Lyapunov 轨道"""
    print("=" * 60)
    print("Lyapunov 轨道设计示例")
    print("=" * 60)

    # 1. 创建系统
    print("\n1. 初始化地月系统")
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    print("系统创建完成")
    print(f"L1 点位置: {system.L1}")

    # 2. 创建动力学对象
    print("\n2. 创建动力学对象")
    dynamics = CR3BP_Dynamics(system)

    # 3. 设置微分修正器
    print("\n3. 设置微分修正器")
    dc = DifferentialCorrection(dynamics)

    # 设置对称性条件（x 轴固定 x0 的 2D 对称轨道）
    x0 = system.L1[0] + 0.01  # L1 点附近
    dc.setup_2D_symmetric_x_fixed_x0(x0=x0)

    print(f"对称性条件: 2D 对称，固定 x0 = {x0:.4f}")

    # 4. 初始猜测
    print("\n4. 设置初始猜测")
    initial_state = np.array([x0, 0.0, 0.0, 0.0, 0.1, 0.0])
    t_half_guess = 1.5  # 半周期猜测

    print(f"初始状态: {initial_state}")
    print(f"半周期猜测: {t_half_guess}")

    # 5. 进行微分修正
    print("\n5. 进行微分修正")
    print("正在计算...")

    orbit, result = dc.correct_orbit(initial_state, t_half=t_half_guess)

    if orbit is not None:
        print("\n✓ 轨道设计成功!")
        print(f"  轨道周期: {orbit.period:.4f} 无量纲时间")
        print(f"  物理周期: {orbit.period * system.characteristic_time / 86400:.2f} 天")
        print(f"  收敛迭代次数: {result['iterations']}")
        print(f"  最终误差: {result['error']:.2e}")

        # 显示轨道参数
        print("\n轨道参数:")
        print(f"  初始状态: {orbit.initial_state}")
        print(f"  半周期状态: {orbit.half_state}")
        print(f"  Jacobi 常数: {orbit.jacobi_constant:.6f}")

        return orbit, system
    else:
        print("\n✗ 轨道设计失败")
        print(f"  错误信息: {result.get('message', '未知错误')}")
        return None, system


def visualize_orbit(orbit, system):
    """可视化轨道"""
    print("\n6. 轨道可视化")
    print("-" * 40)

    try:
        from e2m2e.visualization.plotting import OrbitVisualizer

        print("创建可视化对象...")
        viz = OrbitVisualizer(system)

        print("生成轨道图...")
        viz.create_overview_plot(orbit)

        print("显示图像（关闭窗口继续）...")
        viz.show()

    except ImportError:
        print("警告: 无法导入可视化模块，请确保 matplotlib 已安装")
        print("安装命令: pip install matplotlib")
    except Exception as e:
        print(f"可视化失败: {e}")


def main():
    """主函数"""
    try:
        # 设计轨道
        orbit, system = design_lyapunov_orbit()

        if orbit is not None:
            # 可视化轨道
            visualize_orbit(orbit, system)

            # 保存轨道数据
            print("\n7. 保存轨道数据")
            print("-" * 40)

            # 示例：计算轨道上的点
            t_eval = np.linspace(0, orbit.period, 100)
            states = orbit.propagate(t_eval)

            print(f"轨道已计算 {len(states)} 个状态点")
            print(f"第一个点: {states[0]}")
            print(f"最后一个点: {states[-1]}")

            # 检查周期性
            error = np.linalg.norm(states[0] - states[-1])
            print(f"周期性误差: {error:.2e}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
