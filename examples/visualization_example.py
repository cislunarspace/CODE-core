#!/usr/bin/env python3
"""
可视化模块使用示例

这个示例展示了如何使用 e2m2e 的可视化功能。
"""

import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization.plotting import OrbitVisualizer


def basic_visualization():
    """基本可视化示例"""
    print("=" * 60)
    print("e2m2e 可视化模块使用示例")
    print("=" * 60)

    # 1. 创建地月系统
    print("\n1. 创建地月系统")
    print("-" * 40)
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    print("系统创建完成")
    print(f"质量参数 μ = {system.mu:.6f}")
    print(f"L1点位置: {system.L1}")

    # 2. 创建可视化器
    print("\n2. 创建可视化器")
    print("-" * 40)
    viz = OrbitVisualizer(system)
    print("可视化器创建成功")

    # 3. 生成示例轨道数据
    print("\n3. 生成示例轨道数据")
    print("-" * 40)
    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)

    # 创建一个简单的圆形轨道（围绕L1点）
    amplitude = 0.02
    x0 = system.L1[0]  # L1点的x坐标

    x = x0 + amplitude * np.cos(t)
    y = amplitude * np.sin(t)
    z = np.zeros_like(t)
    vx = -amplitude * np.sin(t)
    vy = amplitude * np.cos(t)
    vz = np.zeros_like(t)

    orbit_states = np.column_stack([x, y, z, vx, vy, vz])
    print(f"轨道数据生成完成，包含 {len(orbit_states)} 个点")

    return system, viz, orbit_states


def demo_2d_projection(system, viz, orbit_states):
    """演示2D投影功能"""
    print("\n4. 2D投影演示")
    print("-" * 40)

    # XY平面投影
    print("绘制XY平面投影...")
    viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Lyapunov Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.axes.set_title('XY Projection - Lyapunov Orbit around L1')
    viz.show()

    # XZ平面投影
    print("绘制XZ平面投影...")
    viz.plot_2d_projection(orbit_states, plane='xz', color='green', label='XZ View')
    viz.axes.set_title('XZ Projection')
    viz.show()

    # YZ平面投影
    print("绘制YZ平面投影...")
    viz.plot_2d_projection(orbit_states, plane='yz', color='purple', label='YZ View')
    viz.axes.set_title('YZ Projection')
    viz.show()


def demo_3d_orbit(viz, orbit_states):
    """演示3D轨道功能"""
    print("\n5. 3D轨道演示")
    print("-" * 40)

    print("绘制3D轨道...")
    viz.plot_3d_orbit(orbit_states, color='red', label='3D Orbit')
    viz.plot_primary_bodies(ax=viz.axes_3d, is_3d=True)
    viz.plot_libration_points(ax=viz.axes_3d, is_3d=True)
    viz.axes_3d.legend()
    viz.axes_3d.set_title('3D View - Lyapunov Orbit')
    viz.show()


def demo_overview_plot(viz, orbit_states):
    """演示概览图功能"""
    print("\n6. 概览图演示")
    print("-" * 40)

    print("创建综合概览图...")
    fig = viz.create_overview_plot(orbit_states)
    fig.suptitle('Orbit Overview - All Projections', fontsize=16)
    viz.show()

    # 保存图形
    print("保存概览图为PNG文件...")
    viz.save('orbit_overview_demo.png', dpi=200)
    print("图形已保存为 'orbit_overview_demo.png'")


def demo_customization(viz, orbit_states):
    """演示自定义设置"""
    print("\n7. 自定义设置演示")
    print("-" * 40)

    # 修改可视化器设置
    viz.figsize = (10, 6)
    viz.orbit_linewidth = 2.0
    viz.orbit_alpha = 0.9
    viz.primary_body_color = 'orange'
    viz.secondary_body_color = 'gray'

    print("自定义设置应用完成:")
    print(f"  图形大小: {viz.figsize}")
    print(f"  轨道线宽: {viz.orbit_linewidth}")
    print(f"  轨道透明度: {viz.orbit_alpha}")
    print(f"  主天体颜色: {viz.primary_body_color}")
    print(f"  次天体颜色: {viz.secondary_body_color}")

    # 使用新设置绘制图形
    viz.plot_2d_projection(orbit_states, plane='xy', color='darkblue', label='Custom Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.axes.set_title('Customized Visualization')
    viz.show()


def main():
    """主函数"""
    try:
        # 基本设置
        system, viz, orbit_states = basic_visualization()

        # 演示各种功能
        demo_2d_projection(system, viz, orbit_states)
        demo_3d_orbit(viz, orbit_states)
        demo_overview_plot(viz, orbit_states)
        demo_customization(viz, orbit_states)

        print("\n" + "=" * 60)
        print("示例完成！")
        print("=" * 60)
        print("\n总结:")
        print("- 成功演示了2D投影（XY, XZ, YZ平面）")
        print("- 成功演示了3D轨道可视化")
        print("- 成功创建了综合概览图")
        print("- 成功演示了自定义设置")
        print("- 图形已保存为 'orbit_overview_demo.png'")

    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("请确保已安装所有依赖:")
        print("  pip install numpy matplotlib")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()