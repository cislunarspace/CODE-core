#!/usr/bin/env python3
"""
可视化模块使用示例

展示如何使用 e2m2e 的 FamilyPlotter 和 TransferPlotter 绘制轨道。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 非交互后端，CI 环境下使用


from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit
from e2m2e.tools.viz import FamilyPlotter, PlotConfig


def generate_dro_family():
    """生成 DRO 轨道族作为可视化数据"""
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # DRO 种子轨道
    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
    seed_dro = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if seed_dro is None:
        return None, None, system

    # 延拓生成轨道族
    continuation = Continuation(corrector=corrector)
    family = continuation.natural_continuation(
        seed_orbit=seed_dro,
        param_range=(0.14, 0.9),
        step_size=0.005,
        verbose=False,
    )

    # 计算每条轨道的 Jacobi 常数
    jacobi_values = [system.get_jacobi_constant(orb.states[0]) for orb in family]

    return family, jacobi_values, system


def main():
    """主函数"""
    print("=" * 60)
    print("e2m2e 可视化模块示例")
    print("=" * 60)

    # 1. 配置绘图风格
    print("\n1. 配置绘图风格")
    config = PlotConfig(title=32, label=28)
    config.apply_rcparams()

    # 2. 生成轨道族数据
    print("\n2. 生成 DRO 轨道族")
    family, jacobi_values, system = generate_dro_family()

    if family is None:
        print("   轨道族生成失败")
        return

    print(f"   轨道族包含 {len(family)} 条轨道")

    # 3. 使用 FamilyPlotter 绘制轨道族
    print("\n3. 绘制轨道族 2D 图")
    plotter = FamilyPlotter(system, config)
    plotter.plot_family_2d(family, jacobi_values, title="DRO Family")
    plotter.save("dro_family_2d.png", dpi=150)
    print("   已保存 dro_family_2d.png")

    # 4. 绘制轨道族概览图
    print("\n4. 绘制轨道族概览图")
    periods = [orb.period for orb in family]
    plotter.plot_jacobi_period(jacobi_values, periods, title="DRO Jacobi vs Period")
    plotter.save("dro_jacobi_period.png", dpi=150)
    print("   已保存 dro_jacobi_period.png")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
