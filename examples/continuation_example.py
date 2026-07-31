#!/usr/bin/env python3
"""
延拓法示例

展示如何使用自然延拓和伪弧长延拓生成 DRO 轨道族。
"""

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit


def natural_continuation_demo():
    """自然延拓生成 DRO 轨道族"""
    print("=" * 60)
    print("自然延拓示例")
    print("=" * 60)

    # 1. 创建系统
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 2. DRO 种子轨道
    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
    seed_dro = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if seed_dro is None:
        print("种子轨道修正失败")
        return

    print(f"\n种子轨道: T = {seed_dro.period:.4f}")

    # 3. 自然延拓
    print("\n执行自然延拓...")
    continuation = Continuation(corrector=corrector)
    family = continuation.natural_continuation(
        seed_orbit=seed_dro,
        param_range=(0.14, 0.9),
        step_size=0.005,
        verbose=False,
    )

    print(f"生成 {len(family)} 条轨道")

    # 4. 打印摘要
    print("\n轨道族摘要:")
    print("-" * 50)
    print(f"{'索引':>4s}  {'x0':>10s}  {'周期':>10s}  {'Jacobi':>10s}")
    print("-" * 50)
    for i, orb in enumerate(family):
        if i % max(1, len(family) // 8) == 0 or i == len(family) - 1:
            C = system.get_jacobi_constant(orb.states[0])
            print(f"{i:4d}  {orb.states[0, 0]:10.6f}  {orb.period:10.4f}  {C:10.4f}")

    return family, system


def main():
    """主函数"""
    try:
        family, system = natural_continuation_demo()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
