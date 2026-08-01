#!/usr/bin/env python3
"""
稳定性分析示例

展示如何使用 StabilityAnalysis 分析周期轨道的 Floquet 稳定性。
"""

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.algorithm.stability import StabilityAnalysis
from e2m2e.data.types.orbit import Orbit


def analyze_single_orbit():
    """分析单条轨道的稳定性"""
    print("=" * 60)
    print("单条轨道稳定性分析")
    print("=" * 60)

    # 1. 创建系统和种子轨道
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
    orbit = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if orbit is None:
        print("轨道修正失败")
        return

    # 2. 稳定性分析
    print(f"\n分析轨道: T = {orbit.period:.4f}")
    analyzer = StabilityAnalysis(orbit, dynamics)
    result = analyzer.analyze()

    # 3. 输出结果
    print("\n特征值:")
    for i, ev in enumerate(result.eigenvalues):
        print(f"  λ{i + 1} = {ev:.4f}  |λ| = {abs(ev):.6f}")

    print("\n稳定性指数:")
    for key, val in result.stability_indices.items():
        if val is not None:
            print(f"  {key} = {val:.6f}")

    print(f"\n稳定性分类: {result.classification.get('stability_type', 'unknown')}")
    print(f"是否稳定: {result.classification.get('is_stable', False)}")


def analyze_family():
    """分析轨道族的稳定性变化"""
    print("\n" + "=" * 60)
    print("轨道族稳定性分析")
    print("=" * 60)

    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 生成轨道族
    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
    seed_dro = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if seed_dro is None:
        print("种子轨道修正失败")
        return

    continuation = Continuation(corrector=corrector)
    family = continuation.natural_continuation(
        seed_orbit=seed_dro,
        param_range=(0.14, 0.9),
        step_size=0.005,
        verbose=False,
    )

    print(f"\n分析 {len(family)} 条轨道的稳定性...")

    # 批量分析
    print(f"\n{'索引':>4s}  {'周期':>10s}  {'Jacobi':>10s}  {'ν_max':>10s}  {'分类':>10s}")
    print("-" * 55)

    for i, orbit in enumerate(family):
        if i % max(1, len(family) // 10) == 0 or i == len(family) - 1:
            analyzer = StabilityAnalysis(orbit, dynamics)
            result = analyzer.analyze()
            C = system.get_jacobi_constant(orbit.states[0])

            nu_max = max(
                (v for v in result.stability_indices.values() if v is not None),
                default=float("nan"),
            )
            stype = result.classification.get("stability_type", "unknown")
            print(f"{i:4d}  {orbit.period:10.4f}  {C:10.4f}  {nu_max:10.4f}  {stype:>10s}")


def main():
    """主函数"""
    try:
        analyze_single_orbit()
        analyze_family()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
