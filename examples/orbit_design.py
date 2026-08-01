#!/usr/bin/env python3
"""
轨道设计示例

展示如何使用 e2m2e 设计 DRO（远距离逆行轨道），
包括微分修正、延拓生成轨道族、以及 Halo 轨道设计。
"""

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit


def design_dro():
    """设计 DRO 轨道并通过延拓生成轨道族"""
    print("=" * 60)
    print("DRO 轨道设计与轨道族生成")
    print("=" * 60)

    # 1. 创建地月系统
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 2. DRO 种子轨道
    initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
    seed_orbit = Orbit(states=[initial_state], times=[0], system=system)
    seed_orbit.period = 3.0

    # 3. 微分修正
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])

    print("\n1. 微分修正 DRO 种子轨道")
    seed_dro = corrector.iterate_correction(initial_guess=seed_orbit, verbose=False)

    if seed_dro is not None:
        print(f"   修正成功，周期 = {seed_dro.period:.6f}")
        C = system.get_jacobi_constant(seed_dro.states[0])
        print(f"   Jacobi 常数 = {C:.6f}")
    else:
        print(f"   修正失败: {corrector.termination_reason}")
        return None, system

    # 4. 延拓生成轨道族
    print("\n2. 自然延拓生成轨道族")
    continuation = Continuation(corrector=corrector)
    family = continuation.natural_continuation(
        seed_orbit=seed_dro,
        param_range=(0.14, 0.9),
        step_size=0.005,
        verbose=False,
    )
    print(f"   轨道族包含 {len(family)} 条轨道")

    # 5. 打印轨道族摘要
    print("\n3. 轨道族摘要")
    print("-" * 40)
    for i, orb in enumerate(family):
        if i % max(1, len(family) // 5) == 0 or i == len(family) - 1:
            C = system.get_jacobi_constant(orb.states[0])
            print(f"   [{i:3d}] T = {orb.period:.4f}, C = {C:.4f}")

    return family, system


def design_halo():
    """设计 Halo 轨道"""
    print("\n" + "=" * 60)
    print("Halo 轨道设计")
    print("=" * 60)

    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # Richardson 三阶解析近似生成初始猜测
    # 小振幅下 Richardson 近似精度更高，微分修正更容易收敛
    z0 = 0.001  # z 方向振幅
    guess = compute_halo_initial_guess(system.mu, z0, L=1, halo_class=0)

    initial_state = np.array(
        [
            guess["x0"],
            0.0,
            z0,
            0.0,
            guess["vy0"],
            0.0,
        ]
    )

    print("\n1. Richardson 解析近似初始猜测")
    print(f"   x0 = {guess['x0']:.6f}")
    print(f"   vy0 = {guess['vy0']:.6f}")
    print(f"   T/2 = {guess['T_half']:.6f}")

    # Halo 微分修正
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

    initial_guess = Orbit(
        states=initial_state.reshape(1, -1),
        times=np.array([0.0]),
        system=system,
    )
    initial_guess.period = guess["T_half"] * 2

    print("\n2. 微分修正 Halo 轨道")
    halo = corrector.iterate_correction(initial_guess=initial_guess, verbose=False)

    if halo is not None:
        print(f"   Halo 周期: {halo.period:.6f}")
        C = system.get_jacobi_constant(halo.states[0])
        print(f"   Jacobi 常数: {C:.6f}")
        return halo, system
    else:
        print(f"   修正失败: {corrector.termination_reason}")
        return None, system


def main():
    """主函数"""
    try:
        # DRO 设计
        family, system = design_dro()

        # Halo 设计
        halo, _ = design_halo()

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
