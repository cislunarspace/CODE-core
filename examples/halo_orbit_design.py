#!/usr/bin/env python3
"""
Halo 轨道设计示例

展示如何使用 Richardson 三阶解析近似生成 Halo 轨道初始猜测，
通过微分修正收敛，并延拓生成 Halo 轨道族。
"""

import numpy as np

from e2m2e.algorithms import DifferentialCorrection
from e2m2e.algorithms.halo_initial_guess import compute_halo_initial_guess
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


def design_single_halo():
    """设计单条 Halo 轨道"""
    print("=" * 60)
    print("Halo 轨道设计")
    print("=" * 60)

    # 1. 创建系统
    system = CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)

    # 2. Richardson 三阶解析近似生成初始猜测
    z0 = 0.001  # z 方向振幅（小振幅下近似精度更高）
    guess = compute_halo_initial_guess(system.mu, z0, L=1, halo_class=0)

    print(f"\nRichardson 解析近似:")
    print(f"  x0    = {guess['x0']:.6f}")
    print(f"  vy0   = {guess['vy0']:.6f}")
    print(f"  T/2   = {guess['T_half']:.6f}")

    # 3. 构造初始状态
    initial_state = np.array([
        guess["x0"], 0.0, z0,
        0.0, guess["vy0"], 0.0,
    ])

    # 4. 微分修正
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=1)

    initial_guess = Orbit(
        states=initial_state.reshape(1, -1),
        times=np.array([0.0]),
        system=system,
    )
    initial_guess.period = guess["T_half"] * 2

    print("\n微分修正 Halo 轨道...")
    halo = corrector.iterate_correction(initial_guess=initial_guess, verbose=False)

    if halo is not None:
        C = system.get_jacobi_constant(halo.states[0])
        print(f"  Halo 周期: {halo.period:.6f}")
        print(f"  Jacobi 常数: {C:.6f}")
        print(f"  迭代次数: {corrector.iteration_count}")
        return halo, system, dynamics
    else:
        print(f"  修正失败: {corrector.termination_reason}")
        return None, system, dynamics


def generate_halo_family(system, dynamics):
    """生成不同 L 点的 Halo 轨道"""
    print("\n" + "=" * 60)
    print("不同 L 点的 Halo 轨道")
    print("=" * 60)

    halos = []
    z0 = 0.001

    for L in [1, 2]:
        guess = compute_halo_initial_guess(system.mu, z0, L=L, halo_class=0)
        initial_state = np.array([guess["x0"], 0.0, z0, 0.0, guess["vy0"], 0.0])

        corrector = DifferentialCorrection(dynamics)
        corrector.setup_halo_orbit_fixed_z0(z0=z0, libration_point=L)

        initial_guess = Orbit(
            states=initial_state.reshape(1, -1),
            times=np.array([0.0]),
            system=system,
        )
        initial_guess.period = guess["T_half"] * 2

        halo = corrector.iterate_correction(initial_guess=initial_guess, verbose=False)
        if halo is not None:
            C = system.get_jacobi_constant(halo.states[0])
            halos.append(halo)
            print(f"  L{L}: T={halo.period:.4f}, C={C:.4f}")
        else:
            print(f"  L{L}: 修正失败")

    print(f"\n成功生成 {len(halos)} 条 Halo 轨道")
    return halos


def main():
    """主函数"""
    try:
        halo, system, dynamics = design_single_halo()
        if halo is not None:
            generate_halo_family(system, dynamics)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
