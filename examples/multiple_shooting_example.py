#!/usr/bin/env python3
"""
多重打靶法示例

展示如何使用 MultipleShooting 和 sample_patch_points 对轨道进行修正。
"""

from e2m2e.algorithms import DifferentialCorrection, MultipleShooting, sample_patch_points
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


def main():
    """主函数"""
    print("=" * 60)
    print("多重打靶法示例")
    print("=" * 60)

    # 1. 创建系统和种子轨道
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
        print("种子轨道修正失败")
        return

    print(f"\n种子轨道: T = {seed_dro.period:.4f}")

    # 2. 采样 patch points
    n_points = 5
    t_patch, state_patch = sample_patch_points(seed_dro, n_points=n_points)
    print(f"\n采样 {n_points} 个 patch points")
    print(f"  时间范围: [{t_patch[0]:.4f}, {t_patch[-1]:.4f}]")

    # 3. 标准多重打靶
    print("\n执行标准多重打靶...")
    ms = MultipleShooting(dynamics=dynamics)
    result = ms.correct(
        t_patch=t_patch,
        state_patch=state_patch,
        max_iter=50,
        tolerance=1e-10,
        var_time=True,
    )

    if result.converged:
        print(f"  收敛！最大残差 = {result.max_residual:.2e}")
    else:
        print("  未收敛")

    # 4. 对比不同 patch points 数量
    print("\n不同 patch points 数量的收敛性:")
    print("-" * 45)
    print(f"{'n_points':>8s}  {'收敛':>6s}  {'最大残差':>12s}")
    print("-" * 45)

    for n in [3, 5, 8, 10]:
        t_p, s_p = sample_patch_points(seed_dro, n_points=n)
        ms_n = MultipleShooting(dynamics=dynamics)
        res = ms_n.correct(
            t_patch=t_p,
            state_patch=s_p,
            max_iter=50,
            tolerance=1e-10,
            var_time=True,
        )
        status = "是" if res.converged else "否"
        residual = f"{res.max_residual:.2e}" if res.converged else "N/A"
        print(f"{n:8d}  {status:>6s}  {residual:>12s}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
