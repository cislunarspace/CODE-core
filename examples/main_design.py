#!/usr/bin/env python3
"""main_design —— 轨道设计示例（对标 orbit-design-module 的 main_design.m）

用 ``e2m2e.algorithm.design.design_orbit`` 设计一条地月 L2 Halo 轨道，
在会合系（synodic）中绘制 x-z 与 x-y 投影。

用法：
    python examples/main_design.py            # 交互式出图
    python examples/main_design.py --save     # 存成 PNG（无头服务器可用）

前置条件：SPICE 内核位于仓库根 ``kernels/``（或设 ``$SPICE_KERNEL_DIR``）。
"""

from __future__ import annotations

import argparse
import pathlib
import time

# 输出图片保存到脚本所在目录（无论从哪运行）
_OUT_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="轨道设计示例（Halo）")
    parser.add_argument("--save", action="store_true", help="存为 PNG 而非交互式显示")
    args = parser.parse_args()

    print("=" * 60)
    print("e2m2e 轨道设计示例（Halo）")
    print("=" * 60)

    # 交互式出图需要 GUI 后端；无头环境（--save）用 Agg 存图
    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    from e2m2e.algorithm.design import design_orbit
    from e2m2e.tools.viz import OrbitVisualizer

    # 1. 端到端设计一条 L2 Halo（CR3BP 初猜 → 星历修正 → 高精度预报）
    print("\n1. 设计 L2 Halo 轨道（amplitude=30000 km，维持 0.2 年）")
    t0 = time.perf_counter()
    result = design_orbit(
        "Halo",
        collinear_point=2,
        amplitude=30000.0,
        phase=0.0,
        duration=0.2,
        output_step=3600.0,
    )
    elapsed = time.perf_counter() - t0
    print(f"   耗时 {elapsed:.1f} s")
    print(f"   Jacobi 常数 = {result.cr3bp_jacobi:.6f}")
    conv = result.correction
    print(f"   星历修正收敛 = {conv.converged}（{conv.iterations} 次迭代）")
    print(f"   星历行数 = {len(result.ephemeris)}")

    # 2. 取 CR3BP 周期轨道（无量纲会合系坐标，可直接画）
    cr3bp = result.cr3bp_orbit
    print(f"\n2. CR3BP 周期轨道周期 = {cr3bp.period:.6f}（无量纲）")

    # 3. 绘图：会合系 x-z / x-y 投影
    print("\n3. 绘制会合系投影")
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system

    system = earth_moon_system()
    viz = OrbitVisualizer(system)

    states = cr3bp.states  # (n,6) 无量纲会合系状态

    ax1 = viz.plot_2d_projection(states, plane="xz", label="Halo x-z")
    viz.plot_primary_bodies(ax=ax1)
    viz.plot_libration_points(ax=ax1)
    ax1.set_xlabel("X (nondimensional)")
    ax1.set_ylabel("Z (nondimensional)")
    ax1.set_title(f"Halo L2  Amplitude=30000 km  C={result.cr3bp_jacobi:.4f}")

    ax2 = viz.plot_2d_projection(states, plane="xy", label="Halo x-y")
    viz.plot_primary_bodies(ax=ax2)
    viz.plot_libration_points(ax=ax2)
    ax2.set_xlabel("X (nondimensional)")
    ax2.set_ylabel("Y (nondimensional)")
    ax2.set_title("Halo L2 x-y 投影")

    if args.save:
        viz.save(str(_OUT_DIR / "main_design_halo.png"), dpi=150)
        print(f"   已保存 {_OUT_DIR / 'main_design_halo.png'}")
    else:
        viz.show()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
