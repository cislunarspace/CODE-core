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

    from _plot_setup import setup_cjk_font

    setup_cjk_font()

    from e2m2e.algorithm.design import design_orbit
    from e2m2e.tools.viz import OrbitVisualizer

    # 1. 端到端设计一条 L2 Halo（CR3BP 初猜 → 星历修正 → 高精度预报）
    print("\n1. 设计 L2 Halo 轨道（amplitude=30000 km，维持 2 年）")
    t0 = time.perf_counter()
    result = design_orbit(
        "Halo",
        collinear_point=2,
        amplitude=30000.0,
        phase=0.0,
        duration=2.0,
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

    # 3. 绘图：会合系 x-z / x-y 投影，观察点对准 L2
    print("\n3. 绘制会合系投影（观察点对准 L2）")
    from e2m2e.algorithm.dynamics import LibrationPoint
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system

    system = earth_moon_system()
    viz = OrbitVisualizer(system)

    states = cr3bp.states  # (n,6) 无量纲会合系状态
    l2 = system.L_points[LibrationPoint.L2]  # L2 会合系坐标 (1.1557, 0, 0)
    moon_x = 1.0 - system.mu  # 月球位置 (0.988, 0, 0)

    # 视口：以 L2 为中心（观察点对准 L2），留出左侧月球
    xlim = (0.96, 1.42)

    def _mark_reference(ax, plane: str) -> None:
        """在图上标出月球与 L2（观察点参考）。"""
        if plane == "xz":
            y_moon, y_l2 = 0.0, l2[2]
        else:
            y_moon, y_l2 = 0.0, 0.0
        # 月球（L2 轨道绕月，画月球标记便于读图）
        ax.scatter(moon_x, y_moon, s=60, c="gray", marker="o", edgecolors="black", zorder=5)
        ax.annotate(
            "月球",
            (moon_x, y_moon),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
        )
        # L2 平动点：十字线 + 标注（观察点）
        ax.axvline(l2[0], color="red", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(y_l2, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.scatter(l2[0], y_l2, marker="x", color="red", s=50, zorder=6)
        ax.annotate(
            "L2",
            (l2[0], y_l2),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=12,
            color="red",
            fontweight="bold",
        )

    ax1 = viz.plot_2d_projection(states, plane="xz", label="L2 Halo 轨道")
    ax1.set_xlim(*xlim)
    ax1.set_ylim(-0.20, 0.20)
    _mark_reference(ax1, "xz")
    ax1.set_xlabel("X（无量纲）")
    ax1.set_ylabel("Z（无量纲）")
    ax1.set_title("L2 Halo 轨道会合系 x-z 投影（振幅 30000 km）")
    ax1.legend(loc="upper right")

    ax2 = viz.plot_2d_projection(states, plane="xy", label="L2 Halo 轨道")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(-0.22, 0.22)
    _mark_reference(ax2, "xy")
    ax2.set_xlabel("X（无量纲）")
    ax2.set_ylabel("Y（无量纲）")
    ax2.set_title("L2 Halo 轨道会合系 x-y 投影")
    ax2.legend(loc="upper right")

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
