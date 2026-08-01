#!/usr/bin/env python3
"""main_design —— 轨道设计示例

用 ``e2m2e.algorithm.design.design_orbit`` 设计一条地月 L2 Halo 轨道，
在会合系（synodic）中绘制 3D 轨道。

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
    # 摄动开关：太阳第三体引力 + 地月非球形（10 阶）+ 炮弹模型光压
    perturbation = {
        "sun_body": 1,
        "planets": 0,
        "earth_nonspherical": 1,
        "moon_nonspherical": 1,
        "solar_radiation": 1,
        "atmosphere": 0,
        "relativity": 0,
        "tide": 0,
        "coupling": 0,
    }
    print("   摄动开关：")
    for k, v in perturbation.items():
        print(f"     {k} = {v}")
    t0 = time.perf_counter()
    result = design_orbit(
        "Halo",
        collinear_point=2,
        amplitude=30000.0,
        phase=0.0,
        duration=2.0,
        output_step=3600.0,
        perturbation=perturbation,
        # 论文式分段打靶拼接（朱彦伟 2026）：逐段独立打靶转星历 + 远月点
        # 分层合并，对 Halo 等不稳定轨道长期保形。默认 two_level 适合
        # 单圈修正 + 短期预报；segmented 显式用于长期设计。
        correction_method="segmented",
    )
    elapsed = time.perf_counter() - t0
    print(f"   耗时 {elapsed:.1f} s")
    print(f"   Jacobi 常数 = {result.cr3bp_jacobi:.6f}")
    conv = result.correction
    print(f"   星历修正收敛 = {conv.converged}（{conv.iterations} 次迭代）")
    print(f"   星历行数 = {len(result.ephemeris)}")

    # 2. 取 CR3BP 周期轨道周期（用作参考量）
    cr3bp = result.cr3bp_orbit
    print(f"\n2. CR3BP 周期轨道周期 = {cr3bp.period:.6f}（无量纲）")

    # 3. 绘图：会合系 3D 轨道（加摄动后的 2 年拟周期预报轨迹）
    print("\n3. 绘制会合系 3D 轨道（加摄动后拟周期轨迹，观察点对准 L2）")
    from e2m2e.algorithm.dynamics import LibrationPoint
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system

    system = earth_moon_system()
    viz = OrbitVisualizer(system)

    # 画加摄动后的高精度预报星历（result.ephemeris），而非 CR3BP 理想周期解：
    # 摄动使轨道偏离闭合周期解，呈现拟周期。synodic_position 是地心归一（月球
    # 在 x=1）；减 mu 平移到质心归一（月球在 1-mu），与 L2 点、地月标记同坐标系。
    states = result.ephemeris.synodic_position.copy()
    states[:, 0] -= system.mu
    l2 = system.L_points[LibrationPoint.L2]  # L2 会合系坐标（质心归一）

    # 一次 3D 绘图：轨道 + 地月天体 + L2 平动点标注
    ax3d = viz.plot_3d_orbit(states, label="L2 Halo 拟周期轨迹（2 年）")
    viz.plot_primary_bodies(ax=ax3d, is_3d=True)
    ax3d.scatter(l2[0], l2[1], l2[2], marker="x", color="red", s=80, zorder=6)
    ax3d.text(l2[0], l2[1], l2[2] + 0.03, "L2", color="red", fontsize=12, fontweight="bold")

    # 视口：X 轴对准 L2 区（留出左侧月球），Y/Z 按 Halo 振幅留余量
    ax3d.set_xlim(0.96, 1.42)
    ax3d.set_ylim(-0.25, 0.25)
    ax3d.set_zlim(-0.25, 0.25)
    ax3d.set_xlabel("X（无量纲）")
    ax3d.set_ylabel("Y（无量纲）")
    ax3d.set_zlabel("Z（无量纲）")
    ax3d.set_title("L2 Halo 拟周期轨迹会合系 3D（振幅 30000 km，2 年）")
    ax3d.legend(loc="upper right")

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
