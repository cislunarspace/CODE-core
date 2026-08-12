#!/usr/bin/env python3
"""main_transfer —— 转移轨道设计示例

用二体 Lambert 求解器（``e2m2e.algorithm.transfer.solve_lambert``，Izzo 算法）
设计一条地月转移轨道，沿转移弧插值得到轨迹并绘制 3D 图。

用法：
    python examples/main_transfer.py            # 交互式出图
    python examples/main_transfer.py --save     # 存成 PNG（无头服务器可用）

无 SPICE 依赖，秒级出结果。
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

# 输出图片保存到脚本所在目录（无论从哪运行）
_OUT_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="转移轨道设计示例（二体 Lambert）")
    parser.add_argument("--save", action="store_true", help="存为 PNG 而非交互式显示")
    parser.add_argument(
        "--log-level", default="WARNING", help="日志级别（DEBUG/INFO/WARNING/ERROR）"
    )
    args = parser.parse_args()

    from e2m2e.tools.logging import configure_logging

    configure_logging(level=args.log_level)

    print("=" * 60)
    print("e2m2e 转移轨道设计示例（二体 Lambert）")
    print("=" * 60)

    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    from _plot_setup import setup_cjk_font

    setup_cjk_font()

    from e2m2e.algorithm.transfer import solve_lambert

    # 1. 构造二体 Lambert 问题：低轨 → 月球距离处，5 天转移
    print("\n1. 二体 Lambert：LEO → 月球距离，转移时间 5 天")
    mu_earth = 398600.4415  # 地球引力常数 (km^3/s^2)
    r0 = np.array([6578.0, 0.0, 0.0])  # LEO 半径 (km)
    rf = np.array([384400.0, 0.0, 0.0])  # 月球距离 (km)
    tof = 5.0 * 86400.0  # 5 天 (s)

    sol = solve_lambert(r0, rf, tof, mu_earth, direction="short", revs=0)
    print(f"   出发速度 v0 = {np.round(sol.v0, 3)} km/s")
    print(f"   到达速度 vf = {np.round(sol.vf, 3)} km/s")

    # 2. 沿转移弧插值出轨迹（v0 出发 → 自由飞行 → vf 到达）
    print("\n2. 插值转移轨迹")
    n = 200
    t = np.linspace(0.0, tof, n)
    # 二体自由飞行解析插值：线性近似即可显示几何（演示用）
    v_avg = (sol.v0 + sol.vf) / 2.0
    positions = r0[None, :] + v_avg[None, :] * t[:, None]
    trajectory = np.column_stack([positions, np.zeros((n, 3))])  # (n,6) 会合系风格

    # 3. 绘图：转移轨迹 3D
    print("\n3. 绘制转移轨迹 3D")
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system

    system = earth_moon_system()

    # 归一化到地月尺度以便在会合系中显示（1 单位 = 384400 km）
    traj_syn = trajectory.copy()
    traj_syn[:, :3] /= 384400.0

    # 绘制转移轨迹 3D（直接用 matplotlib）
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot(
        traj_syn[:, 0],
        traj_syn[:, 1],
        traj_syn[:, 2],
        label="转移弧段",
        linewidth=1.5,
        alpha=0.8,
    )
    # 天体标记（质心归一坐标：主天体在 -mu，次天体在 1-mu）
    mu = system.mu
    ax3d.plot(
        [-mu],
        [0],
        [0],
        marker="o",
        color="blue",
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=1,
        linestyle="None",
        label="Earth",
    )
    ax3d.plot(
        [1 - mu],
        [0],
        [0],
        marker="o",
        color="silver",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1,
        linestyle="None",
        label="Moon",
    )
    ax3d.set_xlabel("X（归一化）")
    ax3d.set_ylabel("Y（归一化）")
    ax3d.set_zlabel("Z（归一化）")
    ax3d.set_title(f"二体 Lambert 转移  TOF=5 天  Δv≈{np.linalg.norm(sol.v0):.3f} km/s")
    ax3d.legend()

    if args.save:
        fig.savefig(
            str(_OUT_DIR / "main_transfer_lambert.png"),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        print(f"   已保存 {_OUT_DIR / 'main_transfer_lambert.png'}")
    else:
        plt.show()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
