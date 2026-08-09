#!/usr/bin/env python3
"""main_control —— 轨道保持示例

先用 ``design_orbit`` 设计一条 Halo 标称轨道，再用 ``control_orbit``
做轨道保持蒙特卡洛仿真，绘制受控轨迹与标称轨迹对比。

用法：
    python examples/main_control.py            # 交互式出图
    python examples/main_control.py --save     # 存成 PNG（无头服务器可用）

前置条件：SPICE 内核位于仓库根 ``kernels/``（或设 ``$SPICE_KERNEL_DIR``）。
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

# 输出图片保存到脚本所在目录（无论从哪运行）
_OUT_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="轨道保持示例（Halo 蒙特卡洛）")
    parser.add_argument("--save", action="store_true", help="存为 PNG 而非交互式显示")
    args = parser.parse_args()

    print("=" * 60)
    print("e2m2e 轨道保持示例（Halo）")
    print("=" * 60)

    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    from _plot_setup import setup_cjk_font

    setup_cjk_font()

    from e2m2e.algorithm.design import design_orbit
    from e2m2e.algorithm.station_keeping import control_orbit
    from e2m2e.api.models import DesignOrbitRequest
    from e2m2e.tools.viz import OrbitVisualizer

    # 1. 设计一条短弧 Halo 标称轨道（供轨道保持）
    print("\n1. 设计 L2 Halo 标称轨道")
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
    result = design_orbit(
        DesignOrbitRequest(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            duration=0.1 * 365.25 * 86400,
            output_step=3600.0,
            perturbation=perturbation,
        )
    )
    print(f"   星历行数 = {len(result.ephemeris)}")

    # 2. 轨道保持蒙特卡洛仿真（少量控制/样本，快速演示）
    print("\n2. 轨道保持蒙特卡洛仿真")
    print("   控制模式 1（目标点宽松），2 个控制周期，1 个蒙特卡洛样本")
    ctl = control_orbit(
        result.ephemeris,
        control_mode=1,
        num_controls=2,
        num_monte_carlo=1,
        control_interval=10.0,
        output_step=3600.0,
        perturbation=perturbation,
    )
    print(f"   失败样本数 = {ctl.num_failed}")
    rows = np.asarray(ctl.sk_statistic.rows)
    if rows.size:
        print(f"   总 Δv = {rows[0, 0]:.3f} m/s，最大 Δv = {rows[0, 1]:.3f} m/s")
    else:
        print("   统计为空（无有效样本）")

    # 3. 绘制标称 vs 受控轨迹（会合系 x-z）
    print("\n3. 绘制标称与受控轨迹对比")
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system

    system = earth_moon_system()
    viz = OrbitVisualizer(system)

    # 标称：会合系无量纲状态
    nominal = result.ephemeris.synodic_position
    ax1 = viz.plot_2d_projection(
        np.column_stack([nominal, np.zeros((len(nominal), 3))]),
        plane="xz",
        label="标称轨道",
    )

    # 受控：最后一次样本的受控星历（若可用）
    if ctl.controlled_ephemeris is not None:
        controlled = ctl.controlled_ephemeris.synodic_position
        viz.plot_2d_projection(
            np.column_stack([controlled, np.zeros((len(controlled), 3))]),
            plane="xz",
            color="orange",
            label="受控轨道",
            ax=ax1,
        )

    viz.plot_primary_bodies(ax=ax1)
    viz.plot_libration_points(ax=ax1)
    ax1.set_xlabel("X（无量纲）")
    ax1.set_ylabel("Z（无量纲）")
    ax1.set_title("L2 Halo 轨道保持：标称 vs 受控（会合系 x-z）")
    ax1.legend(loc="upper right")

    if args.save:
        viz.save(str(_OUT_DIR / "main_control_halo.png"), dpi=150)
        print(f"   已保存 {_OUT_DIR / 'main_control_halo.png'}")
    else:
        viz.show()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
