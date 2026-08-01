#!/usr/bin/env python3
"""main_propagate —— 轨道预报示例（对标 orbit-design-module 的 main_propagate.m）

用高精度力模型（``ForceModel.from_config`` + ``propagate``）从一条 Halo
轨道的初始状态外推 60 天，绘制预报轨迹。

用法：
    python examples/main_propagate.py            # 交互式出图
    python examples/main_propagate.py --save     # 存成 PNG（无头服务器可用）

前置条件：SPICE 内核位于仓库根 ``kernels/``（或设 ``$SPICE_KERNEL_DIR``）。
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

# 输出图片保存到脚本所在目录（无论从哪运行）
_OUT_DIR = pathlib.Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="轨道预报示例（高精度力模型）")
    parser.add_argument("--save", action="store_true", help="存为 PNG 而非交互式显示")
    args = parser.parse_args()

    print("=" * 60)
    print("e2m2e 轨道预报示例（高精度力模型）")
    print("=" * 60)

    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    from _plot_setup import setup_cjk_font

    setup_cjk_font()

    from e2m2e.algorithm.design import design_orbit
    from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system
    from e2m2e.tools.viz import OrbitVisualizer

    # 1. 设计一条短弧 Halo 作为预报起点（duration=0.02 年 ≈ 7.3 天）
    print("\n1. 设计 L2 Halo 短弧（提供初始状态与力模型）")
    result = design_orbit(
        "Halo",
        collinear_point=2,
        amplitude=30000.0,
        phase=0.0,
        duration=0.02,
        output_step=3600.0,
    )
    print(f"   初始状态 = {np.round(result.initial_state, 3)}")

    # 2. 复用设计链路的力模型，从初始状态外推 60 天
    print("\n2. 高精度外推 60 天（日/月/行星引力 + 光压 + 地球/月球非球形）")
    from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
    from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
    from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
    from e2m2e.algorithm.design.design_orbit import default_kernel_dir, load_design_kernels
    from e2m2e.algorithm.dynamics import EphemerisSystem
    from e2m2e.algorithm.forces import ForceModel
    from e2m2e.data.kernels.manager import SPICEManager

    # design_orbit 自动加载了 SPICE 内核并构造系统；这里重建独立系统。
    # 直接构造星历系统 + 力模型（与 design 第 4 段同款）
    _spice = SPICEManager()
    load_design_kernels(_spice, default_kernel_dir())

    system = EphemerisSystem(bodies=["EARTH", "MOON", "SUN"], spice=_spice, origin="EARTH")
    system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=_spice),
    )

    fm = ForceModel.from_config(result.force_config, system)
    fm.max_step = 600.0

    et0 = _spice.utc_to_et(result.epoch_utc)
    duration_days = 60.0
    et_grid = et0 + np.arange(0.0, duration_days * 86400.0 + 3600.0, 3600.0)

    out = fm.propagate(
        result.initial_state,
        (et0, float(et_grid[-1])),
        t_eval=et_grid,
        max_steps=2_000_000,
    )
    states = np.asarray(out["states"], dtype=float)
    print(f"   预报 {len(states)} 步（60 天）")

    # 3. 绘制预报轨迹 3D（J2000，归一化到地月尺度）
    print("\n3. 绘制预报轨迹 3D")
    system_cr3bp = earth_moon_system()
    viz = OrbitVisualizer(system_cr3bp)

    traj = states.copy()
    traj[:, :3] /= 384400.0  # 归一化以在会合系尺度下显示

    ax3d = viz.plot_3d_orbit(traj, label="预报轨迹（60 天）")
    viz.plot_primary_bodies(ax=ax3d, is_3d=True)
    ax3d.set_xlabel("X（归一化）")
    ax3d.set_ylabel("Y（归一化）")
    ax3d.set_zlabel("Z（归一化）")
    ax3d.set_title(f"L2 Halo 高精度预报 60 天  步数={len(states)}")
    ax3d.legend()

    if args.save:
        viz.save(str(_OUT_DIR / "main_propagate_60d.png"), dpi=150)
        print(f"   已保存 {_OUT_DIR / 'main_propagate_60d.png'}")
    else:
        viz.show()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
