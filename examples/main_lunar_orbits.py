#!/usr/bin/env python3
"""main_lunar_orbits：环月轨道设计

本示例用 ``e2m2e.algorithm.design.design_orbit`` 各设计 4 条代表轨道
（LLO / ELFO / DRO / Halo），打印几何特征，并画三张图理解它们：
    图 1  月心距离随时间演化（对数轴）：轨道尺度层次
    图 2  月心 X-Y 平面形态（近月区 / 远距区两个子图）
    图 3  ELFO 冻结性验证（近月点幅角、离心率、近月点高度的演化）

配套讲解见 docs/algorithms/lunar-orbits.md。

用法：
    python examples/main_lunar_orbits.py            # 交互式出图
    python examples/main_lunar_orbits.py --save     # 存成 PNG（无头服务器可用）
    python examples/main_lunar_orbits.py --skip DRO # 跳过某条轨道

前置条件：SPICE 内核位于仓库根 ``kernels/``（或设 ``$SPICE_KERNEL_DIR``）。
全量运行不到 1 分钟（release 构建；传播与打靶均在 Rust 内完成）。
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from e2m2e.data.kernels.manager import SPICEManager

# 输出图片保存到脚本所在目录（无论从哪运行）
_OUT_DIR = pathlib.Path(__file__).resolve().parent

# 月球常数（GRGM900C，与 e2m2e/algorithm/design/frozen_orbit.py 一致）
R_MOON = 1738.0  # 月球参考半径（km）
MU_MOON = 4902.8  # 月球引力参数（km³/s²）


def main() -> None:
    parser = argparse.ArgumentParser(description="环月轨道设计入门示例")
    parser.add_argument("--save", action="store_true", help="存为 PNG 而非交互式显示")
    parser.add_argument(
        "--log-level", default="WARNING", help="日志级别（DEBUG/INFO/WARNING/ERROR）"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["LLO", "ELFO", "DRO", "HALO"],
        help="跳过的轨道（LLO/ELFO/DRO/HALO）",
    )
    args = parser.parse_args()

    from e2m2e.tools.logging import configure_logging

    configure_logging(level=args.log_level)

    print("=" * 64)
    print("e2m2e 环月轨道设计入门示例")
    print("=" * 64)

    # 交互式出图需要 GUI 后端；无头环境（--save）用 Agg 存图
    if args.save:
        import matplotlib

        matplotlib.use("Agg")

    from _plot_setup import setup_cjk_font

    setup_cjk_font()

    from e2m2e.algorithm.design import design_orbit
    from e2m2e.algorithm.design.design_orbit import default_kernel_dir, load_design_kernels
    from e2m2e.api.models import DesignOrbitRequest
    from e2m2e.data.constants import Datum
    from e2m2e.data.kernels.manager import SPICEManager

    char_time_s = Datum.DE421.char_time_s  # 地月特征时间（约 4.34 天）

    # SPICE 内核只加载一次，四条轨道共用
    spice = SPICEManager()
    load_design_kernels(spice, default_kernel_dir())

    # 1. 先讲怎么理解环月轨道
    print("\n" + "-" * 64)
    print("怎么理解环月轨道：按离月距离分四层，每层主导动力学不同")
    print("-" * 64)
    print(
        " ① 近月轨道 LLO       离月面 100~2000 km，月球引力主导（近似开普勒二体），\n"
        "                       周期 2 小时量级，稳定。"
    )
    print(
        " ② 大椭圆冻结轨道 ELFO 近月点低（约 200 km）、远月点数千 km，高偏心；月球非\n"
        "                       球形摄动使拱线旋转，选倾角约 75 度、近月点幅角 270 度\n"
        "                       抵消漂移，轨道冻结，周期约 1 天。"
    )
    print(
        " ③ 远距轨道 DRO/DPO   离月 1~5 万 km，地月三体动力学显著但轨道仍绕月闭合；\n"
        "                       DRO 逆行、中性稳定，周期数天到十几天。"
    )
    print(
        " ④ 平动点轨道 Halo    绕地月 L1/L2 运动的周期轨道，三体动力学主导，不稳定；\n"
        "                       NRHO 是大振幅 Halo 的近直线形态，周期约半个月。"
    )
    print(
        "\n关键变量是轨道尺度：离月越远，地球引力占比越高，动力学从二体开普勒过渡到\n"
        "三体问题，轨道从稳定变为不稳定、需要连续保持。"
    )

    # 2. 设计四条代表轨道（每条对应一层）
    # 每条轨道的传播时长按周期定制：近月轨道周期短，两三天即可覆盖几十圈；
    # DRO 与 Halo 周期以天计，取一整圈多，保证形态完整。
    candidates = [
        {
            "key": "LLO",
            "title": "① 近月圆轨道 LLO（ELFO 管线 e 趋近 0）",
            "orbit_type": "ELFO",
            "semi_major_axis": 1838.0,  # 月面 100 km 高度
            "perilune_height": 100.0,
            "inclination": 75.0,
            "arg_of_pericenter": 270.0,
            "duration_day": 2.0,  # 约 24 圈
            "note": "近圆轨道 e 趋近 0，近月点幅角无定义，冻结诊断对它不适用，漂移数大是正常现象",
        },
        {
            "key": "ELFO",
            "title": "② 大椭圆冻结轨道 ELFO",
            "orbit_type": "ELFO",
            "semi_major_axis": 7000.0,  # 远月点约 1.2 万 km
            "perilune_height": 200.0,
            "inclination": 75.0,
            "arg_of_pericenter": 270.0,
            "duration_day": 4.0,  # 约 4 圈，覆盖冻结诊断窗口
        },
        {
            "key": "DRO",
            "title": "③ 远距离逆行轨道 DRO",
            "orbit_type": "DRO",
            "amplitude": 50000.0,  # 月心距离约 5 万 km
            "duration_day": 9.0,  # 周期 8.4 天，取一整圈多
        },
        {
            "key": "HALO",
            "title": "④ L2 晕轨道 Halo",
            "orbit_type": "HALO",
            "collinear_point": 2,
            "amplitude": 30000.0,
            "phase": 0.0,
            "duration_day": 15.0,  # 周期 14.6 天，取一整圈
        },
    ]

    output_step = 1800.0  # 近月轨道周期只有小时量级，半小时采样才画得出形态

    results: dict[str, tuple[dict, object]] = {}
    for cand in candidates:
        if cand["key"] in args.skip:
            continue
        print(f"\n{'=' * 64}\n设计 {cand['title']}\n{'=' * 64}")
        params = {
            k: v
            for k, v in cand.items()
            if k not in ("key", "title", "orbit_type", "duration_day", "note")
        }
        print(f"   参数: {params}")
        t0 = time.perf_counter()
        request = DesignOrbitRequest(
            orbit_type=cand["orbit_type"],
            duration=cand["duration_day"] * 86400.0,
            output_step=output_step,
            **params,
        )
        result = design_orbit(request, spice=spice)
        elapsed = time.perf_counter() - t0
        results[cand["key"]] = (cand, result)
        _print_summary(cand, result, spice, char_time_s, elapsed)

    if not results:
        print("\n所有轨道都被 --skip 跳过，无事可做。")
        return

    # 3. 绘图
    _plot_all(results, spice, save=args.save)

    print("\n" + "=" * 64)
    print("示例完成！")
    print("=" * 64)


def _moon_centric_position_km(result: object, spice: SPICEManager) -> np.ndarray:
    """星历（GCRS 地心）转月心位置（km）。

    从 GCRS 位置逐点减去月球地心位置（Rust 批量星历）。不经过会合系
    （synodic_position 用瞬时地月距离归一化，乘常数换回 km 会引入
    约 5% 的系统偏差）。
    """
    from e2m2e.integrators import batch_body_states_py

    et0 = spice.utc_to_et(result.epoch_utc)
    n = len(result.ephemeris)
    ets = et0 + np.arange(n) * result.output_step_sec
    moon = np.asarray(
        batch_body_states_py("MOON", "EARTH", [float(t) for t in ets]), dtype=float
    ).reshape(-1, 6)[:, :3]
    return np.asarray(result.ephemeris.position_km) - moon


def _print_summary(
    cand: dict,
    result: object,
    spice: SPICEManager,
    char_time_s: float,
    elapsed: float,
) -> None:
    """打印一条轨道的关键几何特征。"""
    r_moon = _moon_centric_position_km(result, spice)
    r_norm = np.linalg.norm(r_moon, axis=1)
    rp_km = float(r_norm.min()) - R_MOON  # 近月点高度
    ra_km = float(r_norm.max()) - R_MOON  # 远月点高度

    span_day = len(result.ephemeris) * result.output_step_sec / 86400.0
    print(f"   耗时 {elapsed:.1f} s，星历 {len(result.ephemeris)} 点（{span_day:.1f} 天）")
    print(f"   月心距离: 近月点 {rp_km:8.1f} km / 远月点 {ra_km:8.1f} km")
    if cand.get("note"):
        print(f"   说明: {cand['note']}")

    if cand["orbit_type"] == "ELFO":
        # 开普勒周期（月球二体近似）
        a = float(cand["semi_major_axis"])
        period_h = 2.0 * np.pi * np.sqrt(a**3 / MU_MOON) / 3600.0
        print(f"   开普勒周期 ≈ {period_h:.2f} h")
        # 冻结性诊断（design_orbit 的 ELFO 管线直接给出）
        print(
            f"   冻结诊断（{span_day:.1f} 天传播）: "
            f"Δe = {result.drift_e:+.4f}, "
            f"Δω = {result.drift_aop_deg:+.2f}°, "
            f"Δrp = {result.drift_rp_km:+.1f} km, "
            f"ω 年漂移率 = {result.secular_aop_rate_deg_per_year:+.1f}°/年"
        )
    else:
        # CR3BP 参考周期（无量纲）× 特征时间
        period_day = result.cr3bp_orbit.period * char_time_s / 86400.0
        print(f"   CR3BP 参考周期 ≈ {period_day:.2f} 天")
        print(
            f"   Jacobi 常数 = {result.cr3bp_jacobi:.6f}, "
            f"星历修正状态 = {result.correction.status.value}"
            f"（{result.correction.iterations} 次迭代）"
        )


def _plot_all(results: dict[str, tuple[dict, object]], spice: SPICEManager, *, save: bool) -> None:
    """三张图：尺度层次、月心形态、ELFO 冻结性。"""
    import matplotlib.pyplot as plt

    # 轨道数据（月心位置 km、月心距离 km、标签）
    data = {}
    for key, (cand, result) in results.items():
        r_moon = _moon_centric_position_km(result, spice)
        n_pts = len(result.ephemeris)
        days = np.arange(n_pts) * result.output_step_sec / 86400.0
        data[key] = {
            "cand": cand,
            "result": result,
            "r_moon": r_moon,
            "r_norm": np.linalg.norm(r_moon, axis=1),
            "days": days,
        }

    days_all = np.concatenate([d["days"] for d in data.values()])

    # ---- 图 1：月心距离随时间演化（对数轴），看尺度层次 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in data.values():
        rp = d["r_norm"].min() - R_MOON
        ra = d["r_norm"].max() - R_MOON
        ax.plot(
            d["days"],
            d["r_norm"],
            linewidth=1.2,
            label=f"{d['cand']['key']}（近月 {rp:.0f} / 远月 {ra:.0f} km）",
        )
    ax.set_yscale("log")
    ax.axhline(R_MOON, color="gray", linewidth=0.8, linestyle="--")
    ax.text(days_all[-1] * 0.98, R_MOON * 1.05, "月面", color="gray", ha="right")
    ax.set_xlabel("时间（天）")
    ax.set_ylabel("月心距离（km，对数轴）")
    ax.set_title("环月轨道尺度层次：从 1800 km 到 7 万 km")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _show_or_save(fig, "main_lunar_orbits_scale.png", save)

    # ---- 图 2：月心 X-Y 平面形态（近月区 / 远距区两个子图） ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    def _draw_moon(ax: object) -> None:
        from matplotlib.patches import Circle

        ax.add_patch(Circle((0, 0), R_MOON, color="silver", zorder=3))
        ax.set_aspect("equal")

    for key in ("LLO", "ELFO"):
        if key in data:
            r = data[key]["r_moon"]
            ax1.plot(r[:, 0], r[:, 1], linewidth=0.8, label=data[key]["cand"]["key"])
    _draw_moon(ax1)
    ax1.set_xlim(-15000, 15000)
    ax1.set_ylim(-15000, 15000)
    ax1.set_xlabel("X（km）")
    ax1.set_ylabel("Y（km）")
    ax1.set_title("近月区：LLO 与 ELFO")
    ax1.legend(loc="upper right", fontsize=9)

    for key in ("DRO", "HALO"):
        if key in data:
            r = data[key]["r_moon"]
            ax2.plot(r[:, 0], r[:, 1], linewidth=0.8, label=data[key]["cand"]["key"])
    _draw_moon(ax2)
    ax2.set_xlim(-85000, 85000)
    ax2.set_ylim(-85000, 85000)
    ax2.set_xlabel("X（km）")
    ax2.set_ylabel("Y（km）")
    ax2.set_title("远距区：DRO 与 Halo")
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("月心惯性系 X-Y 平面形态")
    fig.tight_layout()
    _show_or_save(fig, "main_lunar_orbits_xy.png", save)

    # ---- 图 3：ELFO 冻结性验证（ω、e、rp 演化） ----
    if "ELFO" in data:
        result = data["ELFO"]["result"]
        days = data["ELFO"]["days"]
        el = result.moon_centric_elements
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(days, el["aop"], linewidth=1.0)
        axes[0].set_ylabel("近月点幅角 ω（°）")
        axes[0].set_title("ELFO 冻结性验证：ω 稳定在 270 度附近（拱线不旋转）")
        axes[1].plot(days, el["e"], linewidth=1.0)
        axes[1].set_ylabel("离心率 e")
        axes[2].plot(days, el["rp"] - R_MOON, linewidth=1.0)
        axes[2].set_ylabel("近月点高度（km）")
        axes[2].set_xlabel("时间（天）")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _show_or_save(fig, "main_lunar_orbits_elfo_frozen.png", save)


def _show_or_save(fig: object, name: str, save: bool) -> None:
    """统一收尾：保存 PNG 或交互式显示。"""
    if save:
        fig.savefig(str(_OUT_DIR / name), dpi=150, bbox_inches="tight", pad_inches=0.1)
        print(f"   已保存 {_OUT_DIR / name}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
