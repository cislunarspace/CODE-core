"""
绘制 Halo 轨道族（仅读取已有 JSON，不生成轨道）

与 transfer-orbit-design 中 ``plot_dro_family.py`` 的用法一致：在脚本顶部配置
``FAMILY_JSON_PATH`` 指向 ``generate_halo_family.py`` 已保存的文件，或通过命令行传入路径。

数据流::

    generate_halo_family.py  ->  *.json  ->  本脚本  ->  *.png

用法::

    # 1）修改本文件中的 FAMILY_JSON_PATH 后直接运行
    python scripts/plot/plot_halo_family.py

    # 2）命令行指定 JSON（相对项目根或绝对路径）
    python scripts/plot/plot_halo_family.py output/halo/halo_L1_N_family_3857325361.json

    # 3）使用某目录下最新的 halo_*_family_*.json
    python scripts/plot/plot_halo_family.py --latest

    # 仅保存 PNG、不弹窗
    python scripts/plot/plot_halo_family.py path/to/family.json --no-show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from e2m2e.core import OrbitFamily, CR3BP_System
from e2m2e.visualization.plotting import OrbitVisualizer, compute_stability_for_family

from scripts.utils.common import MU

# =============================================================================
# 用户配置：已生成的轨道族 JSON（与 generate_halo_family.py 输出一致）
# 命令行未传入路径且未使用 --latest 时，使用此处路径。
# =============================================================================
FAMILY_JSON_PATH = project_root / "output" / "halo" / "halo_L1_N_family_3857325998.json"

DEFAULT_HALO_DIR = project_root / "output" / "halo"


def find_latest_family_json(directory: Path) -> Path | None:
    """在目录中按修改时间选取最新的 halo_*_family_*.json。"""
    candidates = sorted(
        directory.glob("halo_*_family_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 JSON 读取 Halo 轨道族并绘图（不生成轨道，仅 OrbitFamily.load_from_file）",
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="轨道族 JSON 路径；省略则使用脚本中的 FAMILY_JSON_PATH",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="与位置参数等价，便于兼容旧用法",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=f"使用 {DEFAULT_HALO_DIR} 下最新的 halo_*_family_*.json（显式选用，非默认行为）",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_HALO_DIR),
        help="与 --latest 配合：在此目录下搜索（默认 output/halo）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="PNG 输出目录，默认与 JSON 同目录",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1,
        help="起始轨道索引，-1 表示从第一条",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="结束轨道索引（含），-1 表示到最后一条",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不调用 plt.show()",
    )
    return parser.parse_args()


def plot_halo_family(
    family_path: Path,
    output_dir: Path,
    plot_start: int = -1,
    plot_end: int = -1,
    show: bool = True,
) -> None:
    system = CR3BP_System(mu=MU, primary="earth", secondary="moon")

    family_result = OrbitFamily.load_from_file(filename=str(family_path), system=system)
    n_orbits = len(family_result)
    family_name = family_path.stem

    print(f"已加载: {family_path}")
    print(f"轨道条数: {n_orbits}")

    if plot_start == -1 and plot_end == -1:
        ps, pe = 0, n_orbits - 1
    elif plot_start == -1:
        ps, pe = 0, min(plot_end, n_orbits - 1)
    elif plot_end == -1:
        ps, pe = min(plot_start, n_orbits - 1), n_orbits - 1
    else:
        ps = min(plot_start, n_orbits - 1)
        pe = min(plot_end, n_orbits - 1)

    print(f"绘制索引 [{ps}, {pe}]，共 {pe - ps + 1} 条")

    print("正在计算 Jacobi 常数...")
    jacobi_values = family_result.get_jacobi_constants().tolist()
    print(f"Jacobi 范围: {min(jacobi_values):.6f} ~ {max(jacobi_values):.6f}")

    print("正在计算稳定性指数（可能较慢）...")
    stability_values = compute_stability_for_family(family_result, family_result.system)
    print(f"λmax 范围: {min(stability_values):.6f} ~ {max(stability_values):.6f}")

    orbit_plotter = OrbitVisualizer(system=system)
    orbit_plotter.primary_body_color = "blue"
    orbit_plotter.secondary_body_color = "silver"
    orbit_plotter.libration_point_colors = ["gray"] * 5
    orbit_plotter.libration_point_markers = ["^"] * 5
    orbit_plotter.libration_point_sizes = [60] * 5

    cmap = matplotlib.colormaps["coolwarm"]
    jacobi_min = min(jacobi_values)
    jacobi_max = max(jacobi_values)
    jacobi_range = jacobi_max - jacobi_min if jacobi_max != jacobi_min else 1.0

    halo_center_x = 0.9
    halo_center_y = 0.0
    halo_center_z = 0.0
    halo_radius = 0.4

    seed_orbit = family_result[0]
    seed_jacobi = jacobi_values[0]
    seed_stability = stability_values[0]
    orbit_loop_start = 1 if ps == 0 else ps

    # ---------- 1. XZ ----------
    fig_global_2d, ax_global_2d = plt.subplots(figsize=(12, 10))
    label = f"Seed Halo (C={seed_jacobi:.4f}, λmax={seed_stability:.4f})"
    orbit_plotter.plot_2d_projection(seed_orbit, plane="xz", color="red", label=label, ax=ax_global_2d)
    for idx in range(orbit_loop_start, pe + 1):
        orbit = family_result[idx]
        norm_j = (jacobi_values[idx] - jacobi_min) / jacobi_range
        orbit_plotter.plot_2d_projection(
            orbit, plane="xz", color=cmap(norm_j), show_start=False, ax=ax_global_2d
        )
    orbit_plotter.plot_primary_bodies(ax=ax_global_2d)
    orbit_plotter.plot_libration_points(ax=ax_global_2d)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=jacobi_min, vmax=jacobi_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_global_2d, shrink=0.8).set_label("Jacobi Constant", fontsize=12)
    ax_global_2d.set_xlabel("X (nondimensional)", fontsize=12)
    ax_global_2d.set_ylabel("Z (nondimensional)", fontsize=12)
    ax_global_2d.set_title(
        f"Halo Orbit Family (XZ) — {n_orbits} orbits\n"
        f"C = [{jacobi_min:.4f}, {jacobi_max:.4f}], λmax = [{min(stability_values):.4f}, {max(stability_values):.4f}]",
        fontsize=12,
    )
    ax_global_2d.legend(loc="upper right", fontsize=10, markerscale=1.0, framealpha=0.9)
    ax_global_2d.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_2d_view.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_global_2d)

    # ---------- 2. 3D ----------
    fig_global_3d = plt.figure(figsize=(14, 10))
    ax_global_3d = fig_global_3d.add_subplot(111, projection="3d")
    orbit_plotter.plot_3d_orbit(
        seed_orbit,
        color="red",
        label=f"Seed Halo (C={seed_jacobi:.4f})",
        ax=ax_global_3d,
        show_start=True,
    )
    for idx in range(orbit_loop_start, pe + 1):
        orbit = family_result[idx]
        norm_j = (jacobi_values[idx] - jacobi_min) / jacobi_range
        orbit_plotter.plot_3d_orbit(orbit, color=cmap(norm_j), ax=ax_global_3d, show_start=False)
    orbit_plotter.plot_primary_bodies(ax=ax_global_3d, is_3d=True)
    orbit_plotter.plot_libration_points(ax=ax_global_3d, show_labels=True, is_3d=True)
    ax_global_3d.set_xlim(halo_center_x - halo_radius, halo_center_x + halo_radius)
    ax_global_3d.set_ylim(halo_center_y - halo_radius, halo_center_y + halo_radius)
    ax_global_3d.set_zlim(halo_center_z - halo_radius, halo_center_z + halo_radius)
    ax_global_3d.set_xlabel("X (nondimensional)", fontsize=12)
    ax_global_3d.set_ylabel("Y (nondimensional)", fontsize=12)
    ax_global_3d.set_zlabel("Z (nondimensional)", fontsize=12)
    ax_global_3d.set_title(
        f"Halo Orbit Family (3D) — {n_orbits} orbits\n"
        f"C = [{jacobi_min:.4f}, {jacobi_max:.4f}], λmax = [{min(stability_values):.4f}, {max(stability_values):.4f}]",
        fontsize=12,
    )
    sm_3d = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=jacobi_min, vmax=jacobi_max))
    sm_3d.set_array([])
    plt.colorbar(sm_3d, ax=ax_global_3d, shrink=0.6, pad=0.1).set_label("Jacobi Constant", fontsize=11)
    ax_global_3d.legend(loc="upper right", fontsize=10)
    ax_global_3d.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_3d_view.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_global_3d)

    # ---------- 3. Period & stability ----------
    sort_idx = np.argsort(jacobi_values)
    jacobi_sorted = np.array(jacobi_values)[sort_idx]
    periods_sorted = np.array(family_result.periods)[sort_idx]
    stability_sorted = np.array(stability_values)[sort_idx]

    fig_jacobi, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Jacobi Constant", fontsize=12)
    ax1.set_ylabel("Period (nondimensional)", color="tab:blue", fontsize=12)
    (line_period,) = ax1.plot(jacobi_sorted, periods_sorted, "o-", color="tab:blue", markersize=5, label="Period")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Stability Index (λmax)", color="tab:red", fontsize=12)
    (line_stability,) = ax2.plot(jacobi_sorted, stability_sorted, "s-", color="tab:red", markersize=5, label="λmax")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(f"Halo Orbit Family — Period and Stability (n = {n_orbits})", fontsize=13)
    ax1.legend([line_period, line_stability], ["Period", "Stability Index (λmax)"], loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_period_stability.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_jacobi)

    # ---------- 4. Overview ----------
    fig_overview = plt.figure(figsize=(18, 14))
    ax_ov1 = fig_overview.add_subplot(221)
    orbit_plotter.plot_2d_projection(
        seed_orbit, plane="xz", color="red", label=f"Seed (C={seed_jacobi:.4f})", ax=ax_ov1
    )
    for idx in range(orbit_loop_start, pe + 1):
        orbit = family_result[idx]
        norm_j = (jacobi_values[idx] - jacobi_min) / jacobi_range
        orbit_plotter.plot_2d_projection(orbit, plane="xz", color=cmap(norm_j), show_start=False, ax=ax_ov1)
    orbit_plotter.plot_primary_bodies(ax=ax_ov1)
    orbit_plotter.plot_libration_points(ax=ax_ov1)
    sm_ov = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=jacobi_min, vmax=jacobi_max))
    sm_ov.set_array([])
    plt.colorbar(sm_ov, ax=ax_ov1, shrink=0.8).set_label("Jacobi Constant", fontsize=10)
    ax_ov1.set_title(f"XZ ({n_orbits} orbits)", fontsize=11)
    ax_ov1.set_xlabel("X", fontsize=10)
    ax_ov1.set_ylabel("Z", fontsize=10)
    ax_ov1.legend(loc="upper right", fontsize=8)
    ax_ov1.set_aspect("equal")

    ax_ov2 = fig_overview.add_subplot(222)
    orbit_plotter.plot_2d_projection(seed_orbit, plane="xy", color="red", label="Seed", ax=ax_ov2)
    for idx in range(orbit_loop_start, pe + 1):
        orbit = family_result[idx]
        norm_j = (jacobi_values[idx] - jacobi_min) / jacobi_range
        orbit_plotter.plot_2d_projection(orbit, plane="xy", color=cmap(norm_j), show_start=False, ax=ax_ov2)
    orbit_plotter.plot_primary_bodies(ax=ax_ov2)
    orbit_plotter.plot_libration_points(ax=ax_ov2)
    ax_ov2.set_title("XY", fontsize=11)
    ax_ov2.set_xlabel("X", fontsize=10)
    ax_ov2.set_ylabel("Y", fontsize=10)
    ax_ov2.legend(loc="upper right", fontsize=8)
    ax_ov2.set_aspect("equal")

    ax_ov3 = fig_overview.add_subplot(223)
    ax_ov3.set_xlabel("Jacobi Constant", fontsize=10)
    ax_ov3.set_ylabel("Period", color="tab:blue", fontsize=10)
    (line_p,) = ax_ov3.plot(jacobi_sorted, periods_sorted, "o-", color="tab:blue", markersize=4)
    ax_ov3.tick_params(axis="y", labelcolor="tab:blue")
    ax_ov3_r = ax_ov3.twinx()
    ax_ov3_r.set_ylabel("λmax", color="tab:red", fontsize=10)
    (line_s,) = ax_ov3_r.plot(jacobi_sorted, stability_sorted, "s-", color="tab:red", markersize=4)
    ax_ov3_r.tick_params(axis="y", labelcolor="tab:red")
    ax_ov3.set_title("Jacobi vs Period & Stability", fontsize=11)
    ax_ov3.legend([line_p, line_s], ["Period", "λmax"], loc="upper right", fontsize=8)
    ax_ov3.grid(True, alpha=0.3)

    ax_ov4 = fig_overview.add_subplot(224, projection="3d")
    orbit_plotter.plot_3d_orbit(seed_orbit, color="red", label="Seed", ax=ax_ov4, show_start=True)
    for idx in range(orbit_loop_start, pe + 1):
        orbit = family_result[idx]
        norm_j = (jacobi_values[idx] - jacobi_min) / jacobi_range
        orbit_plotter.plot_3d_orbit(orbit, color=cmap(norm_j), ax=ax_ov4, show_start=False)
    orbit_plotter.plot_primary_bodies(ax=ax_ov4, is_3d=True)
    ax_ov4.set_xlim(halo_center_x - halo_radius, halo_center_x + halo_radius)
    ax_ov4.set_ylim(halo_center_y - halo_radius, halo_center_y + halo_radius)
    ax_ov4.set_zlim(halo_center_z - halo_radius, halo_center_z + halo_radius)
    ax_ov4.set_title("3D", fontsize=11)
    ax_ov4.set_xlabel("X", fontsize=10)
    ax_ov4.set_ylabel("Y", fontsize=10)
    ax_ov4.set_zlabel("Z", fontsize=10)
    ax_ov4.view_init(elev=20, azim=-60)

    fig_overview.suptitle(
        f"Halo Orbit Family Overview — Earth–Moon CR3BP (n = {n_orbits})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_overview.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_overview)

    print(f"\n图表已保存至: {output_dir}")
    print(f"  - {family_name}_2d_view.png")
    print(f"  - {family_name}_3d_view.png")
    print(f"  - {family_name}_period_stability.png")
    print(f"  - {family_name}_overview.png")


def _resolve_json_path(user_path: str) -> Path:
    p = Path(user_path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def main() -> None:
    args = parse_args()
    if args.no_show or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    if args.latest:
        search_dir = Path(args.dir)
        if not search_dir.is_absolute():
            search_dir = (project_root / search_dir).resolve()
        found = find_latest_family_json(search_dir)
        if found is None:
            print(f"[error] 在 {search_dir} 未找到 halo_*_family_*.json")
            print("请先生成: python scripts/generate/generate_halo_family.py")
            sys.exit(1)
        family_path = found
        print(f"[info] --latest: 使用 {family_path}")
    elif args.json_file:
        family_path = _resolve_json_path(args.json_file)
    elif args.file:
        family_path = _resolve_json_path(args.file)
    else:
        family_path = Path(FAMILY_JSON_PATH)
        if not family_path.is_absolute():
            family_path = (project_root / family_path).resolve()
        else:
            family_path = family_path.resolve()
        print(f"[info] 使用脚本内 FAMILY_JSON_PATH: {family_path}")

    if not family_path.is_file():
        print(f"[error] 文件不存在: {family_path}")
        print("请修改本脚本顶部的 FAMILY_JSON_PATH，或传入 JSON 路径，或使用 --latest。")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else family_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_halo_family(
        family_path=family_path,
        output_dir=output_dir,
        plot_start=args.start,
        plot_end=args.end,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
