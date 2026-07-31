"""NRHO 打靶收敛性诊断：近月点加密 vs 等时间采样。

打印两种采样策略下的残差收敛曲线，量化近月点加密对 NRHO 多重打靶收敛的改善。
诊断脚本（非测试）：默认无断言，只报告残差改善倍数；``--strict`` 下若近月点
加密残差反而显著大于等时间采样，则以非零退出码示意采样策略可能有误。

数据依赖：
- ``NRHO_REFERENCE_DATA``：指向 ``earth-moon_halo_L2_S.xlsx``（来自
  transfer-orbit-design 仓库的 ``data/cr3bp_data/raw/``）。未设置则报错退出。
- ``SPICE_KERNEL_DIR``：DE440/DE438/DE435 内核所在目录（与测试套件同源）。

运行：``python scripts/nrho_perilune_clustering_diagnostic.py``
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.solver.multiple_shooting import (
    MultipleShooting,
    sample_patch_points,
    sample_patch_points_perilune_clustered,
)
from e2m2e.data.types.orbit import Orbit

MU = 1.21506683e-2
TU_SECONDS = 4.34811305 * 86400  # 秒
REFERENCE_EPOCH = "2025-06-21T11:00:06"


def _resolve_xlsx_path() -> str:
    """从 NRHO_REFERENCE_DATA 环境变量解析 xlsx 路径，缺失则报错退出。"""
    path = os.environ.get("NRHO_REFERENCE_DATA")
    if not path:
        sys.exit("未设置 NRHO_REFERENCE_DATA 环境变量（需指向 earth-moon_halo_L2_S.xlsx）。")
    if not os.path.exists(path):
        sys.exit(f"NRHO_REFERENCE_DATA 指向的文件不存在：{path}")
    return path


def _resolve_kernel_path() -> str:
    """解析 SPICE 内核路径，缺失则报错退出。"""
    kernel_dir = os.environ.get("SPICE_KERNEL_DIR", "")
    for name in ("de440.bsp", "de440s.bsp", "de438.bsp", "de435.bsp"):
        candidate = os.path.join(kernel_dir, name) if kernel_dir else name
        if os.path.exists(candidate):
            return candidate
    sys.exit("未找到 DE440/DE438/DE435 SPICE 内核，请设置 SPICE_KERNEL_DIR。")


def load_nrho(xlsx_path: str, index: int = 0):
    """加载 NRHO 并构造带密集时间序列的 Orbit 供采样。"""
    with zipfile.ZipFile(xlsx_path) as z:
        rows = (
            ET.parse(z.open("xl/worksheets/sheet1.xml"))
            .getroot()
            .findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row")
        )
        cells = []
        for c in rows[index + 1].findall(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"
        ):
            v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
            if v is not None:
                cells.append(float(v.text))
    state0 = np.array(cells[:6])
    period = cells[7]
    sys_ = CR3BP_System(mu=MU, primary="earth", secondary="moon")
    dyn = CR3BP_Dynamics(system=sys_)
    ts = np.linspace(0, period, 200)
    res = dyn.propagate(state0, (0, period), t_eval=ts)
    orbit = Orbit(res["states"], res["time"])
    orbit.period = period
    return orbit, dyn


def run_correction(dynamics, t_patch_j2000, state_patch_j2000, max_iter=15, tol=1e-3):
    """运行多重打靶。诊断用：15 次迭代足以看收敛趋势，tol=1e-3 km 记录残差下限。"""
    ms = MultipleShooting(dynamics=dynamics)
    return ms.correct(
        t_patch=t_patch_j2000,
        state_patch=state_patch_j2000,
        var_time=True,
        max_iter=max_iter,
        tolerance=tol,
    )


def main(strict: bool = False) -> int:
    xlsx_path = _resolve_xlsx_path()
    kernel_path = _resolve_kernel_path()

    # 构造星历动力学（与 tests/conftest.py 的 spice_eph_dynamics fixture 对齐）
    from e2m2e.algorithm.coordinate import SynodicJ2000System
    from e2m2e.algorithm.dynamics.ephemeris_dynamics import EphemerisDynamics
    from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
    from e2m2e.data.kernels.manager import SPICEManager
    from e2m2e.mbse.data.enums import ReferenceFrame

    spice = SPICEManager()
    spice.load_kernel(kernel_path)
    try:
        reference_et = spice.utc_to_et(REFERENCE_EPOCH)
        cr3bp_sys = CR3BP_System(mu=MU, primary="earth", secondary="moon")
        cr3bp_dyn = CR3BP_Dynamics(system=cr3bp_sys)
        syn_j2000 = SynodicJ2000System(cr3bp_system=cr3bp_sys, spice=spice)

        eph_system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"],
            spice=spice,
            origin="EARTH",
            frame=ReferenceFrame.J2000,
        )
        eph_dyn = EphemerisDynamics(system=eph_system)
        # 与测试套件一致的宽松参数（生产用 1e-12）
        eph_dyn.rtol = 1e-10
        eph_dyn.atol = 1e-10
        eph_dyn.max_step = 600.0

        orbit, _ = load_nrho(xlsx_path, 0)
        tc = TU_SECONDS

        # --- 等时间采样（基线）---
        t_eq, s_eq = sample_patch_points(orbit, 8)
        state_eq_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=s_eq, t_syn_arr=t_eq, et0=reference_et
        )
        t_eq_j2000 = reference_et + t_eq * tc

        # --- 近月点加密 ---
        t_cl, s_cl = sample_patch_points_perilune_clustered(
            orbit, cr3bp_dyn, n_base=6, n_perilune=5, perilune_window=0.12
        )
        state_cl_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=s_cl, t_syn_arr=t_cl, et0=reference_et
        )
        t_cl_j2000 = reference_et + t_cl * tc

        print(f"\n{'=' * 70}")
        print("NRHO 打靶收敛诊断（EphemerisDynamics 点质量）")
        print(f"{'=' * 70}")
        print(f"等时间: {len(t_eq)} 点, t/T = {np.round(t_eq / orbit.period, 3)}")
        print(f"近月点加密: {len(t_cl)} 点, t/T = {np.round(t_cl / orbit.period, 3)}")

        res_eq = run_correction(eph_dyn, t_eq_j2000, state_eq_j2000)
        res_cl = run_correction(eph_dyn, t_cl_j2000, state_cl_j2000)

        print(f"\n--- 等时间采样 ({len(t_eq)} 点) ---")
        print(f"  收敛: {res_eq.converged}, 迭代: {res_eq.outer_iterations}")
        print(f"  最终残差: {res_eq.max_residual:.4e} km")
        print(f"  残差历史: {[f'{r:.2e}' for r in res_eq.residual_history]}")

        print(f"\n--- 近月点加密 ({len(t_cl)} 点) ---")
        print(f"  收敛: {res_cl.converged}, 迭代: {res_cl.outer_iterations}")
        print(f"  最终残差: {res_cl.max_residual:.4e} km")
        print(f"  残差历史: {[f'{r:.2e}' for r in res_cl.residual_history]}")

        improvement = res_eq.max_residual / max(res_cl.max_residual, 1e-30)
        print(f"\n残差改善倍数: {improvement:.1f}x")
        print(f"{'=' * 70}\n")

        if strict and res_cl.max_residual > res_eq.max_residual * 2:
            print(
                f"  ✗ 近月点加密残差 {res_cl.max_residual:.2e} 远大于等时间 "
                f"{res_eq.max_residual:.2e}，采样策略可能有误",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        spice.unload_kernel(kernel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="近月点加密残差显著劣于等时间采样时以非零码退出",
    )
    args = parser.parse_args()
    raise SystemExit(main(strict=args.strict))
