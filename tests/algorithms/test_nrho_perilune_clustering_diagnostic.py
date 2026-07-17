"""NRHO 打靶收敛性诊断：近月点加密 vs 等时间采样。

这不是严格的 pass/fail 测试，而是诊断工具：打印两种采样策略下的残差收敛
曲线，量化近月点加密对 NRHO 多重打靶收敛的改善。

运行：pytest tests/algorithms/test_nrho_perilune_clustering_diagnostic.py -v -s
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from e2m2e.algorithms import MultipleShooting
from e2m2e.algorithms.multiple_shooting import (
    sample_patch_points,
    sample_patch_points_perilune_clustered,
)
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit

pytestmark = pytest.mark.spice

MU = 1.21506683e-2
TU_SECONDS = 4.34811305 * 86400  # 秒
XLSX_PATH = "/home/ouyangjiahong/codes/transfer-orbit-design/data/cr3bp_data/raw/earth-moon_halo_L2_S.xlsx"


def _load_nrho(index: int = 0):
    """加载 NRHO 并构造带密集时间序列的 Orbit 供采样。"""
    with zipfile.ZipFile(XLSX_PATH) as z:
        rows = ET.parse(z.open("xl/worksheets/sheet1.xml")).getroot().findall(
            ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"
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
    # 一次积分一圈，供插值采样
    ts = np.linspace(0, period, 200)
    res = dyn.propagate(state0, (0, period), t_eval=ts)
    orbit = Orbit(res["states"], res["time"])
    orbit.period = period
    return orbit, dyn


def _run_correction(dynamics, t_patch_j2000, state_patch_j2000, max_iter=15, tol=1e-3):
    """运行多重打靶并返回结果。

    诊断用：15 次迭代足以看收敛趋势，tol=1e-3 km 记录残差下限。
    """
    ms = MultipleShooting(dynamics=dynamics)
    return ms.correct(
        t_patch=t_patch_j2000,
        state_patch=state_patch_j2000,
        var_time=True,
        max_iter=max_iter,
        tolerance=tol,
    )


@pytest.fixture
def nrho_orbit_dense(cr3bp_system, cr3bp_dynamics):
    """加载 NRHO 并构造带密集时间序列的 Orbit 供采样。"""
    return _load_nrho(0)


class TestNRHOPeriluneClusteringDiagnostic:
    """对比等时间采样 vs 近月点加密的 NRHO 打靶残差。"""

    def test_diagnostic_equal_vs_clustered(
        self, cr3bp_dynamics, spice_syn_j2000, spice_eph_dynamics, reference_et
    ):
        """打印两种采样策略的残差收敛曲线。"""
        orbit, _ = _load_nrho(0)
        cr3bp_dyn = cr3bp_dynamics
        tc = TU_SECONDS

        # --- 等时间采样（基线）---
        t_eq, s_eq = sample_patch_points(orbit, 8)
        state_eq_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=s_eq, t_syn_arr=t_eq, et0=reference_et
        )
        t_eq_j2000 = reference_et + t_eq * tc

        # --- 近月点加密 ---
        t_cl, s_cl = sample_patch_points_perilune_clustered(
            orbit, cr3bp_dyn, n_base=6, n_perilune=5, perilune_window=0.12
        )
        state_cl_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=s_cl, t_syn_arr=t_cl, et0=reference_et
        )
        t_cl_j2000 = reference_et + t_cl * tc

        print(f"\n{'='*70}")
        print(f"NRHO 打靶收敛诊断（EphemerisDynamics 点质量）")
        print(f"{'='*70}")
        print(f"等时间: {len(t_eq)} 点, t/T = {np.round(t_eq/orbit.period, 3)}")
        print(f"近月点加密: {len(t_cl)} 点, t/T = {np.round(t_cl/orbit.period, 3)}")

        res_eq = _run_correction(spice_eph_dynamics, t_eq_j2000, state_eq_j2000)
        res_cl = _run_correction(spice_eph_dynamics, t_cl_j2000, state_cl_j2000)

        print(f"\n--- 等时间采样 ({len(t_eq)} 点) ---")
        print(f"  收敛: {res_eq.converged}, 迭代: {res_eq.outer_iterations}")
        print(f"  最终残差: {res_eq.max_residual:.4e} km")
        print(f"  残差历史: {[f'{r:.2e}' for r in res_eq.residual_history]}")

        print(f"\n--- 近月点加密 ({len(t_cl)} 点) ---")
        print(f"  收敛: {res_cl.converged}, 迭代: {res_cl.outer_iterations}")
        print(f"  最终残差: {res_cl.max_residual:.4e} km")
        print(f"  残差历史: {[f'{r:.2e}' for r in res_cl.residual_history]}")

        # 量化改善
        improvement = res_eq.max_residual / max(res_cl.max_residual, 1e-30)
        print(f"\n残差改善倍数: {improvement:.1f}x")
        print(f"{'='*70}")

        # 诊断性断言（非严格 pass/fail，记录现状）
        # 近月点加密应使残差更小——如果反而更大，说明采样策略有问题
        assert res_cl.max_residual <= res_eq.max_residual * 2, (
            f"近月点加密残差 {res_cl.max_residual:.2e} 远大于等时间 {res_eq.max_residual:.2e}，"
            f"采样策略可能有误"
        )
