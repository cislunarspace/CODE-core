"""NRHO 星历修正回归（#463）。

根因是默认「近月点加密」离散策略：贴月 NRHO 上 LM 残差卡在约 10² km，
或拼接点网格与积分输出长度不一致。对照实验表明「删近月点附近节点」可收敛。

本文件只锁 ``design_orbit`` 对外行为，规模按 ADR 0021 收紧：

- 单条贴月短弧（近月高 2000 km、约 1 个 NRHO 周期）——修复前必失败、
  修复后收敛；足够覆盖采样策略接缝，不必再挂 30 天 GUI 全长。
- GUI 默认量级与采样×段长矩阵见 ``scripts/nrho_ephemeris_correction_matrix.py``，
  不进默认 pytest。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.design import design_orbit
from e2m2e.data.templates import ConvergenceState
from tests.algorithm.design.conftest import make_design_request

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
    requires_spice,
]

# 贴月短弧：~1 个 NRHO 周期（2000 km 约 6.7 天）。修复前 clustered 在此
# 规模即失败（残差 ~10² km 或星历长度不一致）。
DURATION_SEC = 8 * 86400.0
OUTPUT_STEP_SEC = 7200.0  # 2 h 步长，压低星历表规模，不影响打靶收敛判据


@pytest.fixture(scope="module")
def nrho_tight_short_result():
    """L2 南族、近月高 2000 km、8 天 segmented——#463 最小回归样本。"""
    return design_orbit(
        make_design_request(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=2000.0,
            phase=0.0,
            duration=DURATION_SEC,
            output_step=OUTPUT_STEP_SEC,
            correction_method="segmented",
        )
    )


def test_tight_short_nrho_converges(nrho_tight_short_result):
    """贴月短弧修正收敛：采样策略接缝的行为锁。"""
    res = nrho_tight_short_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction.max_residual < 2e-2


def test_tight_short_nrho_ephemeris_aligned(nrho_tight_short_result):
    """星历非空且位置/速度/会合系与时间网格等长。"""
    eph = nrho_tight_short_result.ephemeris
    n_expected = int(DURATION_SEC / OUTPUT_STEP_SEC) + 1
    assert eph is not None
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected


def test_tight_short_nrho_synodic_near_rectilinear(nrho_tight_short_result):
    """会合系保持近直线形态（贴月、|y| 小、南族 z 负向幅度）。"""
    syn = np.asarray(nrho_tight_short_result.ephemeris.synodic_position)
    x, y, z = syn[:, 0], syn[:, 1], syn[:, 2]
    assert x.min() > 0.95 and x.max() < 1.12, f"会合系 x 超界: [{x.min():.4f}, {x.max():.4f}]"
    assert abs(y).max() < 0.12, f"会合系 |y| 过大: {abs(y).max():.4f}"
    assert z.min() < -0.1, f"南族 z 幅度不足: zmin={z.min():.4f}"
    assert abs(y).max() < (z.max() - z.min()), (
        f"非近直线: |y|_max={abs(y).max():.4f}, z_span={z.max() - z.min():.4f}"
    )
