"""NRHO 星历修正回归（#463 / #473）。

#463：默认近月点加密在贴月 NRHO 上残差卡约 10² km。
#473：5.7.2 的删近月点默认在 GUI 量级（phase=0.5、约 1 个月）上
仍有历元空洞（星历长度断言）与合并层不收敛；生产默认改为等时间 +
``revs_per_group=1``。

本文件只锁 ``design_orbit`` 对外行为：

- 贴月短弧（近月高 2000 km、约 8 天）——#463 场景不得回退。
- GUI 默认量级（近月高 5000 km、phase=0.5、约 30 天、1 h 步长）——
  #473 必锁：收敛 + 星历与时间网格等长。
- 采样×段长的开发期对照矩阵脚本已随 #472 收口移除（见 git 历史）。
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

# 贴月短弧：~1 个 NRHO 周期（2000 km 约 6.7 天）。相位 0.5 是新生产
# 离散默认下稳定的贴月代表样本；原 0.0 旧删近月采样案例不再适用（#473）。
TIGHT_DURATION_SEC = 8 * 86400.0
TIGHT_OUTPUT_STEP_SEC = 7200.0  # 2 h，压低星历表规模

# GUI 默认量级：L2 南、5000 km、phase=0.5、约 1 个月、1 h 输出。
GUI_DURATION_SEC = 30 * 86400.0
GUI_OUTPUT_STEP_SEC = 3600.0

# phase=0 与请求模型的 [0, 1] 契约一致；取短弧只验证该相位的修正可用性。
PHASE_ZERO_DURATION_SEC = 8 * 86400.0
PHASE_ZERO_OUTPUT_STEP_SEC = 7200.0

# #508：9:2 贴月共振成员（近月高 1500 km）略超一圈的 8 天弧。主路径
# （1 圈/段 + 合并层）对该成员不收敛，须回退单段 2 圈。
RESONANT_DURATION_SEC = 8 * 86400.0
RESONANT_OUTPUT_STEP_SEC = 7200.0


@pytest.fixture(scope="module")
def nrho_tight_short_result():
    """L2 南族、近月高 2000 km、8 天 segmented——#463 最小回归样本。"""
    return design_orbit(
        make_design_request(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=2000.0,
            phase=0.5,
            duration=TIGHT_DURATION_SEC,
            output_step=TIGHT_OUTPUT_STEP_SEC,
            correction_method="segmented",
        )
    )


@pytest.fixture(scope="module")
def nrho_gui_default_result():
    """GUI 默认量级 NRHO——#473 回归样本（等时间 + 1 圈/段）。"""
    return design_orbit(
        make_design_request(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=5000.0,
            phase=0.5,
            duration=GUI_DURATION_SEC,
            output_step=GUI_OUTPUT_STEP_SEC,
            correction_method="segmented",
        )
    )


@pytest.fixture(scope="module")
def nrho_phase_zero_result():
    """L2 南、5000 km、phase=0——请求模型允许的相位端点。"""
    return design_orbit(
        make_design_request(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=5000.0,
            phase=0.0,
            duration=PHASE_ZERO_DURATION_SEC,
            output_step=PHASE_ZERO_OUTPUT_STEP_SEC,
            correction_method="segmented",
        )
    )


@pytest.fixture(scope="module")
def nrho_resonant_result():
    """9:2 贴月共振成员 8 天弧——#508 回退链回归样本。"""
    return design_orbit(
        make_design_request(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=1500.0,
            phase=0.5,
            duration=RESONANT_DURATION_SEC,
            output_step=RESONANT_OUTPUT_STEP_SEC,
            correction_method="segmented",
        )
    )


def test_tight_short_nrho_converges(nrho_tight_short_result):
    """贴月短弧修正收敛：#463 场景在新默认策略下仍可用。"""
    res = nrho_tight_short_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction_method == "segmented"
    assert res.correction.max_residual < 2e-2


def test_tight_short_nrho_ephemeris_aligned(nrho_tight_short_result):
    """贴月短弧星历非空且与时间网格等长。"""
    eph = nrho_tight_short_result.ephemeris
    n_expected = int(TIGHT_DURATION_SEC / TIGHT_OUTPUT_STEP_SEC) + 1
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


def test_nrho_phase_zero_converges(nrho_phase_zero_result):
    """phase=0 不被算法层拒绝，并能得到收敛的标称星历。"""
    res = nrho_phase_zero_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.ephemeris is not None


def test_resonant_slightly_over_one_rev_converges(nrho_resonant_result):
    """略超一圈的 9:2 贴月成员收敛（#508：主路径不收敛须回退单段）。"""
    res = nrho_resonant_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction.max_residual < 2e-2
    eph = res.ephemeris
    n_expected = int(RESONANT_DURATION_SEC / RESONANT_OUTPUT_STEP_SEC) + 1
    assert eph is not None
    assert len(eph) == n_expected


def test_gui_default_nrho_converges(nrho_gui_default_result):
    """GUI 默认量级（phase=0.5、约 1 个月）收敛。"""
    res = nrho_gui_default_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction.max_residual < 2e-2


def test_gui_default_nrho_ephemeris_aligned(nrho_gui_default_result):
    """星历非空且点数与时间网格严格一致（#473 历元洞回归）。"""
    eph = nrho_gui_default_result.ephemeris
    # et_grid = arange(0, duration + 0.5*step, step) → 30 天 / 1 h = 721 点
    n_expected = int(GUI_DURATION_SEC / GUI_OUTPUT_STEP_SEC) + 1
    assert eph is not None
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected
