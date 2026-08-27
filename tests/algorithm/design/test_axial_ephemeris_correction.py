"""Axial 星历修正回归。

GUI 默认参数（L2、振幅 5000 km、phase=0、约 1 个月、two_level）必须走
固定时间打靶：Axial 从 Lyapunov 族 1:1 共振分岔（Gómez Type B）产生，
分岔邻域面内周期 = 面外周期，时间平移与面外相位平移近似简并，雅可比
列病态，自由时间打靶在此会 LM 停滞（STAGNATION_DETECTED）。固定时间
后约 10 s 收敛到容差内。

本文件只锁 ``design_orbit`` 对外行为：GUI 默认量级收敛 + 星历与时间
网格等长。
"""

from __future__ import annotations

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

# GUI 默认量级：L2、5000 km、phase=0、约 1 个月、1 h 输出、two_level。
GUI_DURATION_SEC = 30 * 86400.0
GUI_OUTPUT_STEP_SEC = 3600.0


@pytest.fixture(scope="module")
def axial_gui_default_result():
    """GUI 默认参数 Axial 固定时间打靶回归样本。"""
    return design_orbit(
        make_design_request(
            orbit_type="AXIAL",
            collinear_point=2,
            amplitude=5000.0,
            phase=0.0,
            duration=GUI_DURATION_SEC,
            output_step=GUI_OUTPUT_STEP_SEC,
            correction_method="two_level",
        )
    )


def test_gui_default_axial_converges(axial_gui_default_result):
    """GUI 默认量级 Axial 收敛（自由时间打靶在简并邻域停滞，不得回退）。"""
    res = axial_gui_default_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction.max_residual < 2e-2
    assert res.correction_method == "two_level"


def test_gui_default_axial_ephemeris_aligned(axial_gui_default_result):
    """星历非空且点数与时间网格严格一致。"""
    eph = axial_gui_default_result.ephemeris
    # et_grid = arange(0, duration + 0.5*step, step) → 30 天 / 1 h = 721 点
    n_expected = int(GUI_DURATION_SEC / GUI_OUTPUT_STEP_SEC) + 1
    assert eph is not None
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected
