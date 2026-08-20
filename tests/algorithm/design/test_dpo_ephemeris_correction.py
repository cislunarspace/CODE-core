"""DPO 星历修正回归（#484）。

DPO 在星历模型下不稳定，不能沿用 DRO 的"修正一圈后自由外推"路径。
本模块锁住 GUI 默认量级：传入 ``two_level`` 后算法自动改走全程分段打靶，
并验证修正收敛、星历时间网格无缺口及轨迹仍保持月球附近有界。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.design import design_orbit
from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system
from e2m2e.data.templates import ConvergenceState
from tests.algorithm.design.conftest import make_design_request

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
    requires_spice,
]

GUI_DURATION_SEC = 30 * 86400.0
GUI_OUTPUT_STEP_SEC = 3600.0


@pytest.fixture(scope="module")
def dpo_gui_default_result():
    """GUI 默认量级 DPO，two_level 入参应自动重定向为 segmented。"""
    return design_orbit(
        make_design_request(
            orbit_type="DPO",
            amplitude=20000.0,
            phase=0.5001,
            duration=GUI_DURATION_SEC,
            output_step=GUI_OUTPUT_STEP_SEC,
            correction_method="two_level",
        )
    )


def test_gui_default_dpo_converges(dpo_gui_default_result):
    """GUI 默认量级 DPO 经自动重定向后收敛。"""
    res = dpo_gui_default_result
    assert res.correction is not None
    assert res.correction.status is ConvergenceState.CONVERGED
    assert res.correction.max_residual < 2e-2


def test_gui_default_dpo_ephemeris_aligned(dpo_gui_default_result):
    """星历非空，且点数与 1 h 时间网格严格一致。"""
    eph = dpo_gui_default_result.ephemeris
    n_expected = int(GUI_DURATION_SEC / GUI_OUTPUT_STEP_SEC) + 1
    assert eph is not None
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected


def test_gui_default_dpo_stays_moon_bounded(dpo_gui_default_result):
    """修正后轨迹不脱离月球附近的 DPO 包络。"""
    system = earth_moon_system()
    assert system.characteristic_length is not None
    syn = np.asarray(dpo_gui_default_result.ephemeris.synodic_position)
    # EphemerisTable 的会合系位置是地心无量纲坐标；恢复质心坐标后再取月距。
    barycentric_syn = syn + np.array([system.mu, 0.0, 0.0])
    moon_position = np.array([1.0 - system.mu, 0.0, 0.0])
    moon_distance_km = (
        np.linalg.norm(barycentric_syn - moon_position, axis=1) * system.characteristic_length
    )
    assert moon_distance_km.min() > 5000.0, f"月距过近: {moon_distance_km.min():.0f} km"
    assert moon_distance_km.max() < 40000.0, f"月距发散: {moon_distance_km.max():.0f} km"
