"""``design_orbit`` 编排入口的最小真实调用冒烟（ADR 0037 决策 2）。

选 ELFO 场景：无星历修正（``correction=None``），仅"经典根数 → 全摄动传播
→ 月心漂移分析"一段链路，是 ``design_orbit`` 最便宜的端到端真实路径；最短
弧段（约一个轨道周期）证明链路连通与返回类型契约，不重复物理可行性穷举。
长弧/多候选的冻结轨道集成覆盖已随端到端清理移除（见 ADR 0037 增补）。
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


def test_design_orbit_elfo_minimal_real_call():
    """最小 ELFO 真实调用：链路连通 + 返回类型契约（ELFO 无星历修正）。"""
    result = design_orbit(
        make_design_request(
            orbit_type="ELFO",
            semi_major_axis=3000.0,
            inclination=75.0,
            arg_of_pericenter=270.0,
            perilune_height=200.0,
            duration=14400.0,  # ≈1 个轨道周期（a=3000 km 绕月），最短有效弧段
            output_step=1200.0,
        )
    )
    assert result.orbit_type == "ELFO"
    assert result.cr3bp_orbit is None
    assert result.correction is None
    assert result.correction_method is None
    assert np.isnan(result.cr3bp_jacobi)
    assert result.initial_state.shape == (6,)
    assert len(result.ephemeris) > 0
    assert result.drift_e is not None
    assert result.moon_centric_elements is not None


@pytest.mark.time_budget(40)  # 星历修正 1 圈实测约 17 s，不可再压；40s 为 ≥2 倍余量
def test_design_orbit_ro_31_minimal_real_call():
    """RO 3:1 最小真实调用（#627 验收）：闭合残差 < 1e-6 的精确通约周期轨道。

    链路：共振周期条件 Kepler 初猜 → 固定半周期 CR3BP 修正（T = T☾/3）
    → 星历修正（two_level 固定时刻，近圆轨道时间平移病态故不用
    var_time）→ 1 圈标称星历。断言整条链路收敛 + CR3BP 轨道闭合残差
    达标 + 分类学实测打标 resonant_3_1。
    """
    result = design_orbit(
        make_design_request(
            orbit_type="RO",
            resonance_p=3,
            resonance_q=1,
            phase=0.0,
            duration=800000.0,  # ≈1 个 RO 周期（9.1 天），最短有效弧段
            output_step=14400.0,
        )
    )
    assert result.orbit_type == "RO"
    assert result.status is ConvergenceState.CONVERGED
    orbit = result.cr3bp_orbit
    assert orbit is not None and orbit.period is not None
    # 闭合残差（无量纲）达标 + 周期精确通约（T = T☾/3）
    assert orbit.closure_error is not None and orbit.closure_error < 1e-6
    assert abs(orbit.period - 2.0 * np.pi / 3.0) / (2.0 * np.pi / 3.0) < 1e-9
    # 星历修正收敛（固定时刻打靶）
    assert result.correction is not None
    assert result.correction.status is ConvergenceState.CONVERGED
    assert len(result.ephemeris) > 0
