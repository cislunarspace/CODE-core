"""``design_orbit`` 编排入口的最小真实调用冒烟（ADR 0037 决策 2）。

选 ELFO 场景：无星历修正（``correction=None``），仅“经典根数 → 全摄动传播
→ 月心漂移分析”一段链路，是 ``design_orbit`` 最便宜的端到端真实路径；最短
弧段（约一个轨道周期）证明链路连通与返回类型契约，不重复物理可行性穷举。
长弧/多候选的冻结轨道集成覆盖已随端到端清理移除（见 ADR 0037 增补）。

本文件同时覆盖 LYAPUNOV CR3BP 初猜到 segmented 星历修正的真实路径。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.design import design_orbit
from e2m2e.algorithm.design.tuning import CORRECTION_TOL_KM
from e2m2e.status import ConvergenceState
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


# RO 的星历链路冒烟按 ADR 0037 归属 scripts/design_ro_ephemeris_smoke.py:
# tests/conftest.py 把 Rust rayon 钉单线程, two_level 星历修正单线程实测
# >600 s(多线程独立进程实测 219 s), 超出 pytest 可容规模。
# Lyapunov 的星历链路冒烟：实测 46.23 s（debug 扩展、完整 CR3BP + segmented 链路）。
def test_design_orbit_lyapunov_ephemeris_correction_smoke():
    result = design_orbit(
        make_design_request(
            orbit_type="LYAPUNOV",
            collinear_point=2,
            amplitude=12000.0,
            duration=1_270_000.0,
            output_step=7200.0,
        )
    )
    assert result.orbit_type == "LYAPUNOV"
    assert result.cr3bp_orbit is not None
    assert result.correction is not None
    assert result.correction.status is ConvergenceState.CONVERGED
    assert result.correction.max_residual < CORRECTION_TOL_KM
    assert result.correction_method == "segmented"
    assert len(result.ephemeris) > 0


def test_design_orbit_dpo_default_rev_budget_smoke():
    """#841：DPO 默认请求按两圈折算时长，segmented 星历修正收敛。

    主族单圈成员周期约 3.33 天，默认两圈约 6.7 天。固定一年会把圈数推到
    110 圈，实测一小时未完成，见 ADR 0061 的复审修订。
    """
    result = design_orbit(make_design_request(orbit_type="DPO", amplitude=20000.0))
    assert result.correction is not None
    assert result.correction.status is ConvergenceState.CONVERGED
    assert result.correction.max_residual < CORRECTION_TOL_KM
    assert 6.0 < result.duration_day < 7.0
    assert len(result.ephemeris) > 0
