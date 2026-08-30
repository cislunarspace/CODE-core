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
