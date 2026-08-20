"""冻结轨道设计集成测试（L2，需 SPICE 内核）。

通过 ``design_orbit`` 公共入口（duck-typed 请求，接口层模型的校验由
tests/api 覆盖）驱动单候选传播，验证返回结构与物理一致性。
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


def test_single_candidate_structure():
    """a=3000 km 候选传播 3 天，返回 OrbitDesignResult 且字段正确。"""
    result = design_orbit(
        make_design_request(
            orbit_type="ELFO",
            semi_major_axis=3000.0,
            inclination=75.0,
            arg_of_pericenter=270.0,
            perilune_height=200.0,
            duration=3 * 86400.0,
            output_step=3600.0,
        )
    )
    assert result.orbit_type == "ELFO"
    assert result.cr3bp_orbit is None
    assert result.correction is None
    assert result.correction_method is None
    assert np.isnan(result.cr3bp_jacobi)
    assert result.drift_e is not None
    assert result.drift_rp_km is not None
    assert result.initial_state.shape == (6,)
    assert len(result.ephemeris) == 73  # 3 天 / 1 小时 + 1
    assert result.moon_centric_elements is not None
    assert "aop" in result.moon_centric_elements
