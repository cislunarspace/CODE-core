"""冻结轨道设计集成测试（L2，需 SPICE 内核）。

通过 ``design_orbit(DesignOrbitRequest(orbit_type="ELFO", ...))`` 公共入口
驱动单候选传播，验证返回结构与物理一致性。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"),
)
_SPICE_AVAILABLE = os.path.isdir(_SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.l2,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
]


def test_single_candidate_structure():
    """a=3000 km 候选传播 3 天，返回 OrbitDesignResult 且字段正确。"""
    result = design_orbit(
        DesignOrbitRequest(
            orbit_type="ELFO",
            semi_major_axis=3000.0,
            duration=3 * 86400.0,
            output_step=3600.0,
        )
    )
    assert result.orbit_type == "ELFO"
    assert result.cr3bp_orbit is None
    assert result.correction is None
    assert np.isnan(result.cr3bp_jacobi)
    assert result.drift_e is not None
    assert result.drift_rp_km is not None
    assert result.initial_state.shape == (6,)
    assert len(result.ephemeris) == 73  # 3 天 / 1 小时 + 1
    assert result.moon_centric_elements is not None
    assert "aop" in result.moon_centric_elements
