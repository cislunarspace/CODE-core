"""Lissajous 轨道设计端到端测试。

通过 ``design_orbit(DesignOrbitRequest(orbit_type="LISSAJOUS", ...))`` 全流程
（CR3BP 初猜 → two_level 星历修正 → 高精度长期预报），验证收敛性、振幅有界、
输出形状正确。

依赖 SPICE 内核（``design_orbit`` 自动加载 ``kernels/``）。
内核缺失时整组跳过。

属 tests/orbit_design 三层分层中的 L3（scenarios，端到端）：在 L1 初猜
与 L2 修正/延拓的单点校验之上，覆盖 design_orbit 全链路集成行为。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels"),
)
_SPICE_AVAILABLE = os.path.isdir(_SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
    pytest.mark.l3,
]

# 0.05 年 ≈ 18 天
DURATION_SEC = 0.05 * 365.25 * 86400


class TestDesignOrbitLissajous:
    """Lissajous 设计端到端。"""

    @pytest.mark.parametrize("collinear_point", [1, 2])
    def test_end_to_end_converges(self, collinear_point):
        """CR3BP → two_level 星历修正 → 高精度传播，全流程不抛异常。"""
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=collinear_point,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
                output_step=3600.0,
            )
        )
        assert result.correction.converged
        assert result.ephemeris is not None
        assert result.orbit_type == "LISSAJOUS"

    def test_amplitude_bounded(self):
        """传播后位置振幅不爆炸——地月空间合理范围。"""
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=2,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
            )
        )
        eph = result.ephemeris
        pos_norms = np.linalg.norm(eph.position_km, axis=1)
        assert pos_norms.max() < 1e6, f"最大地心距 {pos_norms.max():.0f} km 超出地月空间合理范围"
        assert pos_norms.min() > 1e4, f"最小地心距 {pos_norms.min():.0f} km 超出地月空间合理范围"

    def test_output_shape(self):
        """输出 ephemeris 形状正确（行数、列结构）。"""
        output_step = 3600.0
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=2,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
                output_step=output_step,
            )
        )
        eph = result.ephemeris
        expected_rows = math.ceil(DURATION_SEC / output_step)
        # 允许 ±2 行容差（累积舍入）
        assert abs(len(eph) - expected_rows) <= 2, f"预期约 {expected_rows} 行，实际 {len(eph)} 行"
        assert eph.position_km.shape == (len(eph), 3)
        assert eph.velocity_mps.shape == (len(eph), 3)
        assert eph.synodic_position.shape == (len(eph), 3)

    def test_epoch_matches_input(self):
        """起始历元匹配输入 epoch。"""
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=2,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
            )
        )
        eph = result.ephemeris
        assert eph.year[0] == 2025
        assert eph.month[0] == 1
        assert eph.day[0] == 1

    def test_initial_state_shape(self):
        """initial_state 为 6 维向量 (km, km/s)。"""
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=2,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
            )
        )
        assert result.initial_state.shape == (6,)

    def test_cr3bp_orbit_present(self):
        """cr3bp_orbit 和 jacobi 应存在。"""
        result = design_orbit(
            DesignOrbitRequest(
                orbit_type="LISSAJOUS",
                collinear_point=2,
                amplitude_in=500.0,
                amplitude_out=2000.0,
                phase_in=0.0,
                phase_out=0.0,
                epoch="2025-01-01T00:00:00",
                duration=DURATION_SEC,
            )
        )
        assert result.cr3bp_orbit is not None
        assert isinstance(result.cr3bp_jacobi, float)
        assert result.cr3bp_orbit.states is not None
        assert len(result.cr3bp_orbit.states) > 1
