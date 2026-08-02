"""L4/L5 三角平动点轨道设计端到端测试。

通过 ``design_orbit("L4"/"L5", ...)`` 全流程（CR3BP 初猜 → two_level
星历修正 → 高精度长期预报），验证收敛性、振幅有界、输出形状正确。

依赖 SPICE 内核（``design_orbit`` 自动加载 ``kernels/``）。
内核缺失时整组跳过。
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from e2m2e.algorithm.design import design_orbit

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels"),
)
_SPICE_AVAILABLE = os.path.isdir(_SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
]


class TestDesignOrbitTriangular:
    """L4/L5 设计端到端。"""

    @pytest.mark.parametrize("point", ["L4", "L5"])
    def test_end_to_end_converges(self, point):
        """CR3BP → two_level 星历修正 → 高精度传播，全流程不抛异常。"""
        result = design_orbit(
            point,
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=0.05,
            output_step=3600.0,
        )
        assert result.correction.converged
        assert result.ephemeris is not None
        assert result.orbit_type == point

    def test_amplitude_bounded(self):
        """传播后位置振幅不爆炸——地月空间合理范围。"""
        result = design_orbit(
            "L4",
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=0.05,
        )
        eph = result.ephemeris
        pos_norms = np.linalg.norm(eph.position_km, axis=1)
        assert pos_norms.max() < 1e6, f"最大地心距 {pos_norms.max():.0f} km 超出合理范围"
        assert pos_norms.min() > 1e4, f"最小地心距 {pos_norms.min():.0f} km 超出合理范围"

    def test_output_shape(self):
        """输出 ephemeris 形状正确（行数、列结构）。"""
        duration = 0.05
        output_step = 3600.0
        result = design_orbit(
            "L5",
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=duration,
            output_step=output_step,
        )
        eph = result.ephemeris
        expected_rows = math.ceil(duration * 365.25 * 86400 / output_step)
        assert abs(len(eph) - expected_rows) <= 2, f"预期约 {expected_rows} 行，实际 {len(eph)} 行"
        assert eph.position_km.shape == (len(eph), 3)
        assert eph.velocity_mps.shape == (len(eph), 3)
        assert eph.synodic_position.shape == (len(eph), 3)

    def test_epoch_matches_input(self):
        """起始历元匹配输入 epoch。"""
        result = design_orbit(
            "L4",
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=0.05,
        )
        eph = result.ephemeris
        assert eph.year[0] == 2025
        assert eph.month[0] == 1
        assert eph.day[0] == 1

    def test_initial_state_shape(self):
        """initial_state 为 6 维向量 (km, km/s)。"""
        result = design_orbit(
            "L5",
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=0.05,
        )
        assert result.initial_state.shape == (6,)

    def test_cr3bp_orbit_present(self):
        """cr3bp_orbit 和 jacobi 应存在。"""
        result = design_orbit(
            "L4",
            amplitude_in=300.0,
            amplitude_out=1000.0,
            phase_in=0.0,
            phase_out=0.0,
            epoch="2025-01-01T00:00:00",
            duration=0.05,
        )
        assert result.cr3bp_orbit is not None
        assert isinstance(result.cr3bp_jacobi, float)
        assert result.cr3bp_orbit.states is not None
        assert len(result.cr3bp_orbit.states) > 0
