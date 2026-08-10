"""algorithm/propagation 模块测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest
from kernel_helpers import SPICE_KERNEL_DIR

from e2m2e.algorithm.propagation import _extract_bodies, propagate_orbit

pytestmark = pytest.mark.integrator


# ---------------------------------------------------------------------------
# SPICE 与 kernel 可用性检测
# ---------------------------------------------------------------------------


def _has_spice_kernels() -> bool:
    """检查 SPICE .tls + .bsp 都可用。"""
    if not os.path.isdir(SPICE_KERNEL_DIR):
        return False
    has_tls = any(f.endswith(".tls") for f in os.listdir(SPICE_KERNEL_DIR))
    has_bsp = any(f.endswith(".bsp") for f in os.listdir(SPICE_KERNEL_DIR))
    return has_tls and has_bsp


_requires_spice = pytest.mark.skipif(
    not _has_spice_kernels(),
    reason="SPICE kernels (.tls + .bsp) not available",
)


class TestExtractBodies:
    def test_single_body(self):
        cfg = {
            "version": 1,
            "forces": [
                {
                    "name": "g",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH"},
                }
            ],
        }
        assert _extract_bodies(cfg) == ["EARTH"]

    def test_unique_and_uppercase(self):
        cfg = {
            "version": 1,
            "forces": [
                {
                    "name": "g1",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "earth"},
                },
                {
                    "name": "g2",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH"},
                },
                {
                    "name": "g3",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "MOON"},
                },
            ],
        }
        assert _extract_bodies(cfg) == ["EARTH", "MOON"]

    def test_empty_forces(self):
        assert _extract_bodies({"version": 1, "forces": []}) == []

    def test_missing_body(self):
        assert _extract_bodies({"version": 1, "forces": [{"params": {"mu": 1.0}}]}) == []


@_requires_spice
class TestPropagateOrbit:
    def test_default_three_body(self, spice_manager, reference_epoch):
        # 用默认三体配置（_DEFAULT_FORCE_CONFIG）：地球点质量 + 月球/太阳
        # ThirdBodyGravity。历史 bug：该配置曾把 MOON/SUN 配成朝向地心的
        # PointMassGravity，太阳 mu 主导使任何合理初值都步长坍缩；现已修复，
        # 本测试即其回归保护。
        initial_state = np.array(
            [
                7000.0,  # km
                0.0,
                0.0,
                0.0,  # km/s
                7.7,
                0.0,
            ]
        )
        result = propagate_orbit(
            initial_state=initial_state,
            epoch=reference_epoch,
            duration=86400.0,
            output_step=3600.0,
        )
        assert len(result) == 25  # 0..24h inclusive
        assert result.position_km.shape == (25, 3)
        assert result.velocity_mps.shape == (25, 3)
        assert not np.allclose(result.position_km[0], result.position_km[-1])

    def test_epoch_tuple(self, spice_manager):
        # 非奇异初值（地球点质量即可，聚焦 epoch 元组解析）
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.7, 0.0])
        force_config = {
            "version": 1,
            "forces": [
                {
                    "name": "earth",
                    "type": "PointMassGravity",
                    "params": {"body": "EARTH", "mu": 398600.435507},
                },
            ],
        }
        result = propagate_orbit(
            initial_state=initial_state,
            epoch=[2025, 6, 21, 11, 0, 6.0],
            duration=3600.0,
            force_config=force_config,
            output_step=3600.0,
        )
        assert len(result) == 2

    def test_invalid_initial_state_shape(self):
        with pytest.raises(ValueError, match="initial_state"):
            propagate_orbit(
                initial_state=np.zeros(5),
                epoch="2025-06-21T11:00:00",
                duration=3600.0,
            )

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration"):
            propagate_orbit(
                initial_state=np.zeros(6),
                epoch="2025-06-21T11:00:00",
                duration=0.0,
            )

    def test_custom_force_config(self, spice_manager, reference_epoch):
        force_config = {
            "version": 1,
            "forces": [
                {
                    "name": "earth",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "EARTH", "mu": 398600.435507},
                },
                {
                    "name": "moon",
                    "type": "PointMassGravity",
                    "enabled": True,
                    "params": {"body": "MOON", "mu": 4902.800118},
                },
            ],
        }
        result = propagate_orbit(
            initial_state=np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
            epoch=reference_epoch,
            duration=86400.0,
            force_config=force_config,
            output_step=3600.0,
        )
        assert len(result) == 25
