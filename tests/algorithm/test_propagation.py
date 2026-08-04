"""algorithm/propagation 模块测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from e2m2e.algorithm.propagation import _extract_bodies, propagate_orbit

# ---------------------------------------------------------------------------
# SPICE 与 kernel 可用性检测（与 tests/algorithms/normal_form/test_hamiltonian.py 一致）
# ---------------------------------------------------------------------------

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels"),
)


def _has_spice_kernels() -> bool:
    """检查 SPICE .tls + .bsp 都可用。"""
    if not os.path.isdir(_SPICE_KERNEL_DIR):
        return False
    has_tls = any(f.endswith(".tls") for f in os.listdir(_SPICE_KERNEL_DIR))
    has_bsp = any(f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR))
    return has_tls and has_bsp


_requires_spice = pytest.mark.skipif(
    not _has_spice_kernels(),
    reason="SPICE kernels (.tls + .bsp) not available",
)


@pytest.fixture
def _spice_loaded():
    """加载 leapseconds + 任一可用 SPK；不可用时跳过。"""
    if not _has_spice_kernels():
        pytest.skip("SPICE kernels not available")
    import spiceypy as spice

    for fname in ("naif0012.tls", "naif0011.tls"):
        path = os.path.join(_SPICE_KERNEL_DIR, fname)
        if os.path.exists(path):
            spice.furnsh(path)
    for fname in ("de440.bsp", "de440s.bsp", "de438.bsp", "de435.bsp", "de430.bsp"):
        path = os.path.join(_SPICE_KERNEL_DIR, fname)
        if os.path.exists(path):
            spice.furnsh(path)
            return path
    pytest.skip("No SPICE ephemeris kernel (.bsp) found")


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
    def test_default_three_body(self, _spice_loaded, reference_epoch):
        initial_state = np.array(
            [
                -6000.0,  # km
                -2500.0,
                0.0,
                5.0,  # km/s
                -4.5,
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

    def test_epoch_tuple(self, _spice_loaded):
        initial_state = np.zeros(6)
        result = propagate_orbit(
            initial_state=initial_state,
            epoch=[2025, 6, 21, 11, 0, 6.0],
            duration=3600.0,
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

    def test_custom_force_config(self, _spice_loaded, reference_epoch):
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
