"""algorithm/coordinate spacetime_convert 统一入口测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest
from kernel_helpers import SPICE_KERNEL_DIR

from e2m2e.algorithm.coordinate import spacetime_convert
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.data


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


def _time_ephemeris_available() -> bool:
    """检查 de440t.bsp 是否存在（含时间星历的历表）。"""
    if not os.path.isdir(SPICE_KERNEL_DIR):
        return False
    return os.path.exists(os.path.join(SPICE_KERNEL_DIR, "de440t.bsp"))


@_requires_spice
class TestSpacetimeConvertSynodic:
    def test_j2000_to_synodic_and_back(self, spice_manager, earth_moon_system):
        et0_jd = 2459000.0
        j2000_state = np.array([380000.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        result_syn = spacetime_convert(
            "j2000_to_synodic",
            j2000_state,
            epoch=0.0,
            et0_jd=et0_jd,
        )
        assert result_syn["state"].shape == (6,)
        assert result_syn["transform_type"] == "j2000_to_synodic"
        assert result_syn["status"] is ConvergenceState.CONVERGED
        assert result_syn["cause"] is FailureCause.NONE
        assert result_syn["message"] == "任务完成"

        result_back = spacetime_convert(
            "synodic_to_j2000",
            result_syn["state"],
            epoch=0.0,
            et0_jd=et0_jd,
        )
        np.testing.assert_allclose(result_back["state"], j2000_state, rtol=1e-6)

    def test_unknown_transform_type(self):
        with pytest.raises(ValueError, match="未知"):
            spacetime_convert(
                "bad_type",
                np.zeros(6),
                epoch=0.0,
                et0_jd=2459000.0,
            )

    def test_state_must_be_1d(self):
        with pytest.raises(ValueError, match="state"):
            spacetime_convert(
                "j2000_to_synodic",
                np.zeros((2, 6)),
                epoch=0.0,
                et0_jd=2459000.0,
            )


_time_ephemeris_available_mark = pytest.mark.skipif(
    not _time_ephemeris_available(),
    reason="de440t.bsp not available",
)


@_time_ephemeris_available_mark
class TestSpacetimeConvertGcrsEbcrs:
    def test_gcrs_to_ebcrs_round_trip(self):
        ephemeris_path = os.path.join(SPICE_KERNEL_DIR, "de440t.bsp")
        position = np.array([7000.0, 0.0, 0.0])
        jd_tt = 2459000.0

        result_eb = spacetime_convert(
            "gcrs_to_ebcrs",
            position,
            epoch=jd_tt,
            ephemeris_path=ephemeris_path,
        )
        assert result_eb["state"].shape == (3,)

        result_gc = spacetime_convert(
            "ebcrs_to_gcrs",
            result_eb["state"],
            epoch=result_eb["time"],
            ephemeris_path=ephemeris_path,
        )
        np.testing.assert_allclose(result_gc["state"], position, rtol=1e-6)

    def test_missing_ephemeris_path(self):
        with pytest.raises(ValueError, match="ephemeris_path"):
            spacetime_convert(
                "gcrs_to_ebcrs",
                np.zeros(3),
                epoch=2459000.0,
            )

    def test_gcrs_requires_3d_position(self):
        with pytest.raises(ValueError, match="3 维"):
            spacetime_convert(
                "gcrs_to_ebcrs",
                np.zeros(6),
                epoch=2459000.0,
                ephemeris_path="dummy.bsp",
            )
