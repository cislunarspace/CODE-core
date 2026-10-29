"""DRO 长期星历有界性判定与结果状态的阈值行为。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.design.design_orbit import (
    _design_stages,
    _dro_ephemeris_divergence_km,
    _ephemeris_result_status,
)
from e2m2e.data.templates import CHAR_LENGTH_KM
from e2m2e.data.types.trajectory import EphemerisTable
from e2m2e.status import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface


def _table(max_moon_dist_km: float) -> EphemerisTable:
    """会合系地心归一，月球在 x=1。"""
    count = 3
    synodic = np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    synodic[1, 0] += max_moon_dist_km / CHAR_LENGTH_KM
    zeros = np.zeros(count)
    return EphemerisTable(
        year=np.zeros(count, dtype=int),
        month=np.zeros(count, dtype=int),
        day=np.zeros(count, dtype=int),
        hour=np.zeros(count, dtype=int),
        minute=np.zeros(count, dtype=int),
        second=zeros,
        position_km=np.zeros((count, 3)),
        velocity_mps=np.zeros((count, 3)),
        synodic_position=synodic,
    )


def _status(orbit_type: str, amplitude_km: float, max_moon_dist_km: float):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _ephemeris_result_status(
            orbit_type,
            amplitude_km,
            _table(max_moon_dist_km),
            CHAR_LENGTH_KM,
            365.25,
        )
    return result, caught


def test_dro_escape_sets_diverged_status():
    result, caught = _status("DRO", 150000.0, 1.6595e7)
    assert result[0] is ConvergenceState.DIVERGED
    assert result[1] is FailureCause.DIVERGENCE_DETECTED
    assert "月心距离最大" in result[2]
    assert result[3] is ConvergenceState.DIVERGED
    assert len(caught) == 1


def test_dro_bounded_keeps_converged_status():
    result, caught = _status("DRO", 200000.0, 313089.1)
    assert result[0] is ConvergenceState.CONVERGED
    assert result[1] is FailureCause.NONE
    assert result[3] is ConvergenceState.CONVERGED
    assert caught == []


def test_non_dro_keeps_converged_status():
    result, caught = _status("NRHO", 200000.0, 1.6595e7)
    assert result[0] is ConvergenceState.CONVERGED
    assert caught == []


def test_diverged_propagation_stage():
    stages = _design_stages(propagation_status=ConvergenceState.DIVERGED)
    assert stages[-1].name == "propagation"
    assert stages[-1].result_status is ConvergenceState.DIVERGED


def test_divergence_threshold_uses_design_amplitude():
    assert _dro_ephemeris_divergence_km(
        "DRO", 150000.0, _table(400000.0), CHAR_LENGTH_KM
    ) is None
    assert _dro_ephemeris_divergence_km(
        "DRO", 150000.0, _table(500000.0), CHAR_LENGTH_KM
    ) == pytest.approx(500000.0)
