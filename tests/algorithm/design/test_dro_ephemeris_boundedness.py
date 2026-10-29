"""DRO 长期星历有界性告警的阈值行为。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.design.design_orbit import _warn_if_dro_ephemeris_unbounded
from e2m2e.data.templates import CHAR_LENGTH_KM
from e2m2e.data.types.trajectory import EphemerisTable

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


def _warnings(orbit_type: str, amplitude_km: float, max_moon_dist_km: float):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_dro_ephemeris_unbounded(
            orbit_type,
            amplitude_km,
            _table(max_moon_dist_km),
            CHAR_LENGTH_KM,
            365.25,
        )
    return caught


def test_dro_escape_warns():
    caught = _warnings("DRO", 150000.0, 1.6595e7)
    assert len(caught) == 1
    assert "月心距离最大" in str(caught[0].message)


def test_dro_bounded_does_not_warn():
    assert _warnings("DRO", 200000.0, 313089.1) == []


def test_non_dro_does_not_warn():
    assert _warnings("NRHO", 200000.0, 1.6595e7) == []
