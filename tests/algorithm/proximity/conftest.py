"""proximity 测试共享夹具。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.dro import _correct_dro
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.proximity import floquet_modes
from e2m2e.data.types.orbit import Orbit


@pytest.fixture(scope="session")
def dro21_raw() -> tuple[Orbit, CR3BP_Dynamics]:
    """2:1 地月 DRO 与地月动力学，session 级只读缓存。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    seed = _correct_dro(dynamics, 0.79188556619742, None)
    orbit = _correct_dro(dynamics, (1.0 - dynamics.system.mu) - 0.192, seed)
    return orbit, dynamics


@pytest.fixture
def dro21(dro21_raw: tuple[Orbit, CR3BP_Dynamics]) -> tuple[Orbit, CR3BP_Dynamics]:
    """2:1 地月 DRO 与地月动力学，函数级返回轨道副本。"""
    orbit, dynamics = dro21_raw
    return orbit.copy(), dynamics


@pytest.fixture(scope="session")
def modes21(dro21_raw: tuple[Orbit, CR3BP_Dynamics]):
    """2:1 DRO 的 Floquet 模态，session 级只读缓存。"""
    orbit, dynamics = dro21_raw
    return floquet_modes(orbit, dynamics)
