"""
pytest configuration and shared fixtures for e2m2e tests
"""

import pytest
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, CoordinateTransformation, Orbit


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon CR3BP system"""
    return CR3BP_System.from_known_system("earth_moon")


@pytest.fixture
def sun_earth_system():
    """Create a Sun-Earth CR3BP system"""
    return CR3BP_System.from_known_system("sun_earth")


@pytest.fixture
def sun_jupiter_system():
    """Create a Sun-Jupiter CR3BP system"""
    return CR3BP_System.from_known_system("sun_jupiter")


@pytest.fixture
def earth_moon_dynamics(earth_moon_system):
    """Create Earth-Moon CR3BP dynamics"""
    return CR3BP_Dynamics(system=earth_moon_system)


@pytest.fixture
def earth_moon_coordinate(earth_moon_system):
    """Create Earth-Moon coordinate transformation"""
    return CoordinateTransformation(system=earth_moon_system)


@pytest.fixture
def sample_state():
    """Sample state vector near L1"""
    return np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])


@pytest.fixture
def sample_orbit():
    """Create a sample orbit for testing"""
    # Create simple periodic-like orbit data
    t = np.linspace(0, 1, 50)
    # Simple circular-ish motion in rotating frame
    x = 0.8 + 0.1 * np.cos(2 * np.pi * t)
    y = 0.1 * np.sin(2 * np.pi * t)
    z = np.zeros_like(t)
    vx = -0.1 * 2 * np.pi * np.sin(2 * np.pi * t)
    vy = 0.1 * 2 * np.pi * np.cos(2 * np.pi * t)
    vz = np.zeros_like(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    return Orbit(states=states, times=t)


@pytest.fixture
def initialized_system(earth_moon_system):
    """Earth-Moon system with characteristic scales set"""
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
    return system
