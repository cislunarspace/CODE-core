"""
pytest configuration and shared fixtures for e2m2e tests
"""

import os

import numpy as np
import pytest

from e2m2e.core import CoordinateTransformation, CR3BP_Dynamics, CR3BP_System, Orbit


def pytest_configure(config):
    config.addinivalue_line("markers", "spice: marks tests requiring SPICE kernel files")


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
    return earth_moon_system


# =============================================================================
# Ephemeris model fixtures (需求: DRO CR3BP→星历模型转换)
# =============================================================================
SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"),
)

# 地月系统物理参数
MU = 1.21506683e-2
DU = 3.84405e5  # km
TU_SECONDS = 4.34811305 * 86400  # 秒
VU = DU / TU_SECONDS  # km/s


@pytest.fixture
def spice_kernel_dir():
    """返回 SPICE 内核文件所在目录，不存在或无内核文件则跳过"""
    if not os.path.isdir(SPICE_KERNEL_DIR):
        pytest.skip("SPICE kernel directory not found, set SPICE_KERNEL_DIR")
    bsp_files = [f for f in os.listdir(SPICE_KERNEL_DIR) if f.endswith(".bsp")]
    if not bsp_files:
        pytest.skip("No .bsp kernel files found in SPICE kernel directory")
    return SPICE_KERNEL_DIR


@pytest.fixture
def spice_kernel_path():
    """返回DE440内核文件路径，不存在则跳过"""
    kernel_file = os.path.join(SPICE_KERNEL_DIR, "de440.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de440s.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de438.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de435.bsp")
    if not os.path.exists(kernel_file):
        pytest.skip("DE440/DE438/DE435 SPICE kernel not found, set SPICE_KERNEL_DIR")
    return kernel_file


@pytest.fixture
def reference_epoch():
    """参考历元: 2025-06-21 11:00:06 UTC (J2000后的ET秒数)"""
    return "2025-06-21T11:00:06"


@pytest.fixture
def dro_31_state():
    """3:1 DRO初始状态（CR3BP旋转系，无量纲）"""
    return np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])


@pytest.fixture
def dro_31_period():
    """3:1 DRO周期（无量纲TU）"""
    return 2.095
