"""pytest 配置与共享 fixture。

提供地月/日地/木日 CR3BP 系统、
SPICE 内核 fixture 与参考历元。
"""

import os

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


def pytest_configure(config):
    config.addinivalue_line("markers", "spice: marks tests requiring SPICE kernel files")


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon CR3BP system"""
    return CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture
def sun_earth_system():
    """Create a Sun-Earth CR3BP system"""
    return CR3BP_System(mu=3.0039e-06, primary="Sun", secondary="Earth")._with_default_scales()


@pytest.fixture
def sun_jupiter_system():
    """Create a Sun-Jupiter CR3BP system"""
    return CR3BP_System(mu=0.0009535, primary="Sun", secondary="Jupiter")._with_default_scales()


@pytest.fixture
def earth_moon_dynamics(earth_moon_system):
    """Create Earth-Moon CR3BP dynamics"""
    return CR3BP_Dynamics(system=earth_moon_system)


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

# 定义 body-fixed 帧所需的 SPICE 内核文件名(issue #187):
# 地球 ITRF93 需要 earth_latest_high_prec.bpc;text PCK 与月球 MOON_PA 需要
# pck00010.tpc、SPICELunaCurrentKernel.bpc、SPICELunaFrameKernel.tf。
BODY_FIXED_KERNELS = [
    "earth_latest_high_prec.bpc",
    "pck00010.tpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
]


def load_body_fixed_kernels(spice) -> list[str]:
    """向 ``spice``(SPICEManager)furnsh 所有可用的 body-fixed 内核。

    加载 ``kernels/`` 下定义 ITRF93 / MOON_PA 帧所需的 BPC/TPC/TF 内核
    (见 :data:`BODY_FIXED_KERNELS`),使 ``GravityField`` 的 body-fixed 坐标轴
    (ITRFSpiceAxes)能解析 SPICE 旋转。文件不存在时静默跳过该项。

    Args:
        spice: 已初始化的 :class:`~e2m2e.core.spice.SPICEManager`。

    Returns:
        实际 furnsh 的内核绝对路径列表;调用方应在 teardown 时对其逆序 unload。
    """
    loaded: list[str] = []
    if not os.path.isdir(SPICE_KERNEL_DIR):
        return loaded
    for name in BODY_FIXED_KERNELS:
        path = os.path.join(SPICE_KERNEL_DIR, name)
        if os.path.exists(path):
            spice.load_kernel(path)
            loaded.append(path)
    return loaded


def unload_kernels(spice, paths: list[str]) -> None:
    """逆序卸载 ``load_body_fixed_kernels`` 之类返回的内核路径列表。"""
    for path in reversed(paths):
        spice.unload_kernel(path)


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


# =============================================================================
# SPICE ephemeris fixtures
#
# Six test files previously rebuilt this chain locally:
#   tests/algorithms/test_multiple_shooting.py
#   tests/algorithms/test_dro_ephemeris_correction.py
#   tests/algorithms/test_dc_via_propagate.py
#   tests/algorithms/test_patch_point_utils.py
#   tests/core/coordinate/test_synodic_j2000.py
#   tests/core/dynamics/test_ephemeris_dynamics.py
#
# The previous test_dc_via_propagate._make_eph_dynamics helper had a kernel
# leak (mgr.unload_kernel never called). These fixtures fix that by going
# through proper yield-teardown.
# =============================================================================


@pytest.fixture
def spice_manager(spice_kernel_path):
    """SPICEManager with DE440/DE438/DE435 kernel loaded; auto-unload after test."""
    from e2m2e.core.spice import SPICEManager

    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)
    yield mgr
    mgr.unload_kernel(spice_kernel_path)


@pytest.fixture
def spice_eph_system(spice_manager):
    """Earth-Moon-Sun ephemeris system in J2000, with origin at Earth."""
    from e2m2e.core.ephemeris_system import EphemerisSystem
    from e2m2e.mbse.data.enums import ReferenceFrame

    return EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=spice_manager,
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )


@pytest.fixture
def spice_eph_dynamics(spice_eph_system):
    """Ephemeris N-body dynamics with relaxed rtol/atol/max_step for fast tests."""
    from e2m2e.core.ephemeris_dynamics import EphemerisDynamics

    d = EphemerisDynamics(system=spice_eph_system)
    # 这些宽松参数让测试中的星历传播比生产快 ~10×
    # 生产代码（_run_propagate 等）使用更严格的 1e-12。
    d.rtol = 1e-10
    d.atol = 1e-10
    d.max_step = 600.0
    return d


@pytest.fixture
def spice_syn_j2000(earth_moon_system, spice_manager):
    """Synodic ↔ J2000 coordinate transformer wired to the standard CR3BP system.

    基于 ``CoordinateSystem`` 的 ``SynodicJ2000System`` 实现，接口包括
    ``synodic_to_j2000``、``j2000_to_synodic``、``batch_synodic_to_j2000``、
    ``batch_j2000_to_synodic``，以及 ``cr3bp_system``、``spice`` 属性。
    """
    from e2m2e.core.synodic_j2000 import SynodicJ2000System

    return SynodicJ2000System(
        cr3bp_system=earth_moon_system,
        spice=spice_manager,
    )
