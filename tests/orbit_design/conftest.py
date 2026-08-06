"""tests/orbit_design 共享 fixture。

地月 CR3BP 系统/动力学的 session 级缓存，供初猜及后续阶段（correction/
continuation/multiple_shooting/ephemeris）测试复用。

注意：本目录的 ``earth_moon_system``/``earth_moon_dynamics`` 覆盖
``tests/conftest.py`` 的函数级同名 fixture——orbit_design 测试统一采用
更精确的地月质量比 μ=0.01215058560962404 与默认特征尺度（地月距
384405 km、周期 27.32 d），与 lissajous/axial 初猜的历史取值一致。

阶段 2 起将在此追加 corrected_orbit 等 session 缓存 fixture。
"""

import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System

EARTH_MOON_MU = 0.01215058560962404


@pytest.fixture(scope="session")
def earth_moon_system() -> CR3BP_System:
    """标准地月 CR3BP 系统（session 级，只读复用）。"""
    return CR3BP_System(mu=EARTH_MOON_MU, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture(scope="session")
def earth_moon_dynamics(earth_moon_system: CR3BP_System) -> CR3BP_Dynamics:
    """标准地月 CR3BP 动力学（session 级，只读复用）。"""
    return CR3BP_Dynamics(system=earth_moon_system)
