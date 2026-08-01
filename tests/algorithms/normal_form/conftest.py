"""Normal-form 切片测试的共享 fixture。

提供与 qiao Global_File 约定一致的地月 CR3BP 系统（已设置特征尺度），
供 ``test_context``、``test_units`` 等模块复用。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.dynamics import CR3BP_System


@pytest.fixture
def earth_moon_cr3bp() -> CR3BP_System:
    """与 conftest.earth_moon_system 等价的 CR3BP 系统实例。

    本测试目录不复用根 conftest 的 fixture，避免与其他切片在 mu 默认值上
    耦合（本切片以 qiao 的 mu 为权威值）。
    """
    return CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")


@pytest.fixture
def earth_moon_system(earth_moon_cr3bp: CR3BP_System) -> CR3BP_System:
    """已设置特征尺度的地月 CR3BP 系统（与库内其它测试一致）。"""
    earth_moon_cr3bp.set_characteristic_scales(
        distance=384405.0,
        period=27.32 * 86400.0,
    )
    return earth_moon_cr3bp
