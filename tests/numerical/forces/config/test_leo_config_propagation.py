"""LEO 配置构建 vs 手动构建传播一致性测试（需 SPICE 内核）。"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import DragModel, ForceModel, GravityField
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere
from tests.numerical.forces.conftest import EARTH_MU, EARTH_RE, keplerian_to_cartesian

pytestmark = pytest.mark.force


@pytest.mark.spice
def test_leo_config_vs_manual_propagation_match(earth_icrf_system):
    """to_config/from_config 重建的力模型传播轨迹与手动构建一致（< 1e-12）。"""
    system = earth_icrf_system

    # 手动构建 LEO 力模型（J2 + 阻力）
    fm_manual = ForceModel(system)
    fm_manual.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
    fm_manual.add_force(
        DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0, cd=2.2),
        name="drag",
    )

    # 经 config 重建
    config = ForceModel.to_config(fm_manual)
    fm_config = ForceModel.from_config(config, system)

    # 同一初始状态与时间区间传播
    a0 = EARTH_RE + 400.0
    y0 = keplerian_to_cartesian(a0, 0.001, 51.6, 0.0, 0.0, 0.0, EARTH_MU)
    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 600.0)
    t_eval = np.linspace(et0, et0 + 600.0, 50)

    result_manual = fm_manual.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)
    result_config = fm_config.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    np.testing.assert_allclose(result_manual["states"], result_config["states"], atol=1e-12)
