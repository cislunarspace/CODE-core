"""相对论修正力模型的物理规律验证。

相对论修正导致 GPS/LEO 轨道一天传播的可观测终端位置漂移，量级应符合
Schwarzschild 修正的物理预期。配置 round-trip 见 ``config/test_force_config.py``；
缓存一致性见 ``tests/numerical/integrators/bindings/test_ephem_cache.py``。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, GravityField, RelativisticCorrection
from tests.numerical.forces.conftest import EARTH_RE, keplerian_to_cartesian

pytestmark = [pytest.mark.force, pytest.mark.spice]


# 地球角动量矢量近似值（km²/s），用于 Lense-Thirring 量级参考：
# 地球自转角动量 = I·ω ≈ 8.0e37 kg·m²/s 换算到 km²/s 量级（约 1e3），
# 仅要求量级正确（测试断言的是修正量级上下界，不是精确值）。
_EARTH_ANGULAR_MOMENTUM = np.array([0.0, 0.0, 1.18e3])


@pytest.mark.spice
def test_gps_relativistic_position_difference_magnitude(earth_icrf_system):
    """GPS 轨道 1 天传播，相对论修正导致可观测的终端位置漂移。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    # GPS 类轨道
    a0 = 26560.0
    y0 = keplerian_to_cartesian(a0, 0.0, 55.0, 0.0, 0.0, 0.0, mu)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.array([et0, et0 + 86400.0])

    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm_without = ForceModel(system, forces=[gravity])
    result_without = fm_without.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    relcorr = RelativisticCorrection(
        central_body="EARTH",
        angular_momentum_vector=_EARTH_ANGULAR_MOMENTUM,
    )
    fm_with = ForceModel(system, forces=[gravity, relcorr])
    result_with = fm_with.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    pos_diff = np.linalg.norm(result_with["states"][-1, :3] - result_without["states"][-1, :3])
    # 物理量级：GPS 轨道 Schwarzschild 修正约 0.3 mm/天（3e-7 km）。
    # 下界收到 1/10 物理量级，防止数量级漂移（远低于物理 3 个数量级 → 收紧到 0.1×）。
    # 上界 0.01 km（10 cm/天）覆盖 Lense-Thirring / de Sitter 等次级项贡献。
    assert 3e-8 <= pos_diff <= 0.01, f"GPS 1-day position diff = {pos_diff:.6e} km"


@pytest.mark.spice
def test_leo_relativistic_position_difference_magnitude(earth_icrf_system):
    """LEO 轨道 1 天传播，相对论修正导致可观测的终端位置漂移。"""
    system = earth_icrf_system
    mu = system.gravitational_parameter("EARTH")

    r_earth = EARTH_RE
    a0 = r_earth + 400.0
    y0 = keplerian_to_cartesian(a0, 0.0, 51.6, 0.0, 0.0, 0.0, mu)

    et0 = system.spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.array([et0, et0 + 86400.0])

    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm_without = ForceModel(system, forces=[gravity])
    result_without = fm_without.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    relcorr = RelativisticCorrection(
        central_body="EARTH",
        angular_momentum_vector=_EARTH_ANGULAR_MOMENTUM,
    )
    fm_with = ForceModel(system, forces=[gravity, relcorr])
    result_with = fm_with.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    pos_diff = np.linalg.norm(result_with["states"][-1, :3] - result_without["states"][-1, :3])
    # 物理量级：LEO Schwarzschild 修正约 2.5 mm/天（2.5e-6 km）。
    # 下界收到 1/10 物理量级，防止数量级漂移（原 1e-3 km 比物理宽 2.7 个数量级 → 收紧到 0.1×）。
    # 上界 0.01 km（10 cm/天）覆盖 Lense-Thirring / de Sitter 等次级项贡献。
    assert 2.5e-7 <= pos_diff <= 0.01, f"LEO 1-day position diff = {pos_diff:.6e} km"
