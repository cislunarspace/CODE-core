"""DragModel 的 Rust 端到端传播验证（需 SPICE 内核）。

Python 单点 ``compute_acceleration`` 已删除；阻力物理行为由
Rust ``propagate_compiled`` 承载。本文件用真实 Rust 传播验证：

- 阻力导致轨道能量与半长轴下降；
- 传播结果状态有限、形状正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PointMassGravity
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere, NRLMSISE00Atmosphere
from e2m2e.algorithm.forces.drag import DragModel
from tests.numerical.forces.conftest import EARTH_MU, EARTH_RE

pytestmark = [pytest.mark.force, pytest.mark.spice]


def _leo_400km_state():
    """400 km 圆轨道状态（ITRF 系内阻力显著）。"""
    r = EARTH_RE + 400.0
    v = np.sqrt(EARTH_MU / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def _semi_major_axis(state):
    r_norm = np.linalg.norm(state[:3])
    v_norm = np.linalg.norm(state[3:6])
    return -EARTH_MU / (2.0 * (0.5 * v_norm**2 - EARTH_MU / r_norm))


def test_drag_propagation_decreases_orbital_energy(earth_icrf_system):
    """Rust 传播的 drag 使 LEO 轨道能量与半长轴下降。"""
    system = earth_icrf_system
    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")

    y0 = _leo_400km_state()

    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)
    gravity = PointMassGravity(body="EARTH", mu=EARTH_MU)
    fm = ForceModel(system, forces=[gravity, drag])
    fm.rtol = 1e-10
    fm.atol = 1e-10

    result = fm.propagate(y0, (et0, et0 + 3600.0), max_steps=200_000)

    states = result["states"]
    assert states.shape[1] == 6
    assert np.all(np.isfinite(states))

    def energy(state):
        r_norm = np.linalg.norm(state[:3])
        v_norm = np.linalg.norm(state[3:6])
        return 0.5 * v_norm**2 - EARTH_MU / r_norm

    energies = np.array([energy(s) for s in states])
    assert energies[-1] < energies[0], "阻力应使比机械能下降"

    a0 = _semi_major_axis(states[0])
    a1 = _semi_major_axis(states[-1])
    assert a1 < a0, "阻力应使半长轴下降"


def test_nrlmsise00_drag_propagation_decreases_semi_major_axis(earth_icrf_system):
    """NRLMSISE-00 大气经 Rust ``drag_nrlmsise00`` 分支驱动传播并降轨。

    同一轨道与相同时长下与指数大气的结果应有可分辨差异（两个模型在 400 km
    的密度不同），否则说明力元组未真正切换到 NRLMSISE-00。
    """
    system = earth_icrf_system
    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")

    y0 = _leo_400km_state()
    gravity = PointMassGravity(body="EARTH", mu=EARTH_MU)

    def propagate(atmosphere):
        drag = DragModel(atmosphere=atmosphere, area=10.0, mass=1000.0, cd=2.2)
        fm = ForceModel(system, forces=[gravity, drag])
        fm.rtol = 1e-10
        fm.atol = 1e-10
        return fm.propagate(y0, (et0, et0 + 3600.0), max_steps=200_000)["states"]

    states = propagate(NRLMSISE00Atmosphere())
    assert states.shape[1] == 6
    assert np.all(np.isfinite(states))

    assert _semi_major_axis(states[-1]) < _semi_major_axis(states[0]), "阻力应使半长轴下降"

    exp_states = propagate(ExponentialAtmosphere())
    assert not np.allclose(states[-1], exp_states[-1], rtol=1e-9, atol=1e-9), (
        "NRLMSISE-00 与指数大气的终态应可分辨（否则力元组未切换大气模型）"
    )
