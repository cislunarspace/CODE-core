"""霍曼转移验收测试（测法 C：实测远地点反算 Δv2）。

验证双脉冲转移总 Δv 与末态圆轨道。
"""

import numpy as np
import pytest

from e2m2e.core.forces import ForceModel
from e2m2e.core.forces.thrust import ImpulsiveBurn


class _FakeSystem:
    """仅用于传播测试的最小 System 桩。"""

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return 398600.4415


def test_hohmann_transfer_two_burns(point_mass_force):
    """霍曼转移：Δv1@近地点 → 半周期 → 实测远地点反算 Δv2 → 总 Δv<1% + 末态圆。"""
    mu = 398600.4415
    r1, r2 = 6678.0, 42164.0  # LEO 300km → GEO
    v_circ1 = np.sqrt(mu / r1)
    v_circ2 = np.sqrt(mu / r2)
    a_transfer = (r1 + r2) / 2.0
    v_perigee = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_apogee_theory = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))
    half_period = np.pi * np.sqrt(a_transfer**3 / mu)

    dv1 = v_perigee - v_circ1  # 理论 Δv1（+y，prograde）

    # 初始：r1 圆轨道近地点 [r1,0,0]，速度 [0, v_circ1, 0]
    y0 = np.array([r1, 0.0, 0.0, 0.0, v_circ1, 0.0])

    fm = ForceModel(_FakeSystem(), forces=[point_mass_force])

    # 第一段：施加 Δv1 @ t0，传半周期到远地点
    burn1 = ImpulsiveBurn(epoch=0.0, delta_v=np.array([0.0, dv1, 0.0]))
    leg1 = fm.propagate_maneuvers(y0, (0.0, half_period), burns=[burn1])

    apogee = leg1["states"][-1]
    r_apogee = np.linalg.norm(apogee[:3])
    v_apogee_actual = apogee[3:6]

    # 远地点位置 ≈ r2
    assert r_apogee == pytest.approx(r2, rel=1e-4)

    # 反算 Δv2：远地点在 -x 侧，prograde 圆速度为 [0, -v_circ2, 0]
    v_target = np.array([0.0, -v_circ2, 0.0])
    dv2_needed = v_target - v_apogee_actual
    dv2_needed_mag = np.linalg.norm(dv2_needed)

    # 总 Δv 误差 < 1%（测法 C 核心）
    dv_total_theory = dv1 + (v_circ2 - v_apogee_theory)
    dv_total = dv1 + dv2_needed_mag
    assert abs(dv_total - dv_total_theory) / dv_total_theory < 0.01

    # 第二段：施加 Δv2 @ 远地点，再传 1/4 GEO 周期，验证末态圆轨道
    burn2 = ImpulsiveBurn(epoch=half_period, delta_v=dv2_needed)
    geo_quarter = 0.5 * np.pi * np.sqrt(r2**3 / mu)
    leg2 = fm.propagate_maneuvers(y0, (0.0, half_period + geo_quarter), burns=[burn1, burn2])
    final = leg2["states"][-1]
    r_final = np.linalg.norm(final[:3])
    v_final = np.linalg.norm(final[3:6])
    energy_final = 0.5 * v_final**2 - mu / r_final

    assert r_final == pytest.approx(r2, rel=1e-3)
    assert v_final == pytest.approx(v_circ2, rel=1e-3)
    assert energy_final == pytest.approx(-mu / (2.0 * r2), rel=1e-3)
