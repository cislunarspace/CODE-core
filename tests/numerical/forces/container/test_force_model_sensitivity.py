"""力模型参数敏感列（变分方程对 Cr/Cd 的一阶偏导）测试。

ASSIST（Holman et al. 2023）式参数敏感列：在 STM 增广系统后追加
``S_p = ∂[r,v]/∂p``，满足 ``Ṡ_p = A·S_p + [0; ∂a/∂p]``，初值零。

正确性从两个独立方向验证（ASSIST §4.2 的方法论）：

1. **跨积分器一致性**：RK（PD78）与 IAS15 两个独立积分器积分同一套
   变分方程，敏感列应一致到 ~1e-8。两条实现共享的只有 ``Ṡ_p`` 右端项，
   积分路径完全独立。
2. **shadow-particle 有限差分**：扰动参数 ε 重传播，``Δstate/ε`` 与解析
   敏感列比较。短弧段（2 h）上 FD 干净，容差取 1e-2——足以捕捉漏项、
   错号这类 O(1) 错误；长弧段的 FD 被星历力模型的轨迹噪声地板
   （~1e-7 km/天量级）主导，不适合做紧断言。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import (
    ForceModel,
    GravityField,
    PointMassGravity,
    SolarRadiationPressure,
    ThirdBodyGravity,
)
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere
from e2m2e.algorithm.forces.drag import DragModel
from tests.numerical.forces.conftest import (
    EARTH_MU,
    EARTH_RE,
    FakeSystem,
    keplerian_to_cartesian,
)

pytestmark = pytest.mark.force

EPOCH_UTC = "2025-06-21T11:00:06"
CR_NOMINAL = 1.5
CD_NOMINAL = 2.2


def _sens_column(system, y0, t_span, param_label, integrator):
    """一次带敏感列的传播，返回末点 ``∂[r,v]/∂p``。"""
    if param_label == "srp_cr":
        forces = [
            PointMassGravity("EARTH"),
            ThirdBodyGravity("MOON"),
            ThirdBodyGravity("SUN"),
            SolarRadiationPressure(area=20.0, mass=1000.0, cr=CR_NOMINAL),
        ]
    else:
        forces = [
            GravityField(body="EARTH", degree=2, order=0),
            DragModel(
                atmosphere=ExponentialAtmosphere(),
                area=10.0,
                mass=1000.0,
                cd=CD_NOMINAL,
            ),
        ]
    fm = ForceModel(system, forces=forces)
    fm.rtol = 1e-13
    fm.max_step = 600.0
    res = fm.propagate(
        y0,
        t_span,
        t_eval=[t_span[0], t_span[1]],
        with_stm=True,
        sens_params=[param_label],
        integrator=integrator,
    )
    assert res["sens_params"] == [param_label]
    return res["sensitivity"][-1, :, 0]


def _shadow_fd(system, y0, t_span, param_label, scale_eps, integrator):
    """扰动参数重传播（state-only），返回 ``Δstate/Δp`` 的 FD 敏感列。"""
    nominal = {"srp_cr": CR_NOMINAL, "drag_cd": CD_NOMINAL}[param_label]
    states = []
    for scale in (1.0, 1.0 + scale_eps):
        if param_label == "srp_cr":
            forces = [
                PointMassGravity("EARTH"),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
                SolarRadiationPressure(area=20.0, mass=1000.0, cr=CR_NOMINAL * scale),
            ]
        else:
            forces = [
                GravityField(body="EARTH", degree=2, order=0),
                DragModel(
                    atmosphere=ExponentialAtmosphere(),
                    area=10.0,
                    mass=1000.0,
                    cd=CD_NOMINAL * scale,
                ),
            ]
        fm = ForceModel(system, forces=forces)
        fm.rtol = 1e-13
        fm.max_step = 600.0
        r = fm.propagate(y0, t_span, t_eval=[t_span[0], t_span[1]], integrator=integrator)
        states.append(r["states"][-1])
    return (states[1] - states[0]) / (scale_eps * nominal)


def _cislunar_case(system):
    et0 = system.spice.utc_to_et(EPOCH_UTC)
    y0 = np.array([384400.0 * 0.6, 0.0, 0.0, 0.0, 0.8, 0.0])
    return y0, et0


def _leo_case(system):
    et0 = system.spice.utc_to_et(EPOCH_UTC)
    y0 = keplerian_to_cartesian(EARTH_RE + 300.0, 0.001, 51.6, 0.0, 0.0, 0.0, EARTH_MU)
    return y0, et0


@pytest.mark.spice
def test_srp_cr_sensitivity_cross_integrator(earth_icrf_system):
    """SRP Cr 敏感列：RK 与 IAS15 两个独立积分器一致。"""
    y0, et0 = _cislunar_case(earth_icrf_system)
    t_span = (et0, et0 + 86400.0)
    s_rk = _sens_column(earth_icrf_system, y0, t_span, "srp_cr", "rk")
    s_ias = _sens_column(earth_icrf_system, y0, t_span, "srp_cr", "ias15")
    rel = np.linalg.norm(s_rk - s_ias) / np.linalg.norm(s_ias)
    assert rel < 1e-8, f"两积分器 Cr 敏感列相对偏差 {rel:.3e}"


@pytest.mark.spice
def test_drag_cd_sensitivity_cross_integrator(earth_icrf_system):
    """阻力 Cd 敏感列：RK 与 IAS15 两个独立积分器一致。

    阻力 + 指数大气在 300 km 高度的敏感性随轨道衰减快速放大，一天弧段
    上两积分器在 1e-13 容差下的轨迹噪声被同步放大，阈值取 1e-4。
    """
    y0, et0 = _leo_case(earth_icrf_system)
    t_span = (et0, et0 + 86400.0)
    s_rk = _sens_column(earth_icrf_system, y0, t_span, "drag_cd", "rk")
    s_ias = _sens_column(earth_icrf_system, y0, t_span, "drag_cd", "ias15")
    rel = np.linalg.norm(s_rk - s_ias) / np.linalg.norm(s_ias)
    assert rel < 1e-4, f"两积分器 Cd 敏感列相对偏差 {rel:.3e}"


@pytest.mark.spice
@pytest.mark.parametrize("integrator", ["rk", "ias15"])
def test_srp_cr_sensitivity_vs_shadow_particle(earth_icrf_system, integrator):
    """SRP Cr 敏感列与 shadow-particle FD 一致（短弧段）。

    FD 步长取 ε=1e-2·Cr：Δx 信号 ~ε·S 须压过轨迹噪声（~1e-8 km），
    ε 再大则轨道对参数耦合的二阶项进来；1e-2 时 FD 自身误差 ~1%。
    """
    y0, et0 = _cislunar_case(earth_icrf_system)
    t_span = (et0, et0 + 7200.0)
    s_analytic = _sens_column(earth_icrf_system, y0, t_span, "srp_cr", integrator)
    s_fd = _shadow_fd(earth_icrf_system, y0, t_span, "srp_cr", 1e-2, integrator)
    rel = np.linalg.norm(s_analytic - s_fd) / np.linalg.norm(s_fd)
    assert rel < 5e-2, f"[{integrator}] Cr 敏感列与 FD 相对偏差 {rel:.3e}"


@pytest.mark.spice
@pytest.mark.parametrize("integrator", ["rk", "ias15"])
def test_drag_cd_sensitivity_vs_shadow_particle(earth_icrf_system, integrator):
    """阻力 Cd 敏感列与 shadow-particle FD 一致（短弧段）。"""
    y0, et0 = _leo_case(earth_icrf_system)
    t_span = (et0, et0 + 7200.0)
    s_analytic = _sens_column(earth_icrf_system, y0, t_span, "drag_cd", integrator)
    s_fd = _shadow_fd(earth_icrf_system, y0, t_span, "drag_cd", 1e-2, integrator)
    rel = np.linalg.norm(s_analytic - s_fd) / np.linalg.norm(s_fd)
    assert rel < 5e-2, f"[{integrator}] Cd 敏感列与 FD 相对偏差 {rel:.3e}"


class TestSensParamsContract:
    """参数解析的显式报错路径（无需 SPICE）。"""

    def _fm(self) -> ForceModel:
        return ForceModel(FakeSystem(), forces=[PointMassGravity("EARTH", mu=EARTH_MU)])

    def test_sens_requires_with_stm(self):
        with pytest.raises(ValueError, match="with_stm"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                sens_params=["srp_cr"],
            )

    def test_unknown_label_rejected(self):
        with pytest.raises(ValueError, match="未知敏感参数"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                with_stm=True,
                sens_params=["mu"],
            )

    def test_missing_force_rejected(self):
        """只有点质量力时求 srp_cr：显式报错而非静默忽略。"""
        with pytest.raises(ValueError, match="SolarRadiationPressure"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                with_stm=True,
                sens_params=["srp_cr"],
            )
