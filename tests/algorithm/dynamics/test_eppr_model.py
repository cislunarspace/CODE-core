"""EPPR 帧量与动力学：解析退化、惯性参照等价与 Jacobi 拒绝。

覆盖三层证据：
1. 圆形合成星历 → EPPR EOM 精确退化为 CR3BP EOM；
2. 椭圆合成星历（含倾角）→ EPPR 传播经解析逆变换到惯性，与惯性二体参照等价；
3. 真实星历 → EPPR 传播与惯性二体参照等价；与 ``EphemerisDynamics`` 的差
   落在帧锚定模型误差尺度内（~½ a_sun T²）。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from e2m2e.algorithm.coordinate import EPPRJ2000System
from e2m2e.algorithm.coordinate.eppr_frame import EPPRFrameModel, EPPRFrameState
from e2m2e.algorithm.dynamics import (
    CR3BP_Dynamics,
    EphemerisDynamics,
    EphemerisSystem,
    EPPR_Dynamics,
    EPPRSystem,
)
from e2m2e.data.constants import SECONDS_PER_DAY, Datum

pytestmark = pytest.mark.theory

#: 与 CR3BP 无量纲约定一致的物理尺度（EPPR 隐含的 GM_总 = lstar³/t_c²）。
MU = Datum.DE421.mu
LSTAR = Datum.DE421.char_length_km
TIME_UNIT = Datum.DE421.char_time_s
GM_EFF = LSTAR**3 / TIME_UNIT**2


def _tilt_matrix(inclination: float) -> np.ndarray:
    """绕 x 轴倾斜 ``inclination`` 的旋转矩阵。"""
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]])


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    """牛顿迭代解开普勒方程 ``E - e·sinE = M``。"""
    mean = float(np.mod(mean_anomaly, 2.0 * np.pi))
    eccentric_anomaly = mean if eccentricity < 0.8 else np.pi
    for _ in range(100):
        delta = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean) / (
            1.0 - eccentricity * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= delta
        if abs(delta) < 1e-15:
            break
    return eccentric_anomaly


class SyntheticEPPRFrameModel(EPPRFrameModel):
    """解析质心二体星历的帧量模型（不经 SPICE）。

    月球相对地球走开普勒轨道（半长轴 ``distance``、偏心率 ``eccentricity``、
    倾角 ``tilt_deg``），两主星绕惯性中静止的质心对称分布，故
    ``barycenter_accel ≡ 0``。``distance`` 与 ``time_unit`` 满足
    ``n = sqrt(GM_eff/a³) = 1/time_unit``，即圆形模式下 W_nd = (0,0,1)，
    与 CR3BP 口径逐位对齐。
    """

    def __init__(
        self,
        *,
        mu: float,
        distance: float,
        time_unit: float,
        eccentricity: float = 0.0,
        tilt_deg: float = 0.0,
    ) -> None:
        super().__init__(None, mu)
        self.distance = float(distance)
        self.time_unit = float(time_unit)
        self.eccentricity = float(eccentricity)
        self.tilt = np.deg2rad(float(tilt_deg))
        self.mean_motion = float(np.sqrt(GM_EFF / self.distance**3))

    def moon_state(self, et: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """月球相对地球的 J2000 ``(位置, 速度, 加速度)``（开普勒闭式）。"""
        semi_major = self.distance
        eccentricity = self.eccentricity
        eccentric_anomaly = _solve_kepler(self.mean_motion * et, eccentricity)
        cos_e, sin_e = np.cos(eccentric_anomaly), np.sin(eccentric_anomaly)
        root = np.sqrt(1.0 - eccentricity * eccentricity)
        denominator = 1.0 - eccentricity * cos_e
        r_perifocal = semi_major * np.array([cos_e - eccentricity, root * sin_e, 0.0])
        v_perifocal = (
            semi_major
            * self.mean_motion
            * np.array([-sin_e / denominator, root * cos_e / denominator, 0.0])
        )
        r_norm = float(np.linalg.norm(r_perifocal))
        a_perifocal = -GM_EFF * r_perifocal / r_norm**3
        tilt = _tilt_matrix(self.tilt)
        return tilt @ r_perifocal, tilt @ v_perifocal, tilt @ a_perifocal

    def primary_states(self, et: float) -> tuple[np.ndarray, np.ndarray]:
        """两主星在惯性（质心固定）中的绝对状态 ``(地球, 月球)``。"""
        r_moon, v_moon, _ = self.moon_state(et)
        earth = np.concatenate([-self._mu * r_moon, -self._mu * v_moon])
        moon = np.concatenate([(1.0 - self._mu) * r_moon, (1.0 - self._mu) * v_moon])
        return earth, moon

    def frame_state(self, et: float) -> EPPRFrameState:
        r, v, acc = self.moon_state(et)
        distance = float(np.linalg.norm(r))
        h_vec = np.cross(r, v)
        h_norm = float(np.linalg.norm(h_vec))
        e1 = r / distance
        e3 = h_vec / h_norm
        e2 = np.cross(e3, e1)
        e1_rate = v / distance - r * (float(np.dot(r, v)) / distance**3)
        # 平面开普勒轨道：角动量方向恒定 → ė3 = 0、ė2 = e3 × ė1。
        e2_rate = np.cross(e3, e1_rate)
        rotation = np.column_stack([e1, e2, e3])
        rotation_rate = np.column_stack([e1_rate, e2_rate, np.zeros(3)])
        skew = rotation.T @ rotation_rate
        d_rate = float(np.dot(r, v)) / distance
        return EPPRFrameState(
            et=float(et),
            distance=distance,
            distance_rate=d_rate,
            distance_accel=(float(np.dot(v, v)) + float(np.dot(r, acc))) / distance
            - d_rate**2 / distance,
            rotation=rotation,
            rotation_rate=rotation_rate,
            angular_velocity=np.array([skew[2, 1], skew[0, 2], skew[1, 0]]),
            angular_acceleration=np.array([0.0, 0.0, -2.0 * h_norm * d_rate / distance**3]),
            barycenter_accel=np.zeros(3),
        )


def _eppr_to_inertial(
    model: SyntheticEPPRFrameModel, converter: EPPRJ2000System, state_eppr: np.ndarray, et: float
) -> np.ndarray:
    """EPPR 状态 → 惯性（质心固定）状态：加地球绝对状态到地心转换结果。"""
    t_nd = et / TIME_UNIT
    earth_absolute = model.primary_states(et)[0]
    return converter.eppr_to_j2000(state_eppr, t_nd, 0.0) + earth_absolute


_SAMPLE_STATES = [
    np.array([1.12, 0.02, 0.05, 0.0, 0.85, 0.03]),
    np.array([0.83, -0.11, 0.07, 0.21, 0.42, -0.06]),
    np.array([-0.45, -0.36, 0.12, 0.05, -0.28, 0.09]),
]


def test_circular_ephemeris_reduces_to_cr3bp(earth_moon_system):
    """圆形合成星历（d≡lstar、W=(0,0,1)、脉冲项全零）下 EPPR EOM ≡ CR3BP EOM。"""
    model = SyntheticEPPRFrameModel(mu=MU, distance=LSTAR, time_unit=TIME_UNIT)
    system = EPPRSystem(earth_moon_system, et0=0.0, frame_model=model)
    eppr = EPPR_Dynamics(system)
    cr3bp = CR3BP_Dynamics(earth_moon_system)
    for state in _SAMPLE_STATES:
        for tau in (0.0, 0.7, 3.3):
            assert_allclose(
                eppr.equations_of_motion(tau, state),
                cr3bp.equations_of_motion(tau, state),
                atol=1e-10,
            )


def test_elliptic_inertial_equivalence(earth_moon_system):
    """椭圆（e=0.05、i=20°）合成星历：EPPR 传播 ≡ 惯性二体参照。"""
    model = SyntheticEPPRFrameModel(
        mu=MU, distance=LSTAR, time_unit=TIME_UNIT, eccentricity=0.05, tilt_deg=20.0
    )
    system = EPPRSystem(earth_moon_system, et0=0.0, frame_model=model)
    dynamics = EPPR_Dynamics(system)
    dynamics.rtol = 1e-12
    dynamics.atol = 1e-12
    dynamics.max_step = 0.002
    converter = EPPRJ2000System(cr3bp_system=earth_moon_system, frame_model=model)

    state_eppr0 = _SAMPLE_STATES[0]
    inertial0 = _eppr_to_inertial(model, converter, state_eppr0, 0.0)

    gm_primary = (1.0 - MU) * GM_EFF
    gm_secondary = MU * GM_EFF

    def rhs(t_seconds: float, state: np.ndarray) -> np.ndarray:
        earth, moon = model.primary_states(t_seconds)
        position = state[:3]
        to_earth = position - earth[:3]
        to_moon = position - moon[:3]
        acceleration = (
            -gm_primary * to_earth / np.linalg.norm(to_earth) ** 3
            - gm_secondary * to_moon / np.linalg.norm(to_moon) ** 3
        )
        return np.concatenate([state[3:], acceleration])

    tau_end = 2.0
    reference = solve_ivp(
        rhs,
        (0.0, tau_end * TIME_UNIT),
        inertial0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    assert reference.success

    result = dynamics.propagate(state_eppr0, (0.0, tau_end), t_eval=np.linspace(0.0, tau_end, 21))
    final_eppr = result["states"][-1]
    inertial_final = _eppr_to_inertial(model, converter, final_eppr, tau_end * TIME_UNIT)

    assert_allclose(inertial_final[:3] / LSTAR, reference.y[:3, -1] / LSTAR, atol=1e-8)
    assert_allclose(
        inertial_final[3:] * TIME_UNIT / LSTAR,
        reference.y[3:, -1] * TIME_UNIT / LSTAR,
        atol=1e-8,
    )


def test_jacobi_rejected(earth_moon_system):
    """时变系统：Jacobi 两条路径都必须显式拒绝。"""
    model = SyntheticEPPRFrameModel(mu=MU, distance=LSTAR, time_unit=TIME_UNIT)
    system = EPPRSystem(earth_moon_system, et0=0.0, frame_model=model)
    dynamics = EPPR_Dynamics(system)
    state = _SAMPLE_STATES[0]
    with pytest.raises(NotImplementedError):
        dynamics.compute_jacobi_constant(state)
    with pytest.raises(NotImplementedError):
        dynamics.propagate(state, (0.0, 0.1), with_jacobi=True)
    with pytest.raises(NotImplementedError):
        dynamics.propagate(state, (0.0, 0.1), with_stm=True)


@pytest.mark.spice
@requires_spice
class TestEPPREphemerisComparison:
    """真实星历下的 EPPR 与惯性参照 / EphemerisDynamics 对比。"""

    ET0 = (2459000.0 - 2451545.0) * SECONDS_PER_DAY
    INITIAL_J2000 = np.array([380000.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def _system(self, spice_manager, earth_moon_system):
        return EPPRSystem(earth_moon_system, spice_manager, et0=self.ET0)

    def test_inertial_reference_equivalence(self, spice_manager, earth_moon_system):
        system = self._system(spice_manager, earth_moon_system)
        dynamics = EPPR_Dynamics(system)
        dynamics.rtol = 1e-11
        dynamics.atol = 1e-11
        dynamics.max_step = 0.002
        converter = EPPRJ2000System(cr3bp_system=earth_moon_system, spice=spice_manager)

        state_eppr0 = converter.j2000_to_eppr(self.INITIAL_J2000, 0.0, self.ET0)
        earth0 = spice_manager.get_body_state("EARTH", self.ET0, "J2000", "SOLAR SYSTEM BARYCENTER")
        inertial0 = earth0 + self.INITIAL_J2000

        gm_total = LSTAR**3 / TIME_UNIT**2
        gm_primary = (1.0 - MU) * gm_total
        gm_secondary = MU * gm_total

        def rhs(t_seconds: float, state: np.ndarray) -> np.ndarray:
            earth = spice_manager.get_body_state(
                "EARTH", t_seconds, "J2000", "SOLAR SYSTEM BARYCENTER"
            )
            moon = spice_manager.get_body_state(
                "MOON", t_seconds, "J2000", "SOLAR SYSTEM BARYCENTER"
            )
            position = state[:3]
            to_earth = position - earth[:3]
            to_moon = position - moon[:3]
            acceleration = (
                -gm_primary * to_earth / np.linalg.norm(to_earth) ** 3
                - gm_secondary * to_moon / np.linalg.norm(to_moon) ** 3
            )
            return np.concatenate([state[3:], acceleration])

        tau_end = 0.05
        reference = solve_ivp(
            rhs,
            (self.ET0, self.ET0 + tau_end * TIME_UNIT),
            inertial0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-9,
        )
        assert reference.success

        result = dynamics.propagate(
            state_eppr0, (0.0, tau_end), t_eval=np.linspace(0.0, tau_end, 6)
        )
        et_end = self.ET0 + tau_end * TIME_UNIT
        geocentric = converter.eppr_to_j2000(result["states"][-1], tau_end, self.ET0)
        earth_end = spice_manager.get_body_state(
            "EARTH", et_end, "J2000", "SOLAR SYSTEM BARYCENTER"
        )
        inertial_final = earth_end + geocentric

        assert_allclose(inertial_final[:3] / LSTAR, reference.y[:3, -1] / LSTAR, atol=1e-6)
        assert_allclose(
            inertial_final[3:] * TIME_UNIT / LSTAR,
            reference.y[3:, -1] * TIME_UNIT / LSTAR,
            atol=1e-6,
        )

    def test_ephemeris_dynamics_model_difference_bound(self, spice_manager, earth_moon_system):
        """与 EphemerisDynamics(E+M) 的差落在帧锚定模型误差尺度内（<250 km）。

        EPPR 锚定真实地月质心，``barycenter_accel`` 含太阳对质心的加速
        （≈6e-6 km/s²）；``EphemerisDynamics(bodies=[EARTH, MOON])`` 的隐含
        地心按二体模型积分，未含太阳对地球的加速，故两模型的差为
        ≈½·a_sun·T²（2.1 h ≈ 170 km）。精确等价由合成星历与惯性参照两条腿承担。
        """
        system = self._system(spice_manager, earth_moon_system)
        dynamics = EPPR_Dynamics(system)
        dynamics.rtol = 1e-11
        dynamics.atol = 1e-11
        dynamics.max_step = 0.002
        converter = EPPRJ2000System(cr3bp_system=earth_moon_system, spice=spice_manager)

        state_eppr0 = converter.j2000_to_eppr(self.INITIAL_J2000, 0.0, self.ET0)
        tau_end = 0.02
        result = dynamics.propagate(state_eppr0, (0.0, tau_end))
        eppr_geocentric = converter.eppr_to_j2000(result["states"][-1], tau_end, self.ET0)

        ephemeris_session = EphemerisSystem(
            bodies=["EARTH", "MOON"], spice=spice_manager, origin="EARTH"
        )
        ephemeris = EphemerisDynamics(ephemeris_session)
        ephemeris.rtol = 1e-11
        ephemeris.atol = 1e-11
        reference = ephemeris.propagate(
            self.INITIAL_J2000, (self.ET0, self.ET0 + tau_end * TIME_UNIT)
        )
        difference = np.linalg.norm(eppr_geocentric[:3] - reference["states"][-1][:3])
        assert difference < 250.0
