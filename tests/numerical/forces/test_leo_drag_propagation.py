"""LEO 阻力传播端到端测试（需 SPICE 内核）。"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces import ForceModel, GravityField
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere
from e2m2e.algorithm.forces.drag import DragModel
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = pytest.mark.force


_EARTH_R_KM = 6378.137
_MU_EARTH = 398600.4415  # km³/s²


@pytest.fixture
def leo_system(spice_kernel_path):
    """LEO 传播用的 EphemerisSystem（ICRF + 地球中心）。"""
    from kernel_helpers import load_body_fixed_kernels, unload_kernels

    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    bf_kernels = load_body_fixed_kernels(spice)
    try:
        system = EphemerisSystem(
            bodies=["EARTH"],
            spice=spice,
            origin="EARTH",
        )
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        yield system
    finally:
        unload_kernels(spice, bf_kernels)
        spice.unload_kernel(spice_kernel_path)


def _keplerian_to_cartesian(a, e, i, raan, argp, nu, mu):
    """将开普勒根数转为笛卡尔状态。"""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    nu = np.radians(nu)

    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pqw = np.array(
        [
            -np.sqrt(mu / p) * np.sin(nu),
            np.sqrt(mu / p) * (e + np.cos(nu)),
            0.0,
        ]
    )

    R3_raan = np.array(
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R1_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ]
    )
    R3_argp = np.array(
        [
            [np.cos(argp), -np.sin(argp), 0.0],
            [np.sin(argp), np.cos(argp), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = R3_raan @ R1_i @ R3_argp

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return np.concatenate([r_eci, v_eci])


def _semi_major_axis(state, mu):
    """从状态向量用能量公式计算半长轴。"""
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    energy = v**2 / 2.0 - mu / r
    return -mu / (2.0 * energy)


@pytest.mark.spice
def test_leo_drag_semi_major_axis_decays(leo_system):
    """LEO 1 天传播（J2 + 阻力），半长轴衰减方向正确且量级合理。"""
    system = leo_system
    mu = _MU_EARTH

    a0 = _EARTH_R_KM + 300.0  # 300 km 圆轨道
    e = 0.001
    i = 51.6

    y0 = _keplerian_to_cartesian(a0, e, i, 0.0, 0.0, 0.0, mu)

    atm = ExponentialAtmosphere()
    drag = DragModel(atmosphere=atm, area=10.0, mass=1000.0, cd=2.2)
    gravity = GravityField(body="EARTH", degree=2, order=0)
    fm = ForceModel(system, forces=[gravity, drag])

    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    t_span = (et0, et0 + 86400.0)
    t_eval = np.linspace(et0, et0 + 86400.0, 200)

    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    # Osculating 半长轴随轨道相位振荡（偏心率 + 摄动），振幅 ~9 km，
    # 远大于日衰减 ~2 km。对 a(t) 做线性拟合提取 secular 斜率。
    a_history = np.array([_semi_major_axis(s, mu) for s in result["states"]])
    times_day = (result["time"] - result["time"][0]) / 86400.0
    coeffs = np.polyfit(times_day, a_history, 1)
    delta_a_per_day = coeffs[0]  # km/day

    # 方向：半长轴缩短
    assert delta_a_per_day < 0, f"半长轴应衰减，但 da/day={delta_a_per_day:.4f} km/day"

    # 量级：与解析 da/dt = -rho·BC·sqrt(mu·a) 交叉验证（±50%）
    rho = atm.density(300.0)  # kg/m³
    bc = 2.2 * 10.0 / 1000.0  # m²/kg
    mu_si = _MU_EARTH * 1e9  # m³/s²
    a_si = a0 * 1000  # m
    da_dt_theory = -rho * bc * np.sqrt(mu_si * a_si)  # m/s
    delta_a_theory = da_dt_theory * 86400.0 / 1000.0  # km/day

    relative_error = abs((delta_a_per_day - delta_a_theory) / delta_a_theory)
    assert relative_error < 0.50, (
        f"半长轴衰减量级偏差过大: measured={delta_a_per_day:.4f} km/day, "
        f"theory={delta_a_theory:.4f} km/day, error={relative_error:.1%}"
    )


@pytest.mark.spice
def test_drag_rust_path_respects_configured_f107_ap(leo_system):
    """#315 端到端：Rust 路径必须响应用户配置的 f107/ap。

    全链验证：``DragModel.to_rust_spec``（带 f107/ap）→ lib.rs 解析 7 元组 →
    ``CompiledForce::Drag`` → ``drag_accel`` → ``density(h, f107, ap)``。

    隔离手法：两轮都用 Rust 路径（``propagate_compiled``），仅改大气模型 f107/ap。
    Rust 内部 ITRF93 pxform 帧旋转在两轮间完全相同、相消，残差纯来自密度差。

    判据：bug 修复前 Rust 硬编码 150/15，改 f107/ap 不影响结果 → 两轮末态相同
    （diff≈0）；修复后两轮末态应有可测差异（~10 m 量级）。

    注：不能直接比 Rust vs Python 末态——Rust 用 SPICE ITRF93 pxform、Python 用
    ITRFApproxAxes 近似旋转，固有 ~7 m 基线分歧（drag.rs 决策 1b），且该基线随
    密度放大，会淹没 f107/ap 信号。帧旋转一致性是独立议题，不在 #315 范围。
    """
    system = leo_system
    mu = _MU_EARTH

    a0 = _EARTH_R_KM + 400.0  # 400 km 圆轨道，drag 量级可感知
    y0 = _keplerian_to_cartesian(a0, 0.0, 51.6, 0.0, 0.0, 0.0, mu)

    spice = system.spice
    et0 = spice.utc_to_et("2025-06-21T11:00:06")
    dt = 5400.0  # ~0.6 圈，足以让密度差放大到可测
    t_eval = np.array([et0, et0 + dt])

    def propagate_rust(f107: float, ap: float) -> np.ndarray:
        drag = DragModel(
            atmosphere=ExponentialAtmosphere(f107=f107, ap=ap),
            area=10.0,
            mass=1000.0,
            cd=2.2,
        )
        gravity = GravityField(body="EARTH", degree=2, order=0)
        fm = ForceModel(system, forces=[gravity, drag])
        fm.rtol = 1e-10
        assert fm._can_use_rust_path(), "spice 构建下应走 Rust 路径"
        result = fm.propagate(y0, (et0, et0 + dt), t_eval=t_eval, max_steps=200_000)
        return np.asarray(result["states"][-1])

    state_default = propagate_rust(150.0, 15.0)
    state_hot = propagate_rust(200.0, 50.0)

    diff = np.linalg.norm(state_hot - state_default)
    # 修复前 diff≈0（Rust 忽略 f107/ap）；修复后密度因子 1.44 → drag 差 ~44%，
    # 5400s 短弧末态差约 1e-2 km（~10 m）。下界 1e-3 km（1 m）留一个数量级余量，
    # 远高于 0，可靠区分 bug 是否回归。
    assert diff > 1e-3, (
        f"Rust 路径改 f107/ap 后末态几乎不变（diff={diff:.3e} km），"
        "f107/ap 未透传到 density（#315 回归）"
    )
