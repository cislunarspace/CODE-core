"""tests/orbit_design 共享 fixture。

地月 CR3BP 系统/动力学的 session 级缓存，供初猜及后续阶段（correction/
continuation/multiple_shooting/ephemeris）测试复用。

注意：本目录的 ``earth_moon_system``/``earth_moon_dynamics`` 覆盖
``tests/conftest.py`` 的函数级同名 fixture——orbit_design 测试统一采用
更精确的地月质量比 μ=0.01215058560962404 与默认特征尺度（地月距
384405 km、周期 27.32 d），与 lissajous/axial 初猜的历史取值一致。

阶段 2 追加 7 条代表轨道的 session 缓存（``_corrected_*_cached``，session
内只修正一次）+ 函数级 deepcopy 包装（``corrected_*``，供单测安全改写）。
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.axial_initial_guess import compute_axial_initial_guess
from e2m2e.algorithm.family.halo_initial_guess import compute_halo_initial_guess
from e2m2e.algorithm.family.spo_initial_guess import compute_spo_initial_guess
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates import SEGMENTED_CORRECTION_ORBIT_TYPES
from e2m2e.data.types.orbit import Orbit
from e2m2e.integrators import collinear_center_modes_py
from tests.algorithm.design import seeds

EARTH_MOON_MU = 0.01215058560962404


def make_design_request(orbit_type: str, **overrides) -> SimpleNamespace:
    """构造 ``design_orbit`` 的 duck-typed 设计请求。

    ``e2m2e.algorithm.design.design_orbit`` 运行期只按属性读取请求（对
    ``DesignOrbitRequest`` 的类型标注在 TYPE_CHECKING 下）；接口层模型的
    校验与按类型默认值填充由 tests/api 覆盖。本工厂钉住算法入口实际读取
    的字段集与公共默认值——字段或默认值语义变动时，本目录的集成测试会
    立即暴露。形状参数默认 None，由调用方按场景显式给定。

    ``correction_method`` 默认镜像请求校验层的族级分派（HALO/NRHO/DPO →
    segmented，其余 → two_level）：算法入口要求请求已按族规范化，未规范
    化的不稳定族请求会被入口防御检查拒绝（test_correction_method_contract）。
    """
    fields = {
        "orbit_type": orbit_type,
        "amplitude": None,
        "phase": None,
        "collinear_point": None,
        "north_south": None,
        "amplitude_in": None,
        "amplitude_out": None,
        "phase_in": None,
        "phase_out": None,
        "perilune_height": None,
        "inclination": None,
        "arg_of_pericenter": None,
        "semi_major_axis": None,
        "epoch": (2024, 1, 1, 0, 0, 0.0),
        "duration": None,
        "output_step": 3600.0,
        "perturbation": None,
        "dyb": None,
        "earth_degree": 10,
        "moon_degree": 10,
        "correction_method": (
            "segmented" if orbit_type.upper() in SEGMENTED_CORRECTION_ORBIT_TYPES else "two_level"
        ),
        "correction_revolutions": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture(scope="session")
def earth_moon_system() -> CR3BP_System:
    """标准地月 CR3BP 系统（session 级，只读复用）。"""
    return CR3BP_System(mu=EARTH_MOON_MU, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture(scope="session")
def earth_moon_dynamics(earth_moon_system: CR3BP_System) -> CR3BP_Dynamics:
    """标准地月 CR3BP 动力学（session 级，只读复用）。"""
    return CR3BP_Dynamics(system=earth_moon_system)


# =============================================================================
# 阶段 2：代表轨道 session 缓存
#
# 每条 ``_corrected_*_cached`` 用对应 setup + 标准种子 iterate_correction
# （SPO 用 iterate_full_period_correction）修正一次，session 内复用；
# ``corrected_*`` 是其 deepcopy，供单测改写。修正耗时主要在 STM 传播，
# 缓存后 correction/ 测试只 deepcopy + 断言，durations 应集中在 setup。
# =============================================================================


def _seed_orbit(dynamics: CR3BP_Dynamics, state: np.ndarray, period: float) -> Orbit:
    """构造单点种子 Orbit（t=0 初态 + 周期猜测）。"""
    orbit = Orbit(
        states=np.asarray(state, dtype=float).reshape(1, -1),
        times=np.array([0.0]),
        system=dynamics.system,
    )
    orbit.period = period
    return orbit


def _lyapunov_l1_seed(dynamics: CR3BP_Dynamics) -> tuple[np.ndarray, float]:
    """Lyapunov L1 平面种子：面内线性模态构造，满足 y(0)=ẋ(0)=0。

    与 ``axial_initial_guess._correct_lyapunov_fixed_x0`` 的 guess=None 分支
    同构：相位 phi 使 y(0)=0，x0 = x_L1 − offset，vy0 由特征向量定出。
    """
    system = dynamics.system
    x_l, omega_xy, _, y_ratio = collinear_center_modes_py(float(system.mu), seeds.LYAPUNOV_POINT)
    t_lin = 2.0 * np.pi / omega_xy
    x0 = x_l - seeds.LYAPUNOV_OFFSET
    vy0 = (x0 - x_l) * (-omega_xy * y_ratio)
    state = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
    return state, t_lin


@pytest.fixture(scope="session")
def _corrected_dro_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """DRO 地月（Cui 2025）：固定 x0，自由 vy0 与半周期。"""
    dynamics = earth_moon_dynamics
    state = np.array([seeds.DRO_X0, 0.0, 0.0, 0.0, seeds.DRO_VY0, 0.0])
    seed = _seed_orbit(dynamics, state, seeds.DRO_PERIOD)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(seeds.DRO_X0)
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "DRO 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_halo_l1_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """Halo L1 北族：固定 z0，自由 x0/vy0/半周期；Richardson 种子。"""
    dynamics = earth_moon_dynamics
    g = compute_halo_initial_guess(
        mu=dynamics.system.mu,
        z_amplitude=seeds.HALO_SEED_Z0,
        L=1,
        halo_class=seeds.HALO_SEED_CLASS,
    )
    state = np.array([g["x0"], 0.0, seeds.HALO_SEED_Z0, 0.0, g["vy0"], 0.0])
    seed = _seed_orbit(dynamics, state, 2.0 * g["T_half"])
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(seeds.HALO_SEED_Z0, 1)
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "Halo L1 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_halo_l2_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """Halo L2 北族：固定 z0，自由 x0/vy0/半周期；Richardson 种子。"""
    dynamics = earth_moon_dynamics
    g = compute_halo_initial_guess(
        mu=dynamics.system.mu,
        z_amplitude=seeds.HALO_SEED_Z0,
        L=2,
        halo_class=seeds.HALO_SEED_CLASS,
    )
    state = np.array([g["x0"], 0.0, seeds.HALO_SEED_Z0, 0.0, g["vy0"], 0.0])
    seed = _seed_orbit(dynamics, state, 2.0 * g["T_half"])
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_halo_orbit_fixed_z0(seeds.HALO_SEED_Z0, 2)
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "Halo L2 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_lyapunov_l1_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """Lyapunov L1 平面族：固定 x0，自由 vy0 与半周期；面内线性模态种子。"""
    dynamics = earth_moon_dynamics
    state, period = _lyapunov_l1_seed(dynamics)
    seed = _seed_orbit(dynamics, state, period)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(float(state[0]))
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "Lyapunov L1 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_axial_l1_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """Axial L1（Gómez Type B）：固定 vz0，自由 x0/vy0/半周期；分岔种子。"""
    dynamics = earth_moon_dynamics
    state, period = compute_axial_initial_guess(
        dynamics, collinear_point=seeds.AXIAL_SEED_POINT, vz0=seeds.AXIAL_SEED_VZ0
    )
    seed = _seed_orbit(dynamics, state, period)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_axial_orbit_fixed_vz0(seeds.AXIAL_SEED_VZ0, seeds.AXIAL_SEED_POINT)
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "Axial L1 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_dpo_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """DPO 地月（顺行）：固定 x0，自由 vy0 与半周期；标准种子。"""
    dynamics = earth_moon_dynamics
    state = np.array([seeds.DPO_X0, 0.0, 0.0, 0.0, seeds.DPO_VY0, 0.0])
    seed = _seed_orbit(dynamics, state, seeds.DPO_PERIOD)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(seeds.DPO_X0)
    result = corrector.iterate_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "DPO 修正未收敛"
    return orbit, result


@pytest.fixture(scope="session")
def _corrected_triangular_l4_cached(earth_moon_dynamics: CR3BP_Dynamics) -> Orbit:
    """三角平动点区域周期轨道：SPO L4（短周期族），全周期闭合修正。

    三模态 Triangular 初猜是拟周期的（无周期修正 setup）；L4/L5 区域可用
    全周期闭合修正的周期族是 SPO，故以此代表「Triangular L4」的修正行为。
    """
    dynamics = earth_moon_dynamics
    state, period = compute_spo_initial_guess(
        dynamics.system, seeds.SPO_SEED_POINT, seeds.SPO_SEED_AMPLITUDE_KM
    )
    seed = _seed_orbit(dynamics, state, period)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_spo_fixed_x0(float(state[0]), seeds.SPO_SEED_POINT)
    result = corrector.iterate_full_period_correction(seed, verbose=False)
    orbit = result.orbit
    assert orbit is not None, "SPO L4 修正未收敛"
    return orbit, result


# ---- 函数级 deepcopy 包装：单测可安全改写，session 缓存不受影响 ----


@pytest.fixture
def corrected_dro(_corrected_dro_cached):
    orbit, result = _corrected_dro_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_halo_l1(_corrected_halo_l1_cached):
    orbit, result = _corrected_halo_l1_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_halo_l2(_corrected_halo_l2_cached):
    orbit, result = _corrected_halo_l2_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_lyapunov_l1(_corrected_lyapunov_l1_cached):
    orbit, result = _corrected_lyapunov_l1_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_axial_l1(_corrected_axial_l1_cached):
    orbit, result = _corrected_axial_l1_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_dpo(_corrected_dpo_cached):
    orbit, result = _corrected_dpo_cached
    return copy.deepcopy(orbit), result


@pytest.fixture
def corrected_triangular_l4(_corrected_triangular_l4_cached):
    orbit, result = _corrected_triangular_l4_cached
    return copy.deepcopy(orbit), result
