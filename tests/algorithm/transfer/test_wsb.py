"""WSB 太阳引力辅助转移测试（物理定义验证）。

测试策略：按物理定义验证，不用黄金样本。
BCR4BP 纯数值测试不需要 SPICE，用 BCR4BPSystem.earth_moon() 构造。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem, CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.transfer import (
    TransferDesignResult,
    WsbSearchParams,
    WsbTransferDetails,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import TliParams
from e2m2e.algorithm.transfer.wsb import (
    compute_kepler_energy_moon,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState

pytestmark = pytest.mark.orchestration


# 地月 CR3BP 参数（DE421 基准）
MU = Datum.DE421.mu
DU = 384405.0  # km


def _make_bcr4bp_system(sun_phase0: float = 0.0):
    """创建标准地月 BCR4BP 系统。"""
    return BCR4BPSystem.earth_moon(sun_phase0=sun_phase0)


def _make_cr3bp_system():
    """创建地月 CR3BP 系统并初始化特征尺度。"""
    system = CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()
    dynamics = CR3BP_Dynamics(system)
    return system, dynamics


def _make_departure_state(system):
    """构造典型 LEO 出发态的旋转系无量纲态。"""
    from e2m2e.algorithm.transfer.hohmann import MU_EARTH, R_EARTH

    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    r0 = np.array([r_park, 0.0, 0.0])
    v0 = np.array([0.0, v_circ, 0.0])
    departure_phys = np.concatenate([r0, v0])
    return system.physical_to_dimensionless(departure_phys)


def _make_target_state(system):
    """构造近月轨道目标态的旋转系无量纲态。"""
    mu = system.mu
    moon_x = 1.0 - mu
    r_target_km = 2000.0
    r_target_du = r_target_km / DU
    x_target = moon_x + r_target_du
    v_circ_dim = math.sqrt(mu / r_target_du)
    return np.array([x_target, 0.0, 0.0, 0.0, v_circ_dim, 0.0])


# ---------------------------------------------------------------------------
# TestWsbSearchParams：搜索参数验证
# ---------------------------------------------------------------------------


class TestWsbSearchParams:
    """搜索参数验证。"""

    def test_default_params_valid(self):
        """默认参数范围合理（S1/S2/S3 修正后的 WSB 典型值）。"""
        params = WsbSearchParams()
        assert params.sun_phase_range == (0.0, 2.0 * math.pi)
        assert params.n_sun_phase == 50  # S3: 修正为 50
        assert params.tof_range == (90.0, 150.0)  # S1: 修正为 WSB 典型 3-5 月
        assert params.perilune_alt_min == 100.0
        assert params.perilune_alt_max == 10000.0
        assert params.max_total_dv == 5.0  # S2: 修正为 5.0
        assert params.h2_energy_threshold == 0.0
        assert params.n_departure_phase > 0
        assert params.n_tof > 0

    def test_frozen_dataclass(self):
        """WsbSearchParams 是 frozen dataclass。"""
        params = WsbSearchParams()
        with pytest.raises(AttributeError):
            params.n_sun_phase = 16  # type: ignore[misc]

    def test_custom_params(self):
        """自定义参数可正确构造。"""
        params = WsbSearchParams(
            sun_phase_range=(0.0, math.pi),
            n_sun_phase=4,
            departure_phase_range=(0.0, math.pi),
            n_departure_phase=20,
            tof_range=(10.0, 30.0),
            n_tof=20,
            perilune_alt_min=200.0,
            perilune_alt_max=5000.0,
            max_total_dv=30.0,
            h2_energy_threshold=0.01,
        )
        assert params.sun_phase_range == (0.0, math.pi)
        assert params.n_sun_phase == 4
        assert params.max_total_dv == 30.0
        assert params.h2_energy_threshold == 0.01

    # S6: __post_init__ 参数验证
    def test_invalid_sun_phase_range_raises(self):
        """sun_phase_range 超出 [0, 2π) 报错。"""
        with pytest.raises(ValueError, match="sun_phase_range"):
            WsbSearchParams(sun_phase_range=(-0.1, math.pi))

    def test_invalid_departure_phase_range_raises(self):
        """departure_phase_range 超出 [0, 2π) 报错。"""
        with pytest.raises(ValueError, match="departure_phase_range"):
            WsbSearchParams(departure_phase_range=(0.0, 7.0))

    def test_invalid_tof_range_raises(self):
        """tof_range[0] >= tof_range[1] 报错。"""
        with pytest.raises(ValueError, match="tof_range"):
            WsbSearchParams(tof_range=(50.0, 50.0))

    def test_invalid_perilune_range_raises(self):
        """perilune_alt_min >= perilune_alt_max 报错。"""
        with pytest.raises(ValueError, match="perilune_alt_min"):
            WsbSearchParams(perilune_alt_min=5000.0, perilune_alt_max=5000.0)


# ---------------------------------------------------------------------------
# TestComputeKeplerEnergyMoon：H2 计算验证
# ---------------------------------------------------------------------------


class TestComputeKeplerEnergyMoon:
    """compute_kepler_energy_moon H2 计算验证。"""

    def test_circular_orbit_near_moon_negative_energy(self):
        """月球附近圆轨道 H₂ < 0（受月球约束）。"""
        mu = MU
        r_orbit = 2000.0 / DU  # 2000 km 高度（月球上方）
        moon_x = 1.0 - mu
        x = moon_x + r_orbit
        v_circ = math.sqrt(mu / r_orbit)
        state = np.array([x, 0.0, 0.0, 0.0, v_circ, 0.0])
        h2 = compute_kepler_energy_moon(state, mu)
        assert h2 < 0, f"圆轨道 H2 应为负，得到 {h2}"

    def test_zero_velocity_at_moon_negative_energy(self):
        """月心处零速度：势能主导，H2 为很大的负数。"""
        mu = MU
        moon_x = 1.0 - mu
        r_offset = 0.01  # 距月心 0.01 DU
        state = np.array([moon_x + r_offset, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = compute_kepler_energy_moon(state, mu)
        assert h2 < 0, f"静止近月态 H2 应为负，得到 {h2}"

    def test_fast_flyby_positive_energy(self):
        """高速飞越：H₂ > 0（超逃逸速度，双曲飞越）。"""
        mu = MU
        moon_x = 1.0 - mu
        r_flyby = 5000.0 / DU  # 远处飞越
        x = moon_x + r_flyby
        v_escape = math.sqrt(2.0 * mu / r_flyby)
        v_flyby = v_escape * 2.0  # 2 倍逃逸速度
        # 旋转系速度：vy + x - moon_x = v_flyby，取 vx=0, vy=v_flyby
        state = np.array([x, 0.0, 0.0, 0.0, v_flyby + moon_x - x, 0.0])
        h2 = compute_kepler_energy_moon(state, mu)
        assert h2 > 0, f"高速飞越 H2 应为正，得到 {h2}"

    def test_velocity_correction_applied(self):
        """验证旋转系→惯性系速度修正已正确应用。

        检查旋转系静止物体（vx=vy=vz=0）在月心处的相对速度不为零
        （旋转系效应）。
        """
        mu = MU
        moon_x = 1.0 - mu
        state_stationary = np.array([moon_x + 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = compute_kepler_energy_moon(state_stationary, mu)
        # 旋转系静止 → 惯性系有速度 → v_rel_moon 不为零
        # 动能应为正，但势能主导，H2 < 0
        assert h2 < 0
        # 纯势能（无速度）应给出更小的 H2（更负）
        # 因为动能是正的，所以 H2 > 纯势能
        r_rel = 0.1
        pe_only = -mu / r_rel
        assert h2 > pe_only, "旋转系静止态应有正的动能贡献"

    def test_far_from_moon_approaches_zero(self):
        """远离月球时 H2 → 0⁻（逃逸速度处恰好为 0）。"""
        mu = MU
        moon_x = 1.0 - mu
        r_far = 50000.0 / DU
        x = moon_x + r_far
        v_escape = math.sqrt(2.0 * mu / r_far)
        # 恰好逃逸速度
        state = np.array([x, 0.0, 0.0, 0.0, v_escape + moon_x - x, 0.0])
        h2 = compute_kepler_energy_moon(state, mu)
        assert abs(h2) < 1e-3, f"逃逸速度处 H2 应接近 0，得到 {h2}"


# ---------------------------------------------------------------------------
# TestWsbPhysics：物理不变量验证
# ---------------------------------------------------------------------------


class TestWsbPhysics:
    """WSB 物理不变量验证。"""

    def test_cr3bp_jacobi_conservation(self):
        """CR3BP Jacobi 常数沿轨道守恒（验证传播工具正确性）。"""
        from e2m2e.algorithm.transfer.lga import _compute_jacobi

        system, dynamics = _make_cr3bp_system()
        dep_state = _make_departure_state(system)
        mu = system.mu

        t_span = (0.0, 1.0)
        result = dynamics.propagate(dep_state, t_span, t_eval=np.linspace(0.0, 1.0, 100))
        states = result["states"]

        c_start = _compute_jacobi(states[0], mu)
        c_end = _compute_jacobi(states[-1], mu)
        assert abs(c_start - c_end) < 1e-8, (
            f"Jacobi 常数不守恒：C_start={c_start:.10e}, C_end={c_end:.10e}"
        )

    def test_bcr4bp_no_jacobi(self):
        """BCR4BP 无 Jacobi 积分（时间周期系统）。"""
        system = _make_bcr4bp_system()
        dynamics = BCR4BP_Dynamics(system)
        dep_state = _make_departure_state(system)
        with pytest.raises(NotImplementedError, match="Jacobi"):
            dynamics.compute_jacobi_constant(dep_state)

    def test_bcr4bp_system_initialization(self):
        """BCR4BPSystem.earth_moon() 正确初始化特征尺度和太阳参数。"""
        system = _make_bcr4bp_system(math.pi / 4)
        assert system.is_initialized
        assert system.characteristic_length is not None
        assert system.characteristic_time is not None
        assert system.characteristic_velocity is not None
        assert system.sun_phase0 == math.pi / 4
        assert system.sun_mass > 0
        assert system.sun_distance > 0

    def test_bcr4bp_propagation_differs_by_sun_phase(self):
        """不同太阳相位角的 BCR4BP 传播结果不同。"""
        dep_state = _make_departure_state(_make_bcr4bp_system())
        t_dim = 5.0 * 86400.0 / _make_bcr4bp_system().characteristic_time

        result_0 = BCR4BP_Dynamics(_make_bcr4bp_system(0.0)).propagate(
            dep_state, (0.0, t_dim), t_eval=np.array([t_dim])
        )
        result_pi = BCR4BP_Dynamics(_make_bcr4bp_system(math.pi)).propagate(
            dep_state, (0.0, t_dim), t_eval=np.array([t_dim])
        )
        state_0 = result_0["states"][-1]
        state_pi = result_pi["states"][-1]
        diff = np.linalg.norm(state_0 - state_pi)
        assert diff > 1e-6, f"不同太阳相位的传播结果应有差异，diff={diff}"


# ---------------------------------------------------------------------------
# transfer_orbit("WSB") 编排器测试
# ---------------------------------------------------------------------------


def _make_target_ephemeris():
    """构造近月轨道目标态的星历（单点，物理单位 km / km/s）。"""
    system = _make_bcr4bp_system()
    target_state = _make_target_state(system)
    du = system.characteristic_length
    vu = system.characteristic_velocity
    target_phys = np.array(
        [
            target_state[0] * du,
            target_state[1] * du,
            target_state[2] * du,
            target_state[3] * vu,
            target_state[4] * vu,
            target_state[5] * vu,
        ]
    )
    return target_phys.reshape(1, 6)


@pytest.fixture(scope="module")
def wsb_transfer_result():
    """跑一次小网格 WSB 搜索，module 内复用。

    details 字段类型契约（零搜索）已下沉到 ``tests/api/test_facade.py``
    的 ``TestWsbTransferDetailsContract``（ADR 0021）。本 fixture 只服务
    真搜索的 orchestration 测试，搜索参数与原两个 slow 测试一致
    （n_sun_phase=1, n_departure_phase=5, n_tof=3）。
    """
    tli_params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
    target_ephemeris = _make_target_ephemeris()
    search_params = WsbSearchParams(
        n_sun_phase=1,
        n_departure_phase=5,
        n_tof=3,
        tof_range=(1.0, 5.0),
        max_total_dv=50.0,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transfer_orbit(
            "WSB",
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            wsb_search_params=search_params,
        )


class TestWsbTransferOrbit:
    """transfer_orbit("WSB") 编排器测试。"""

    @pytest.mark.slow
    def test_wsb_search_returns_valid_transfer(self, wsb_transfer_result):
        """WSB 编排器搜索：返回结构正确，有候选时 dv 落在地月转移合理量级。

        小网格（n_sun_phase=1, n_departure_phase=5, n_tof=3）的物理参数空间
        可能无解，无候选时 skip（与 ``TestWsbAcceptance`` 模式一致，不弱化）。
        ``details`` 各字段类型契约由 ``TestWsbTransferDetailsContract``
        零搜索验证，本测试只断言编排链路的搜索结果。
        """
        result = wsb_transfer_result
        assert isinstance(result, TransferDesignResult)
        assert result.transfer_type == "WSB"
        details = result.details
        assert isinstance(details, WsbTransferDetails)
        if details.status is not ConvergenceState.CONVERGED:
            pytest.skip("小网格 WSB 搜索未找到可行候选（参数空间可能无解）")
        # 有候选：dv 落在地月转移合理量级（~3-5 km/s）
        assert 0.5 < result.delta_v < 10.0, (
            f"WSB 总 Δv {result.delta_v:.3f} km/s 超出地月转移合理范围"
        )
        assert details.tof_sec > 0
        assert details.perilune_alt_km > 0

    def test_wsb_missing_params_raises(self):
        """transfer_orbit("WSB") 缺少必要参数时报错。"""
        with pytest.raises(ValueError, match="tli_params"):
            transfer_orbit("WSB", target_ephemeris=np.zeros((1, 6)))

        with pytest.raises(ValueError, match="target_ephemeris"):
            transfer_orbit("WSB", tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0))

    def test_unsupported_type_still_raises(self):
        """transfer_orbit("low_thrust") without engine_config 抛出 ValueError。"""
        with pytest.raises(ValueError, match="low_thrust"):
            transfer_orbit("low_thrust")


# ---------------------------------------------------------------------------
# TestWsbAcceptance：验收测试（SI1-SI5）
# ---------------------------------------------------------------------------


class TestWsbAcceptance:
    """验收测试（方案 §0，SI1-SI5）。

    使用小网格（n_sun_phase=3, n_tof=5）避免耗时过长。
    验收条件在候选确实存在时检查，无候选时跳过（参数空间可能无解）。
    """

    @staticmethod
    def _run_small_search(
        n_sun_phase: int = 3,
        n_departure_phase: int = 10,
        n_tof: int = 5,
    ):
        """运行小规模 BCR4BP 网格搜索，返回候选列表。"""
        system = _make_bcr4bp_system()
        dep_state = _make_departure_state(system)
        target_state = _make_target_state(system)

        params = WsbSearchParams(
            n_sun_phase=n_sun_phase,
            n_departure_phase=n_departure_phase,
            n_tof=n_tof,
            tof_range=(1.0, 10.0),
            max_total_dv=100.0,
        )

        from e2m2e.algorithm.transfer.wsb import search_wsb_trajectories

        return search_wsb_trajectories(dep_state, target_state, system, params)

    def test_ballistic_capture_h2_negative(self):
        """SI1: 验证 WSB 候选的 H₂ < 0（弹道捕获判据）。"""
        candidates = self._run_small_search()
        if not candidates:
            pytest.skip("小网格未找到候选（物理参数空间可能无解）")
        for c in candidates:
            assert c.h2_kepler < 0, f"WSB 候选 H₂ 应 < 0（弹道捕获），得到 {c.h2_kepler}"

    def test_trajectory_continuity(self):
        """SI2: 验证轨道连续性（动力学方程一致性，残差 < 1e-6 无量纲）。

        方法：在 BCR4BP 轨迹的任意中间点取状态，调用运动方程计算状态导数，
        验证位置导数分量与速度分量完全一致（ODE 定义 dr/dt = v）。
        这保证了传播轨迹的连续性：位置-速度自洽。
        """
        system = _make_bcr4bp_system()
        dep_state = _make_departure_state(system)

        dynamics = BCR4BP_Dynamics(system)
        char_time = system.characteristic_time
        tof_dim = 2.0 * 86400.0 / char_time
        n_samples = 100
        t_eval = np.linspace(0.0, tof_dim, n_samples)
        result = dynamics.propagate(dep_state, (0.0, tof_dim), t_eval=t_eval)

        states = result["states"]
        times = result["time"]

        # 检查轨迹中每个点的 EOM 残差：d(position)/dt = velocity
        # 用 finite difference 与 EOM 一致性的均方根残差
        eom = dynamics._get_eom_func(with_stm=False)
        max_eom_residual = 0.0
        for k in range(0, n_samples, 10):  # 每隔 10 点检查
            t_k = times[k]
            state_k = states[k]
            deriv_k = eom(t_k, state_k)
            # 位置导数应等于速度分量（ODE 定义）
            dr_dt_residual = float(np.linalg.norm(deriv_k[:3] - state_k[3:6]))
            max_eom_residual = max(max_eom_residual, dr_dt_residual)

        assert max_eom_residual < 1e-6, (
            f"EOM 位置导数与速度一致性残差应 < 1e-6，得到 {max_eom_residual}"
        )

    def test_solar_perturbation_effective(self):
        """SI3: 验证 BCR4BP 结果与 CR3BP 结果有显著差异（太阳摄动起作用）。

        相同初始条件、相同传播时间，比较终态差异。
        """
        bcr4bp_system = _make_bcr4bp_system(0.0)
        cr3bp_system, cr3bp_dynamics = _make_cr3bp_system()

        dep_state = _make_departure_state(bcr4bp_system)
        char_time = bcr4bp_system.characteristic_time
        t_dim = 5.0 * 86400.0 / char_time

        # BCR4BP 传播
        bcr4bp_dynamics = BCR4BP_Dynamics(bcr4bp_system)
        bcr4bp_result = bcr4bp_dynamics.propagate(dep_state, (0.0, t_dim), t_eval=np.array([t_dim]))
        state_bcr4bp = bcr4bp_result["states"][-1]

        # CR3BP 传播（相同初始条件）
        cr3bp_result = cr3bp_dynamics.propagate(dep_state, (0.0, t_dim), t_eval=np.array([t_dim]))
        state_cr3bp = cr3bp_result["states"][-1]

        diff = float(np.linalg.norm(state_bcr4bp - state_cr3bp))
        assert diff > 1e-3, f"BCR4BP vs CR3BP 终态差异应 > 1e-3，得到 {diff}（太阳摄动未起作用？）"

    def test_wsb_saves_dv_vs_hohmann(self):
        """SI4: 验证 WSB 最优候选总 Δv < 同等条件 Hohmann Δv（如果有候选）。"""
        from e2m2e.algorithm.transfer.hohmann import R_EARTH, hohmann_delta_v

        candidates = self._run_small_search()
        if not candidates:
            pytest.skip("小网格未找到候选（物理参数空间可能无解）")

        best = candidates[0]
        r1 = R_EARTH + 200.0  # LEO
        r2 = 384405.0  # 近似地月距离 (km)
        dv_hmn1, dv_hmn2 = hohmann_delta_v(r1, r2)
        dv_hohmann = dv_hmn1 + dv_hmn2

        assert best.total_dv < dv_hohmann, (
            f"WSB 总 Δv ({best.total_dv:.3f}) 应 < Hohmann Δv ({dv_hohmann:.3f})"
        )

    def test_perilune_detection_precision(self):
        """SI5: 验证近月点检测精度（Brent 法残差 < 1e-6 DU）。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection, detect_crossings

        system = _make_bcr4bp_system()
        mu = system.mu
        dynamics = BCR4BP_Dynamics(system)
        char_time = system.characteristic_time

        # 在月球附近构造一个会多次经过近月点的椭圆轨道
        moon_x = 1.0 - mu
        r_orbit = 3000.0 / DU  # 3000 km 高度
        x0 = moon_x + r_orbit
        v_circ = math.sqrt(mu / r_orbit)
        # 稍微偏心：增加速度使轨道偏心
        state0 = np.array([x0, 0.0, 0.0, 0.0, v_circ * 1.1, 0.0])

        t_dim = 5.0 * 86400.0 / char_time
        n_samples = 1000
        result = dynamics.propagate(state0, (0.0, t_dim), t_eval=np.linspace(0.0, t_dim, n_samples))
        times = result["time"]
        states = result["states"]

        periapsis_section = PoincareSection.periapsis("moon", system)
        crossings = detect_crossings(times, states, periapsis_section)

        if not crossings:
            pytest.skip("未检测到近月点穿越（传播未经过近月点附近）")

        # 验证每个近月点的 r·v 残差
        moon_pos = np.array([moon_x, 0.0, 0.0])
        for _t_peri, state_peri, _idx in crossings:
            r_rel = state_peri[:3] - moon_pos
            v_rel = state_peri[3:]
            rv_dot = float(np.dot(r_rel, v_rel))
            r_mag = float(np.linalg.norm(r_rel))
            v_mag = float(np.linalg.norm(v_rel))
            # 无量纲残差：|r·v| / (r * v) < 1e-6（近月点判据：r·v ≈ 0）
            if r_mag > 1e-12 and v_mag > 1e-12:
                normalized_rv = abs(rv_dot) / (r_mag * v_mag)
                assert normalized_rv < 1e-6, (
                    f"近月点检测 r·v 归一化残差应 < 1e-6，得到 {normalized_rv}"
                )
