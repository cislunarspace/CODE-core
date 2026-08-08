"""BCR4BP（双圆限制性四体问题）系统与动力学测试。

覆盖：
  A. 太阳无量纲参数取值（对照文献标准值）与太阳解析位置；
  B. 太阳质量置零时与 CR3BP_Dynamics 逐点一致（退化检验）；
  C. 雅可比与 STM 的有限差分校验；
  D. propagate 接口契约（形状、时间单调）；
  E. 与星历 ForceModel（地球+月球+太阳点质量，SPICE 内核）的短时间
     外推对比（双圆近似 vs 真实星历，容差依据见测试内注释）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import BCR4BP_Dynamics, BCR4BPSystem, CR3BP_Dynamics, CR3BP_System

MU = 0.0121506683


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def bcr4bp_system():
    """标准地月 BCR4BP 系统。"""
    return BCR4BPSystem.earth_moon()


@pytest.fixture
def bcr4bp_dynamics(bcr4bp_system):
    """BCR4BP 动力学对象。"""
    return BCR4BP_Dynamics(bcr4bp_system)


@pytest.fixture
def cr3bp_system():
    """同参数地月 CR3BP 系统（退化对照）。"""
    return CR3BP_System(mu=MU, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture
def sample_state():
    """3:1 DRO 初始状态（无量纲，与 conftest 的 dro_31_state 一致）。"""
    return np.array([1.1202109158830986, 0.0, 0.0, 0.0, -0.46178983697629084, 0.0])


# =============================================================================
# A. 太阳参数与解析位置
# =============================================================================
class TestSunParameters:
    """太阳无量纲参数取值（对照 BCR4BP 文献标准值）。"""

    def test_sun_mass(self, bcr4bp_system):
        """m_s = GM_sun/GM_EMB（DE440），文献值约 328900.5。"""
        assert bcr4bp_system.sun_mass == pytest.approx(328900.56, rel=1e-4)

    def test_sun_distance(self, bcr4bp_system):
        """a_s = 日地平均距离 / 地月距离，文献值约 389（388.8~389.4 之间）。"""
        assert 388.0 < bcr4bp_system.sun_distance < 390.0

    def test_sun_angular_rate(self, bcr4bp_system):
        """ω_s = 27.32/365.25 - 1 ≈ -0.9252（会合系中逆行，负号）。"""
        assert bcr4bp_system.sun_angular_rate == pytest.approx(-0.92520, rel=1e-4)

    def test_sun_period_is_synodic_month(self, bcr4bp_system):
        """太阳在会合系中的周期 T = 2π/|ω_s| 约为一个会合月（29.53 天）。"""
        period_tu = 2 * np.pi / abs(bcr4bp_system.sun_angular_rate)
        period_days = period_tu * bcr4bp_system.characteristic_time / 86400.0
        assert period_days == pytest.approx(29.53, abs=0.1)

    def test_gravitational_parameter_sun(self, bcr4bp_system):
        """gravitational_parameter('sun') 返回 m_s；primary/secondary 不变。"""
        assert bcr4bp_system.gravitational_parameter("sun") == bcr4bp_system.sun_mass
        assert bcr4bp_system.gravitational_parameter("primary") == pytest.approx(1 - MU)
        assert bcr4bp_system.gravitational_parameter("secondary") == pytest.approx(MU)


class TestSunPosition:
    """太阳解析位置 r_s(t) = a_s·(cos θ, sin θ, 0)，θ = θ0 + ω_s·t。"""

    def test_position_norm_constant(self, bcr4bp_system):
        """双圆近似下 |r_s(t)| ≡ a_s。"""
        for t in [0.0, 0.7, 3.3, 6.7911]:
            assert np.linalg.norm(bcr4bp_system.sun_position(t)) == pytest.approx(
                bcr4bp_system.sun_distance, rel=1e-14
            )

    def test_position_coplanar(self, bcr4bp_system):
        """太阳轨道共面：z 分量恒为 0。"""
        for t in [0.0, 1.1, 4.2]:
            assert bcr4bp_system.sun_position(t)[2] == 0.0

    def test_phase_evolution(self, bcr4bp_system):
        """相位随时间线性演化：r_s(t) 是 r_s(0) 绕 z 轴旋转 ω_s·t。"""
        t = 1.234
        r0 = bcr4bp_system.sun_position(0.0)
        rt = bcr4bp_system.sun_position(t)
        angle = bcr4bp_system.sun_angular_rate * t
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        assert_allclose(rt, rotation @ r0, atol=1e-12)

    def test_initial_phase(self):
        """sun_phase0 控制 t=0 时刻的太阳方向。"""
        system = BCR4BPSystem.earth_moon(sun_phase0=np.pi / 2)
        r0 = system.sun_position(0.0)
        assert r0[0] == pytest.approx(0.0, abs=1e-10)
        assert r0[1] == pytest.approx(system.sun_distance, rel=1e-12)


# =============================================================================
# B. 太阳质量置零 → 退化为 CR3BP
# =============================================================================
class TestZeroSunMassMatchesCR3BP:
    """m_s = 0 时 BCR4BP_Dynamics 应与 CR3BP_Dynamics 逐点一致。"""

    @pytest.fixture
    def zero_sun_dynamics(self, bcr4bp_system):
        bcr4bp_system.sun_mass = 0.0
        return BCR4BP_Dynamics(bcr4bp_system)

    def test_equations_of_motion_pointwise(self, zero_sun_dynamics, cr3bp_system, sample_state):
        """多个时刻、多个状态下运动方程逐点一致（CR3BP 自治，BCR4BP 退化为自治）。"""
        cr3bp = CR3BP_Dynamics(cr3bp_system)
        states = [
            sample_state,
            np.array([0.8, 0.1, 0.05, 0.2, 0.1, -0.03]),
            np.array([-0.5, -0.4, 0.1, 0.0, 0.3, 0.02]),
        ]
        for state in states:
            for t in [0.0, 0.7, 3.3]:
                assert_allclose(
                    zero_sun_dynamics.equations_of_motion(t, state),
                    cr3bp.equations_of_motion(t, state),
                    atol=1e-14,
                )

    def test_jacobian_pointwise(self, zero_sun_dynamics, cr3bp_system, sample_state):
        """m_s = 0 时雅可比 A(t) 与 CR3BP 雅可比一致（太阳项贡献为零）。"""
        cr3bp = CR3BP_Dynamics(cr3bp_system)
        for t in [0.0, 2.5]:
            assert_allclose(
                zero_sun_dynamics.compute_jacobian_A(t, sample_state),
                cr3bp.compute_jacobian_A(sample_state),
                atol=1e-14,
            )

    def test_propagation_matches(self, zero_sun_dynamics, cr3bp_system, sample_state):
        """m_s = 0 时传播结果与 CR3BP 一致（积分容差量级）。"""
        cr3bp = CR3BP_Dynamics(cr3bp_system)
        t_span = (0.0, 1.0)
        t_eval = np.linspace(*t_span, 21)
        res_bcr = zero_sun_dynamics.propagate(sample_state, t_span, t_eval=t_eval)
        res_cr = cr3bp.propagate(sample_state, t_span, t_eval=t_eval)
        assert_allclose(res_bcr["states"], res_cr["states"], atol=1e-9)


# =============================================================================
# C. 雅可比与 STM 有限差分校验
# =============================================================================
class TestJacobianFiniteDiff:
    """compute_jacobian_A 与运动方程的有限差分一致（校验太阳项雅可比）。"""

    def test_jacobian_matches_finite_difference(self, bcr4bp_dynamics, sample_state):
        t = 0.7
        A = bcr4bp_dynamics.compute_jacobian_A(t, sample_state)
        eps = 1e-7
        for i in range(6):
            state_plus = sample_state.copy()
            state_minus = sample_state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            col = (
                bcr4bp_dynamics.equations_of_motion(t, state_plus)
                - bcr4bp_dynamics.equations_of_motion(t, state_minus)
            ) / (2 * eps)
            assert_allclose(A[:, i], col, rtol=1e-6, atol=1e-8)

    def test_sun_jacobian_block_matches_finite_difference(self, bcr4bp_dynamics, sample_state):
        """太阳项对左下 3x3 块的贡献单独校验（对照 ThirdBodyGravity 雅可比公式）。"""
        t = 1.1
        cr3bp_hessian_state = sample_state.copy()
        # 太阳项雅可比 = BCR4BP 左下块 - CR3BP 伪势能 Hessian
        from e2m2e.algorithm.dynamics.potential import pseudo_potential_hessian

        A = bcr4bp_dynamics.compute_jacobian_A(t, sample_state)
        H = pseudo_potential_hessian(MU, *cr3bp_hessian_state[:3])
        J_sun = A[3:, :3] - H

        # 有限差分：只对 sun_acceleration 的位置偏导。
        # 太阳加速度量级 m_s/a_s² ≈ 2，雅可比量级 m_s/a_s³ ≈ 5e-3，
        # eps 取 1e-5 使舍入误差（~eps_machine·|f|/eps）压到 1e-10 以下。
        eps = 1e-5
        for i in range(3):
            pos_plus = sample_state[:3].copy()
            pos_minus = sample_state[:3].copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            col = (
                bcr4bp_dynamics.sun_acceleration(t, pos_plus)
                - bcr4bp_dynamics.sun_acceleration(t, pos_minus)
            ) / (2 * eps)
            assert_allclose(J_sun[:, i], col, rtol=1e-6, atol=1e-10)


class TestSTMFiniteDiff:
    """STM 变分方程（含太阳项）的有限差分校验。"""

    def test_stm_identity_at_t0(self, bcr4bp_dynamics, sample_state):
        """初始 STM 为单位矩阵。"""
        result = bcr4bp_dynamics.propagate(sample_state, (0.0, 0.5), with_stm=True)
        assert_allclose(result["stm"][0], np.eye(6), atol=1e-14)

    def test_stm_matches_finite_difference(self, bcr4bp_dynamics, sample_state):
        """Φ(t,t0)·δx0 与扰动传播的差分一致。

        扰动量 δ=1e-6：线性化误差 O(δ²·|∂f/∂x|) 相对 Φδ 约 1e-5 量级，
        容差取 rtol=1e-4。
        """
        t_span = (0.0, 0.5)
        result = bcr4bp_dynamics.propagate(sample_state, t_span, with_stm=True)
        stm = result["stm"][-1]
        x_final = result["states"][-1]

        delta = 1e-6
        rng = np.random.default_rng(42)
        direction = rng.standard_normal(6)
        direction /= np.linalg.norm(direction)

        perturbed = sample_state + delta * direction
        result_p = bcr4bp_dynamics.propagate(perturbed, t_span)
        dx_fd = result_p["states"][-1] - x_final
        dx_stm = stm @ (delta * direction)

        assert_allclose(dx_stm, dx_fd, rtol=1e-4, atol=1e-9)


# =============================================================================
# D. propagate 接口契约
# =============================================================================
class TestPropagateInterface:
    """propagate 返回形状与时间单调性（与 CR3BP_Dynamics 语义一致）。"""

    def test_states_shape(self, bcr4bp_dynamics, sample_state):
        result = bcr4bp_dynamics.propagate(sample_state, (0.0, 1.0))
        assert result["states"].shape == (len(result["time"]), 6)
        assert np.all(np.diff(result["time"]) > 0)

    def test_stm_shape(self, bcr4bp_dynamics, sample_state):
        t_eval = np.linspace(0.0, 1.0, 11)
        result = bcr4bp_dynamics.propagate(sample_state, (0.0, 1.0), t_eval=t_eval, with_stm=True)
        assert result["stm"].shape == (11, 6, 6)

    def test_time_explicit(self, bcr4bp_dynamics, sample_state):
        """方程显式含时：同一状态不同时刻导数不同（太阳位置随时间变化）。"""
        d0 = bcr4bp_dynamics.equations_of_motion(0.0, sample_state)
        d1 = bcr4bp_dynamics.equations_of_motion(1.0, sample_state)
        assert not np.allclose(d0, d1, atol=1e-10)

    def test_jacobi_not_supported(self, bcr4bp_dynamics, sample_state):
        """BCR4BP 无 Jacobi 积分：with_jacobi=True 与 compute_jacobi_constant 报错。"""
        with pytest.raises(NotImplementedError):
            bcr4bp_dynamics.propagate(sample_state, (0.0, 0.5), with_jacobi=True)
        with pytest.raises(NotImplementedError):
            bcr4bp_dynamics.compute_jacobi_constant(sample_state)


# =============================================================================
# E. 与星历 ForceModel 的短期外推对比（SPICE）
# =============================================================================
@pytest.mark.spice
class TestEphemerisComparison:
    """BCR4BP（双圆近似）与星历 ForceModel（地+月+日点质量）短期传播对比。

    容差依据：两类模型的差异主要来自 BCR4BP/CR3BP 共有的双圆近似——
    真实月球轨道偏心率 e≈0.055（几何量级 e·DU ≈ 2.1e4 km），短期传播
    中体现为轨道误差。实测（DE440s，2025-06-21 历元，3:1 DRO 初值）：
      1 天 |Δr| ≈ 1.1e3 km（CR3BP 对照组同量级 ≈ 1.1e3 km），
      2 天 BCR4BP ≈ 1.36e3 km < CR3BP ≈ 1.70e3 km
      （太阳项开始体现，与星历中太阳摄动 2 天效应 ~3.4e2 km 量级一致）。
    主断言取 1 天传播、容差 2e3 km（约实测值 2 倍，足以抵御内核版本
    与历元差异，同时远小于 e·DU 的几何量级上界 2.1e4 km）。
    次级断言：2 天时 BCR4BP 误差小于 CR3BP 误差（太阳项确实改善了
    与含太阳星历的一致性）。
    """

    def test_short_propagation_matches_ephemeris(
        self,
        spice_manager,
        spice_eph_system,
        spice_syn_j2000,
        reference_epoch,
        sample_state,
    ):
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
        from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
        from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
        from e2m2e.algorithm.forces import ForceModel, PointMassGravity, ThirdBodyGravity

        et0 = spice_manager.utc_to_et(reference_epoch)

        # ForceModel 需要带 coordinate_system 的 system
        if getattr(spice_eph_system, "coordinate_system", None) is None:
            spice_eph_system.coordinate_system = CoordinateSystem(
                axes=ICRSAxes(),
                origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
            )

        # 太阳初相位 θ0：星历太阳位置转到会合系（质心）取方位角
        r_sun_j2000 = np.asarray(spice_eph_system.get_body_position("SUN", et0), dtype=float)
        sun_syn = spice_syn_j2000.j2000_to_synodic(
            np.concatenate([r_sun_j2000, np.zeros(3)]), 0.0, et0
        )
        theta0 = float(np.arctan2(sun_syn[1], sun_syn[0]))

        bcr4bp = BCR4BPSystem.earth_moon(sun_phase0=theta0)
        bcr4bp_dyn = BCR4BP_Dynamics(bcr4bp)
        cr3bp_dyn = CR3BP_Dynamics(spice_syn_j2000.cr3bp_system)

        t_c = bcr4bp.characteristic_time
        j2000_0 = spice_syn_j2000.synodic_to_j2000(sample_state, 0.0, et0)

        def eph_propagate(days: float) -> np.ndarray:
            fm = ForceModel(
                spice_eph_system,
                forces=[
                    PointMassGravity("EARTH"),
                    ThirdBodyGravity("MOON"),
                    ThirdBodyGravity("SUN"),
                ],
            )
            fm.max_step = 600.0
            result = fm.propagate(j2000_0, (et0, et0 + days * 86400.0), max_steps=1_000_000)
            return np.asarray(result["states"][-1])

        def bcr4bp_in_j2000(dyn, days: float) -> np.ndarray:
            tf = days * 86400.0 / t_c
            result = dyn.propagate(sample_state, (0.0, tf))
            return spice_syn_j2000.synodic_to_j2000(result["states"][-1], tf, et0)

        # 主断言：1 天传播，位置差在双圆近似误差量级内
        eph_1d = eph_propagate(1.0)
        err_bcr4bp_1d = float(np.linalg.norm(bcr4bp_in_j2000(bcr4bp_dyn, 1.0)[:3] - eph_1d[:3]))
        assert err_bcr4bp_1d < 2e3, (
            f"BCR4BP 与星历 1 天外推位置差 {err_bcr4bp_1d:.1f} km 超出双圆近似预期量级"
        )

        # 次级断言：2 天时 BCR4BP 比 CR3BP 更接近含太阳的星历
        eph_2d = eph_propagate(2.0)
        err_bcr4bp_2d = float(np.linalg.norm(bcr4bp_in_j2000(bcr4bp_dyn, 2.0)[:3] - eph_2d[:3]))
        err_cr3bp_2d = float(np.linalg.norm(bcr4bp_in_j2000(cr3bp_dyn, 2.0)[:3] - eph_2d[:3]))
        assert err_bcr4bp_2d < err_cr3bp_2d, (
            f"2 天外推 BCR4BP 误差 {err_bcr4bp_2d:.1f} km 应小于 "
            f"CR3BP 误差 {err_cr3bp_2d:.1f} km（太阳项的改善）"
        )


# =============================================================================
# F. 事件检测（Issue #333：BCR4BP 与 CR3BP 事件处理行为一致）
# =============================================================================
class TestBCR4BPEvents:
    """BCR4BP events 检测：验证 scipy 回退与事件语义。

    BCR4BP Rust 路径不支持事件检测；传入 events 时回退 scipy 并发出
    ``warnings.warn``。本类测试验证该回退可用、事件穿越正确、与事后
    检测一致。
    """

    # 初值状态（y=0.05，接近 Earth），保证先下行穿越 y=0 面。
    OFF_PLANE_STATE = np.array([0.8, 0.05, 0.0, 0.0, 0.0, 0.0])

    @pytest.fixture
    def y0_off_plane(self):
        return self.OFF_PLANE_STATE.copy()

    def test_events_no_longer_raise(self, bcr4bp_dynamics, y0_off_plane):
        """传入 events 不应再抛出 NotImplementedError。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        event = section.event(direction=-1)

        result = bcr4bp_dynamics.propagate(y0_off_plane, (0.0, 5.0), events=[event])
        assert "time" in result
        assert "states" in result

    def test_events_emit_warning(self, bcr4bp_dynamics, y0_off_plane):
        """传入 events 时应发出 UserWarning（回退 scipy 提示）。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        event = section.event(direction=-1)

        with pytest.warns(UserWarning, match="回退到 scipy"):
            bcr4bp_dynamics.propagate(y0_off_plane, (0.0, 5.0), events=[event])

    def test_no_warning_without_events(self, bcr4bp_dynamics, sample_state):
        """不传 events 时不应发出警告（正常 Rust / scipy 路径）。"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bcr4bp_dynamics.propagate(sample_state, (0.0, 1.0))

    def test_terminal_event_stops_at_xz_plane(self, bcr4bp_dynamics, y0_off_plane):
        """terminal 事件在首次下行穿越 xz 平面（y=0）时截断积分。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        event = section.event(direction=-1, terminal=True)

        result = bcr4bp_dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[event])

        t_events = result["t_events"][0]
        y_events = result["y_events"][0]
        assert len(t_events) == 1
        # scipy terminal 语义：轨迹末点即事件点
        assert result["time"][-1] == t_events[-1]
        assert result["time"][-1] < 10.0
        # 事件点落在 y=0 截面上，且为下行穿越（vy < 0）
        assert abs(y_events[0][1]) < 1e-10
        assert y_events[0][4] < 0

    def test_direction_filter(self, bcr4bp_dynamics, y0_off_plane):
        """direction 过滤：下行/上行事件各自只记对应方向的穿越。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        down = section.event(direction=-1)
        up = section.event(direction=1)

        result = bcr4bp_dynamics.propagate(y0_off_plane, (0.0, 10.0), events=[down, up])

        t_down, t_up = result["t_events"]
        y_down, y_up = result["y_events"]
        assert len(t_down) > 0
        assert len(t_up) > 0
        assert t_down[0] < t_up[0]
        assert np.all(np.abs(y_down[:, 1]) < 1e-10)
        assert np.all(y_down[:, 4] < 0)
        assert np.all(np.abs(y_up[:, 1]) < 1e-10)
        assert np.all(y_up[:, 4] > 0)

    def test_event_matches_post_hoc_detection(self, bcr4bp_dynamics, y0_off_plane):
        """积分中检测与事后 detect_crossings 的穿越时刻一致。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection, detect_crossings

        section = PoincareSection.plane(axis=1, value=0.0)
        t_eval = np.linspace(0.0, 10.0, 2001)

        result = bcr4bp_dynamics.propagate(
            y0_off_plane,
            (0.0, 10.0),
            t_eval=t_eval,
            events=[section.event(direction=-1)],
        )

        post_hoc = detect_crossings(result["time"], result["states"], section)
        t_post_down = np.array([t for t, state, _ in post_hoc if state[4] < 0])
        t_event_down = np.asarray(result["t_events"][0])

        assert len(t_event_down) == len(t_post_down)
        # 事后检测基于密采样点的线性插值，自身误差 ~dt²；容差取 1e-5
        np.testing.assert_allclose(t_event_down, t_post_down, atol=1e-5)

    def test_single_event_callable_accepted(self, bcr4bp_dynamics, y0_off_plane):
        """events 可传单个 callable（scipy 风格），自动包装为列表。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        result = bcr4bp_dynamics.propagate(
            y0_off_plane,
            (0.0, 10.0),
            events=section.event(direction=-1),
        )
        assert len(result["t_events"]) == 1
        assert len(result["t_events"][0]) > 0

    def test_events_with_stm(self, bcr4bp_dynamics, y0_off_plane):
        """STM 增广传播下事件函数接收 42 维状态；section.event 自动截取前 6 维。"""
        from e2m2e.algorithm.manifold.sections import PoincareSection

        section = PoincareSection.plane(axis=1, value=0.0)
        event = section.event(direction=-1, terminal=True)

        result = bcr4bp_dynamics.propagate(
            y0_off_plane,
            (0.0, 10.0),
            with_stm=True,
            events=[event],
        )

        assert len(result["t_events"][0]) == 1
        # y_events 携带增广状态（6 + 36 = 42 维）
        assert result["y_events"][0].shape == (1, 42)
        assert result["stm"].shape[1:] == (6, 6)
        assert result["time"][-1] == result["t_events"][0][-1]

    def test_no_events_no_event_keys(self, bcr4bp_dynamics, sample_state):
        """不传 events 时返回字典不含事件键（保持原契约）。"""
        result = bcr4bp_dynamics.propagate(sample_state, (0.0, 1.0))
        assert "t_events" not in result
        assert "y_events" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
