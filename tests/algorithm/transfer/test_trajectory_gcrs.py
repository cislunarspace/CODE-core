"""转移弧惯性系表示（#584，ADR 0040 增补）：``trajectory_gcrs_km`` 契约。

gcrs_km 段 = 与 ``trajectory``（会合系质心 km，ADR 0040 §1）逐行对齐的
地心惯性几何（GCRS 约定：地心原点、不旋转轴、物理 km / km/s），共享
``trajectory_times``（时刻数组不双份）。填充规则：

- HMN：两体弧构造系原样（ECI → synodic 显示变换前的地心惯性态）
- LGA/WSB：会合系几何旋回惯性——地心平移 +[μ·DU, 0, 0]、Rz(θ(t))
  （θ = ω·t，ω = 会合平均角速度）、速度加 ω×r 牵连项（真实系间变换，
  区别于 HMN 显示约定的无牵连旋转）
- low_thrust / 搜索零结果：None（无惯性对应段）

真值一律独立构造：转换单测用解析几何（共旋点/惯性静止点）；HMN 用
两体守恒律；LGA/WSB 用合成候选 + 手写逆变换恢复会合几何。
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.results import CandidateSearchResult
from e2m2e.algorithm.transfer import (
    STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    TransferDesignResult,
    _synodic_to_gcrs,
    transfer_orbit,
)
from e2m2e.algorithm.transfer.hohmann import MU_EARTH, R_EARTH, TliParams
from e2m2e.algorithm.transfer.lga import LgaCandidate
from e2m2e.algorithm.transfer.wsb import WsbCandidate
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration

# eci_to_synodic_display 的默认特征常数（HMN 显示约定，ADR 0040 §3）
_HMN_MU_EM = 1.21506683e-2
_HMN_DU_KM = 384400.0

#: 合成候选的已知字段（与 test_maneuver_events.py 同款）
PERILUNE_FRACTION = 0.35
TOF_SEC = 10.0 * 86400.0


def _canonical_system() -> CR3BP_System:
    """惯性换算所用的地月 CR3BP 系统。

    mu 与编排器硬编码值同源（1.21506683e-2——会合几何中地球位于
       [−μ, 0, 0]，逆变换必须用同一个 μ 才能严格可逆），特征尺度取
       DE421 默认。
    """
    return CR3BP_System(mu=1.21506683e-2, primary="Earth", secondary="Moon")._with_default_scales()


def _make_leo_departure_dim(system):
    """典型 LEO 出发态（无量纲会合系），可被真实传播。"""
    r_park = R_EARTH + 200.0
    v_circ = math.sqrt(MU_EARTH / r_park)
    departure_phys = np.array([r_park, 0.0, 0.0, 0.0, v_circ, 0.0])
    return system.physical_to_dimensionless(departure_phys)


def _fake_arrival_arc(system, n: int = 10) -> SimpleNamespace:
    """精化到达弧替身：从近月点延续到到达（形状满足拼接契约即可）。"""
    tu = system.characteristic_time
    remaining_dim = (1.0 - PERILUNE_FRACTION) * (TOF_SEC / tu)
    states = np.tile(np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]), (n, 1))
    times = np.linspace(0.0, remaining_dim * tu, n)
    return SimpleNamespace(states=states, times=times)


@contextmanager
def _ignore_warnings():
    """合成候选不保证真实几何，抑制传播/组装路径的 UserWarning。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _inverse_to_synodic(states_gcrs: np.ndarray, times_sec: np.ndarray, system) -> np.ndarray:
    """手写逆变换（独立参考实现）：GCRS 惯性 → 会合系质心 km。

    r_syn = Rz(−θ)·r_gcrs − [μ·DU, 0, 0]；v_syn = Rz(−θ)·(v_gcrs − ω×r_gcrs)。
    """
    theta = system.mean_motion * np.asarray(times_sec, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    r = states_gcrs[:, :3]
    v = states_gcrs[:, 3:]
    omega = system.mean_motion
    # 牵连项先在惯性分量中扣除，再旋回会合系
    vt = np.column_stack([v[:, 0] + omega * r[:, 1], v[:, 1] - omega * r[:, 0], v[:, 2]])
    out = np.empty_like(states_gcrs)
    out[:, 0] = c * r[:, 0] + s * r[:, 1] - system.mu * system.characteristic_length
    out[:, 1] = -s * r[:, 0] + c * r[:, 1]
    out[:, 2] = r[:, 2]
    out[:, 3] = c * vt[:, 0] + s * vt[:, 1]
    out[:, 4] = -s * vt[:, 0] + c * vt[:, 1]
    out[:, 5] = vt[:, 2]
    return out


# ---------------------------------------------------------------------------
# 转换内核：解析几何真值
# ---------------------------------------------------------------------------


class TestSynodicToGcrsConverter:
    def test_corotating_point_gets_tangential_velocity(self):
        """会合系中静止的点（v_syn=0）在惯性系中共旋：|v| = ω·r，纯切向。"""
        system = _canonical_system()
        du = system.characteristic_length
        r_geo = np.array([0.3 * du, 0.0, 0.0])  # 地心位置，固定在会合系
        r_syn = r_geo - np.array([system.mu * du, 0.0, 0.0])
        times = np.array([0.0, 0.3, 1.7]) * 86400.0
        states = np.tile(np.concatenate([r_syn, np.zeros(3)]), (3, 1))

        gcrs = _synodic_to_gcrs(states, times, system)

        # θ0=0：t=0 时惯性位置即地心原位置
        assert np.allclose(gcrs[0, :3], r_geo, rtol=1e-12)
        # 地心半径守恒（旋转不改长度）
        assert np.allclose(np.linalg.norm(gcrs[:, :3], axis=1), 0.3 * du, rtol=1e-12)
        # 速度纯切向、大小 ω·r（牵连项存在性检验）
        assert np.allclose(
            np.linalg.norm(gcrs[:, 3:], axis=1), system.mean_motion * 0.3 * du, rtol=1e-12
        )
        assert np.allclose(np.einsum("ij,ij->i", gcrs[:, :3], gcrs[:, 3:]), 0.0, atol=1e-9 * du)

    def test_inertially_fixed_point_stays_fixed(self):
        """惯性系中静止的点（v_gcrs=0）：换算后位置恒定、速度为零。"""
        system = _canonical_system()
        du = system.characteristic_length
        p = np.array([0.25 * du, 0.1 * du, 0.05 * du])
        omega_vec = np.array([0.0, 0.0, system.mean_motion])
        times = np.array([0.0, 0.7, 2.9]) * 86400.0
        states = np.empty((len(times), 6))
        for i, t in enumerate(times):
            theta = system.mean_motion * t
            c, s = math.cos(theta), math.sin(theta)
            rot_back = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            states[i, :3] = rot_back @ p - np.array([system.mu * du, 0.0, 0.0])
            states[i, 3:] = rot_back @ (-np.cross(omega_vec, p))

        gcrs = _synodic_to_gcrs(states, times, system)

        assert np.allclose(gcrs[:, :3], p, rtol=1e-12)
        assert np.allclose(gcrs[:, 3:], 0.0, atol=1e-9)

    def test_round_trip_recovers_synodic_states(self):
        """任意状态往返：手写逆变换逐分量恢复会合系输入。"""
        system = _canonical_system()
        du, tu = system.characteristic_length, system.characteristic_time
        rng = np.random.default_rng(584)
        times = np.linspace(0.0, 5.0 * 86400.0, 7)
        scales = np.array([du] * 3 + [du / tu] * 3)
        states = rng.uniform(-1.5, 1.5, (7, 6)) * scales

        gcrs = _synodic_to_gcrs(states, times, system)
        back = _inverse_to_synodic(gcrs, times, system)

        assert np.allclose(back, states, rtol=1e-12, atol=1e-9)

    def test_system_without_scales_raises(self):
        """系统缺 mean_motion（未设特征尺度）时显式报错，不静默错值。"""
        du = _canonical_system().characteristic_length
        states = np.zeros((2, 6))
        bare = SimpleNamespace(mu=0.012, characteristic_length=du)
        with pytest.raises(ValueError, match="mean_motion"):
            _synodic_to_gcrs(states, np.array([0.0, 1.0]), bare)


# ---------------------------------------------------------------------------
# HMN：两体弧构造系原样
# ---------------------------------------------------------------------------


class TestHmnGcrsArc:
    def test_native_inertial_arc(self):
        """HMN 惯性段 = 显示变换前的地心两体弧：端点半径与能量守恒。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        r1 = R_EARTH + 200.0
        r2 = 42164.0
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=r2)

        gcrs = np.asarray(result.trajectory_gcrs_km)
        traj = np.asarray(result.trajectory)
        times = np.asarray(result.trajectory_times)
        assert gcrs.shape == traj.shape == (times.shape[0], 6)
        # 本就是地心系：端点半径直接可查（无需平移）
        assert np.linalg.norm(gcrs[0, :3]) == pytest.approx(r1, rel=1e-6)
        assert np.linalg.norm(gcrs[-1, :3]) == pytest.approx(r2, rel=1e-6)
        # 二体能量守恒（惯性系独立真值：弧线是两体解，显示约定不动它）
        eps = (gcrs[:, 3:] ** 2).sum(axis=1) / 2.0 - MU_EARTH / np.linalg.norm(gcrs[:, :3], axis=1)
        assert np.allclose(eps, eps[0], rtol=1e-9)

    def test_rowwise_norms_match_display_geometry(self):
        """与显示几何同行等长：旋转保范数；HMN 显示无牵连项 → 速度范数相等。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=42164.0)
        traj = np.asarray(result.trajectory)
        gcrs = np.asarray(result.trajectory_gcrs_km)

        geo_syn = traj[:, :3].copy()
        geo_syn[:, 0] += _HMN_MU_EM * _HMN_DU_KM
        assert np.allclose(
            np.linalg.norm(geo_syn, axis=1), np.linalg.norm(gcrs[:, :3], axis=1), rtol=1e-9
        )
        assert np.allclose(
            np.linalg.norm(traj[:, 3:], axis=1), np.linalg.norm(gcrs[:, 3:], axis=1), rtol=1e-9
        )

    def test_state_frame_unchanged(self):
        """惯性段是并行数据系；顶层 state_frame 仍标注会合主几何。"""
        params = TliParams(parking_alt_km=200.0, inclination_deg=0.0)
        result = transfer_orbit("HMN", tli_params=params, target_orbit_radius_km=42164.0)
        assert result.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM


# ---------------------------------------------------------------------------
# LGA：合成候选（确定性真值）
# ---------------------------------------------------------------------------


def _synthetic_lga_candidate(system) -> LgaCandidate:
    tu = system.characteristic_time
    arrival_time_dim = TOF_SEC / tu
    return LgaCandidate(
        departure_phase=1.5,
        out_of_plane_angle=0.0,
        tof_sec=TOF_SEC,
        departure_state=_make_leo_departure_dim(system),
        perilune_state=np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]),
        perilune_alt_km=1500.0,
        perilune_time_dim=PERILUNE_FRACTION * arrival_time_dim,
        arrival_state=np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]),
        dv_departure=3.19,
        dv_arrival=0.87,
        total_dv=4.06,
        jacobi_departure=4.05,
        jacobi_arrival=4.05,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestLgaGcrsArc:
    @pytest.fixture
    def result(self, monkeypatch):
        system = _canonical_system()
        candidate = _synthetic_lga_candidate(system)
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (candidate,), ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.lga._refine_lga_candidate",
            lambda *args, **kwargs: (candidate, _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
            )

    def test_rows_align_and_round_trip_recovers_synodic(self, result):
        """惯性段逐行对齐，手写逆变换恢复会合几何（含牵连项正确性）。"""
        traj = np.asarray(result.trajectory)
        gcrs = np.asarray(result.trajectory_gcrs_km)
        times = np.asarray(result.trajectory_times)
        assert gcrs.shape == traj.shape == (times.shape[0], 6)

        system = _canonical_system()
        back = _inverse_to_synodic(gcrs, times, system)
        assert np.allclose(back, traj, rtol=1e-9, atol=1e-6)

    def test_geocentric_radii_match_rowwise(self, result):
        """地心半径逐行一致：旋转不改地心距（换算未伪造几何）。"""
        traj = np.asarray(result.trajectory)
        gcrs = np.asarray(result.trajectory_gcrs_km)
        system = _canonical_system()
        geo_syn = traj[:, :3].copy()
        geo_syn[:, 0] += system.mu * system.characteristic_length
        assert np.allclose(
            np.linalg.norm(geo_syn, axis=1), np.linalg.norm(gcrs[:, :3], axis=1), rtol=1e-12
        )

    def test_state_frame_unchanged(self, result):
        assert result.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM


# ---------------------------------------------------------------------------
# WSB：合成候选（换算统一用 CR3BP 特征尺度，ADR 0040 §4 尺度差注记）
# ---------------------------------------------------------------------------


def _synthetic_wsb_candidate(system) -> WsbCandidate:
    tu = system.characteristic_time
    arrival_time_dim = TOF_SEC / tu
    return WsbCandidate(
        sun_phase0=0.7,
        departure_phase=2.1,
        tof_sec=TOF_SEC,
        departure_state=_make_leo_departure_dim(system),
        perilune_state=np.array([0.9, 0.1, 0.0, 0.0, 1.2, 0.0]),
        perilune_alt_km=2500.0,
        perilune_time_dim=PERILUNE_FRACTION * arrival_time_dim,
        arrival_state=np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]),
        h2_kepler=-0.5,
        dv_departure=3.19,
        dv_arrival=0.87,
        total_dv=4.06,
        arrival_time_dim=arrival_time_dim,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="synthetic",
    )


class TestWsbGcrsArc:
    @pytest.fixture
    def result(self, monkeypatch):
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem

        system = BCR4BPSystem.earth_moon()
        candidate = _synthetic_wsb_candidate(system)
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_wsb_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (candidate,), ConvergenceState.CONVERGED, FailureCause.NONE, "synthetic"
            ),
        )
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.wsb._refine_wsb_candidate",
            lambda *args, **kwargs: (candidate, _fake_arrival_arc(system)),
        )
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            return transfer_orbit(
                "WSB",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
            )

    def test_rows_align_and_round_trip_recovers_synodic(self, result):
        """惯性段逐行对齐；统一 CR3BP 尺度换算可逆恢复会合几何。"""
        traj = np.asarray(result.trajectory)
        gcrs = np.asarray(result.trajectory_gcrs_km)
        times = np.asarray(result.trajectory_times)
        assert gcrs.shape == traj.shape == (times.shape[0], 6)

        system = _canonical_system()
        back = _inverse_to_synodic(gcrs, times, system)
        assert np.allclose(back, traj, rtol=1e-9, atol=1e-6)

    def test_state_frame_unchanged(self, result):
        assert result.state_frame == STATE_FRAME_SYNODIC_BARYCENTRIC_KM


# ---------------------------------------------------------------------------
# 无惯性段：low_thrust / 搜索零结果
# ---------------------------------------------------------------------------


class TestNoGcrsSegments:
    def test_low_thrust_has_no_gcrs_segment(self, monkeypatch):
        """low_thrust 轨迹是 (M, 7) 力模型状态，无惯性对应段。"""
        import e2m2e.algorithm.transfer as transfer_pkg

        fake_sol = SimpleNamespace(
            states=np.zeros((3, 7)),
            final_mass=990.0,
            fuel_consumed=10.0,
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="synthetic",
            n_iter=5,
            time=np.array([0.0, 1.0, 2.0]),
            segments=(),
        )
        with patch.object(transfer_pkg.LowThrustShooting, "solve_from_qlaw", return_value=fake_sol):
            result = transfer_orbit(
                "low_thrust",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                engine_config=transfer_pkg.EngineConfig(t_max=0.1, isp=3000.0),
                initial_mass=1000.0,
                n_segments=2,
                duration_days=1.0,
                target_state=np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0]),
            )
        assert isinstance(result, TransferDesignResult)
        assert result.trajectory_gcrs_km is None

    def test_lga_zero_result_has_no_gcrs_segment(self, monkeypatch):
        """LGA 搜索零结果：无轨迹亦无惯性段。"""
        monkeypatch.setattr(
            "e2m2e.algorithm.transfer.search_lga_trajectories",
            lambda *args, **kwargs: CandidateSearchResult(
                (), ConvergenceState.INFEASIBLE, FailureCause.NO_INTERSECTION, "无候选"
            ),
        )
        system = _canonical_system()
        target_phys = system.dimensionless_to_physical(np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0]))
        with _ignore_warnings():
            result = transfer_orbit(
                "LGA",
                tli_params=TliParams(parking_alt_km=200.0, inclination_deg=0.0),
                target_ephemeris=target_phys.reshape(1, 6),
            )
        assert result.trajectory is None
        assert result.trajectory_gcrs_km is None
