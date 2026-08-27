"""离散推力工况数据模型与连续油门映射工具测试。

契约要点：数据模型自 geo-nrho 迁入、档位集合参数化、
最短弧约束经合并/切分满足（段密不报错）、与低推力求解器共享
(throttle, θ₁, θ₂) 控制口径。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer.thrust_arcs import (
    DEFAULT_THRUST_LEVELS,
    G0_MPS2,
    ThrustArc,
    ThrustArcSequence,
    angles_from_direction,
    controls_from_sequence,
    direction_from_angles,
    level_from_throttle,
    sequence_from_controls,
)

pytestmark = [pytest.mark.low_thrust, pytest.mark.orchestration]

DT = 900.0  # 段长，小于最短弧，强迫合并


def _controls(throttles: list[float], theta1: float = np.pi / 2) -> np.ndarray:
    return np.array([[t, theta1, 0.0] for t in throttles])


class TestThrustArc:
    def test_validate_rejects_short_arc(self) -> None:
        arc = ThrustArc(0.0, 100.0, 1.0, np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="最短弧"):
            arc.validate(min_duration_s=3600.0)

    def test_validate_rejects_non_unit_burn_direction(self) -> None:
        arc = ThrustArc(0.0, 4000.0, 0.6, np.array([1.0, 1.0, 0.0]))
        with pytest.raises(ValueError, match="单位"):
            arc.validate(min_duration_s=3600.0)

    def test_coast_direction_not_checked_for_unit(self) -> None:
        arc = ThrustArc(0.0, 4000.0, 0.0, np.array([0.0, 0.0, 0.0]))
        arc.validate(min_duration_s=3600.0)

    def test_rejects_inverted_times(self) -> None:
        arc = ThrustArc(100.0, 100.0, 1.0, np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="时间"):
            arc.validate(min_duration_s=10.0)


class TestThrustArcSequence:
    def test_validate_rejects_gap(self) -> None:
        arcs = (
            ThrustArc(0.0, 4000.0, 1.0, np.array([1.0, 0.0, 0.0])),
            ThrustArc(4001.0, 8000.0, 0.0, np.array([1.0, 0.0, 0.0])),
        )
        with pytest.raises(ValueError, match="连续"):
            ThrustArcSequence(arcs).validate(min_duration_s=3600.0)

    def test_properties(self) -> None:
        arcs = (
            ThrustArc(0.0, 4000.0, 1.0, np.array([1.0, 0.0, 0.0])),
            ThrustArc(4000.0, 8000.0, 0.0, np.array([1.0, 0.0, 0.0])),
        )
        seq = ThrustArcSequence(arcs)
        seq.validate(min_duration_s=3600.0)
        assert seq.t_start == 0.0
        assert seq.t_end == 8000.0
        assert seq.total_duration_s == 8000.0

    def test_fuel_kg_uses_engine_params(self) -> None:
        from e2m2e.algorithm.transfer import EngineConfig

        engine = EngineConfig(t_max=0.5, isp=3000.0)
        arcs = (ThrustArc(0.0, 3600.0, 0.6, np.array([1.0, 0.0, 0.0])),)
        seq = ThrustArcSequence(arcs)
        expected = 0.6 * 0.5 * 3600.0 / (3000.0 * G0_MPS2)
        assert seq.fuel_kg(engine) == pytest.approx(expected, rel=1e-12)


class TestLevelFromThrottle:
    def test_nearest_default_levels(self) -> None:
        assert level_from_throttle(0.55, DEFAULT_THRUST_LEVELS) == 0.6
        assert level_from_throttle(0.9, DEFAULT_THRUST_LEVELS) == 1.0
        assert level_from_throttle(0.1, DEFAULT_THRUST_LEVELS) == 0.0

    def test_custom_levels(self) -> None:
        levels = (0.0, 0.5, 1.0)
        assert level_from_throttle(0.55, levels) == 0.5
        assert level_from_throttle(0.8, levels) == 1.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            level_from_throttle(np.nan, DEFAULT_THRUST_LEVELS)


class TestSequenceFromControls:
    def test_uniform_burn_merges_into_single_arc(self) -> None:
        controls = _controls([0.55] * 8)
        seq = sequence_from_controls(np.linspace(0.0, 8 * DT, 9), controls, min_duration_s=3600.0)
        assert len(seq.arcs) == 1
        assert seq.arcs[0].throttle == 0.6
        assert seq.total_duration_s == pytest.approx(8 * DT)

    def test_dense_segments_do_not_raise_and_respect_min_arc(self) -> None:
        # 档位交替、段长远小于最短弧：旧实现直接报错，新实现必须合并
        controls = _controls([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        seq = sequence_from_controls(np.linspace(0.0, 8 * DT, 9), controls, min_duration_s=3600.0)
        seq.validate(min_duration_s=min(3600.0, 8 * DT))
        burn = [a for a in seq.arcs if a.throttle > 0.0]
        assert burn, "合并后应保留点火弧"
        assert seq.total_duration_s == pytest.approx(8 * DT)

    def test_mixed_profile_produces_contiguous_arcs(self) -> None:
        controls = _controls([0.95, 0.9, 0.1, 0.05, 0.6, 0.62])
        seq = sequence_from_controls(100.0 + np.arange(7) * DT, controls, min_duration_s=DT)
        seq.validate(min_duration_s=DT)
        throttles = [a.throttle for a in seq.arcs]
        assert throttles == [1.0, 0.0, 0.6]
        assert seq.t_start == 100.0
        assert seq.t_end == pytest.approx(100.0 + 6 * DT)

    def test_custom_levels_used_in_mapping(self) -> None:
        controls = _controls([0.48] * 4)
        seq = sequence_from_controls(
            np.linspace(0.0, 4 * DT, 5), controls, levels=(0.0, 0.5, 1.0), min_duration_s=DT
        )
        assert all(a.throttle == 0.5 for a in seq.arcs)

    def test_burn_directions_are_unit_vectors(self) -> None:
        controls = np.array([[0.9, 0.3, -0.2]] * 4)
        seq = sequence_from_controls(np.linspace(0.0, 4 * DT, 5), controls, min_duration_s=DT)
        for arc in seq.arcs:
            if arc.throttle > 0.0:
                assert np.linalg.norm(arc.direction) == pytest.approx(1.0, abs=1e-12)

    def test_rejects_bad_control_shape(self) -> None:
        with pytest.raises(ValueError, match="controls"):
            sequence_from_controls(np.array([0.0]), np.zeros((0, 3)), min_duration_s=1.0)

    def test_nonuniform_time_nodes(self) -> None:
        # 非均匀段边界：档位相同的相邻段合并，时间窗保持
        times = np.array([0.0, 100.0, 500.0, 900.0, 2000.0])
        controls = _controls([0.9, 0.9, 0.1, 0.6])
        seq = sequence_from_controls(times, controls, min_duration_s=50.0)
        seq.validate(min_duration_s=50.0, levels=DEFAULT_THRUST_LEVELS)
        assert [a.throttle for a in seq.arcs] == [1.0, 0.0, 0.6]
        assert seq.arcs[0].t_end == 500.0
        assert seq.total_duration_s == pytest.approx(2000.0)

    def test_times_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="times"):
            sequence_from_controls(np.array([0.0, 1.0, 2.0]), _controls([0.5]), min_duration_s=0.1)

    def test_validate_rejects_off_level_throttle(self) -> None:
        arc = ThrustArc(0.0, 4000.0, 0.55, np.array([1.0, 0.0, 0.0]))
        arc.validate(min_duration_s=3600.0)  # 不传 levels 不查档位
        with pytest.raises(ValueError, match="档位"):
            arc.validate(min_duration_s=3600.0, levels=DEFAULT_THRUST_LEVELS)


class TestControlsRoundTrip:
    """sequence_from_controls 与 controls_from_sequence 互为均匀段上的往返。"""

    def test_roundtrip_preserves_snapped_throttle(self) -> None:
        controls = _controls([0.95, 0.9, 0.1, 0.6])
        n = controls.shape[0]
        seq = sequence_from_controls(np.linspace(0.0, n * DT, n + 1), controls, min_duration_s=DT)
        expanded = controls_from_sequence(seq, n)
        assert expanded.shape == (n, 3)
        np.testing.assert_allclose(expanded[:, 0], [1.0, 1.0, 0.0, 0.6], atol=1e-12)

    def test_expanded_directions_match_angles(self) -> None:
        controls = _controls([0.9, 0.9], theta1=0.7)
        seq = sequence_from_controls(np.linspace(0.0, 2 * DT, 3), controls, min_duration_s=DT)
        expanded = controls_from_sequence(seq, 2)
        for row in expanded:
            direction = direction_from_angles(row[1], row[2])
            np.testing.assert_allclose(direction, [np.cos(0.7), np.sin(0.7), 0.0], atol=1e-12)

    def test_angles_direction_roundtrip(self) -> None:
        theta1, theta2 = 0.9, -0.4
        direction = direction_from_angles(theta1, theta2)
        recovered1, recovered2 = angles_from_direction(direction)
        assert recovered1 == pytest.approx(theta1)
        assert recovered2 == pytest.approx(theta2)


@pytest.mark.spice
@pytest.mark.orchestration
class TestEndToEndMapping:
    """端到端验收：连续油门解 → 离散映射 → 重传播 → 终端残差（ADR 0032 决策 5）。

    流程对齐 brief：LowThrustShooting 生成连续油门解，sequence_from_controls
    映射到默认档位（最短弧 = 2 倍段长，强迫合并），controls_from_sequence
    展开后由同一传播链接龙重传播，比对终端状态残差。
    """

    @pytest.fixture
    def earth_ephemeris_system(self, spice_kernel_path):
        from kernel_helpers import load_body_fixed_kernels, unload_kernels

        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
        from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
        from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
        from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
        from e2m2e.data.kernels.manager import SPICEManager

        spice = SPICEManager()
        spice.load_kernel(spice_kernel_path)
        bf_kernels = load_body_fixed_kernels(spice)
        try:
            system = EphemerisSystem(bodies=["EARTH"], spice=spice, origin="EARTH")
            system.coordinate_system = CoordinateSystem(
                axes=ICRSAxes(),
                origin=CelestialBodyOrigin(body="EARTH", spice=spice),
            )
            yield system
        finally:
            unload_kernels(spice, bf_kernels)
            spice.unload_kernel(spice_kernel_path)

    def test_mapped_sequence_terminal_residual(self, earth_ephemeris_system) -> None:
        from e2m2e.algorithm.forces import GravityField
        from e2m2e.algorithm.transfer import EngineConfig, LowThrustShooting
        from e2m2e.data.templates import ConvergenceState

        system = earth_ephemeris_system
        mu = system.gravitational_parameter("EARTH")
        a0 = 6378.137 + 300.0
        v0 = np.sqrt(mu / a0)
        init = np.array([a0, 0.0, 0.0, 0.0, v0, 0.0])

        engine = EngineConfig(t_max=0.5, isp=3000.0)
        n_segments = 8
        duration = 2.0 * 2 * np.pi * np.sqrt(a0**3 / mu)  # 2 圈
        et0 = system.spice.utc_to_et("2025-06-21T11:00:06")

        # 已知连续控制（0.55 油门沿迹）传播得目标；求解器以此收敛出连续油门解
        known_y = np.tile(np.array([0.55, np.pi / 2, 0.0]), n_segments)
        probe = LowThrustShooting(
            system,
            [GravityField("EARTH", degree=0, order=0)],
            engine,
            init,
            initial_mass=1000.0,
            target_state=init.copy(),  # 占位，本实例只用于传播
            t0=et0,
            tf=et0 + duration,
        )
        _, known_states = probe._propagate_chain(known_y)

        shooter = LowThrustShooting(
            system,
            [GravityField("EARTH", degree=0, order=0)],
            engine,
            init,
            initial_mass=1000.0,
            target_state=known_states[-1][:6].copy(),
            t0=et0,
            tf=et0 + duration,
        )
        solution = shooter.solve(n_segments, x0=known_y.copy(), maxiter=30)
        assert solution.status is ConvergenceState.CONVERGED, solution.message

        # 连续解 → 离散映射（最短弧 = 2 倍段长）→ 展开重传播
        controls = np.array(
            [[seg.throttle, *angles_from_direction(seg.direction)] for seg in solution.segments]
        )
        dt = duration / n_segments
        sequence = sequence_from_controls(
            np.linspace(et0, et0 + duration, n_segments + 1), controls, min_duration_s=2 * dt
        )
        sequence.validate(min_duration_s=2 * dt)

        y_discrete = controls_from_sequence(sequence, n_segments).reshape(-1)
        _, discrete_states = shooter._propagate_chain(y_discrete)

        pos_residual = float(np.linalg.norm(discrete_states[-1][:3] - solution.states[-1][:3]))
        vel_residual = float(np.linalg.norm(discrete_states[-1][3:6] - solution.states[-1][3:6]))
        # L1 量级门槛（384 km / 1 m/s）是任务级上界；
        # 本算例实测残差约 0.35 km / 0.0004 m/s，阈值按实测十余倍收紧防回归
        print(f"\n终端残差: {pos_residual:.2f} km, {vel_residual:.4f} m/s")
        assert pos_residual < 5.0, f"位置残差 {pos_residual:.1f} km 超收紧阈值（L1 口径 384 km）"
        assert vel_residual < 0.005, f"速度残差 {vel_residual:.4f} m/s 超收紧阈值（L1 口径 1 m/s）"
