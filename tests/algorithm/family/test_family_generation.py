"""多族族生成入口（#428）的端到端契约测试。

覆盖 design_{nrho,axial,lissajous,spo,lpo,horseshoe}_family：
成员数量上限、状态/时间数组形状、周期闭合、Jacobi 守恒、族振幅语义，
以及 Lissajous 拟周期族的显式标注（契约 A：OrbitFamily 统一容器 +
periodicity 标注，不误称严格周期族）。DRO 族（#502）的专项契约见
test_dro_family.py。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import (
    design_axial_family,
    design_dro_family,
    design_halo_family,
    design_horseshoe_family,
    design_lissajous_family,
    design_lpo_family,
    design_nrho_family,
    design_spo_family,
    earth_moon_system,
)
from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.data.templates import ConvergenceState, FailureCause, LibrationPoint
from e2m2e.data.types.orbit import OrbitFamily

pytestmark = pytest.mark.orchestration


@pytest.fixture(scope="module")
def dynamics() -> CR3BP_Dynamics:
    return CR3BP_Dynamics(earth_moon_system())


def _converged_family(result: FamilyGenerationResult) -> OrbitFamily:
    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert result.generated_members == len(result.family)
    return result.family


def _assert_period_closure_and_jacobi(dynamics: CR3BP_Dynamics, family: OrbitFamily) -> None:
    """周期族成员通用契约：闭合误差与一周期内 Jacobi 守恒。"""
    for orbit in family:
        assert orbit.period is not None
        assert orbit.states.ndim == 2 and orbit.states.shape[1] == 6
        assert orbit.times.ndim == 1 and len(orbit.times) == len(orbit.states)
        propagated = dynamics.propagate(
            orbit.states[0],
            (0.0, orbit.period),
            t_eval=np.linspace(0.0, orbit.period, 200),
            with_jacobi=True,
        )
        end_state = propagated["states"][-1]
        closure = float(np.linalg.norm(end_state - orbit.states[0], ord=np.inf))
        assert closure <= 1e-8
        jacobi = np.asarray(propagated["jacobi"])
        assert np.max(np.abs(jacobi - jacobi[0])) <= 1e-8


class TestHaloFamilyCompatibility:
    def test_halo_family_keeps_periodic_contract(self, dynamics):
        family = design_halo_family(1, 3000.0, n_orbits=2, dynamics=dynamics)
        assert family.family_type == "halo"
        assert 1 <= len(family) <= 2
        _assert_period_closure_and_jacobi(dynamics, family)


class TestFamilyInputValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: design_nrho_family(3, 1, 30000.0, n_orbits=1),
            lambda: design_axial_family(2, 1000.0, n_orbits=0),
            lambda: design_lissajous_family(
                2, 1000.0, 2000.0, 0.0, 0.0, n_orbits=1, sampling_mode="grid"
            ),
            lambda: design_spo_family(
                4,
                5000.0,
                20000.0,
                n_orbits=1,
                continuation_direction="sideways",
            ),
            lambda: design_lpo_family(4, 30000.0, 20000.0, n_orbits=1),
            lambda: design_horseshoe_family(2, 50000.0, 100000.0, n_orbits=1),
            lambda: design_dro_family(20000.0, 20000.0, n_orbits=1),
        ],
    )
    def test_rejects_invalid_parameters_before_numerical_work(self, call):
        with pytest.raises(ValueError):
            call()


class TestDroFamilyCompatibility:
    def test_dro_family_keeps_periodic_contract(self, dynamics):
        result = design_dro_family(5000.0, 20000.0, n_orbits=2, dynamics=dynamics)
        family = _converged_family(result)
        assert family.family_type == "dro"
        assert 1 <= len(family) <= 2
        _assert_period_closure_and_jacobi(dynamics, family)


class TestStructuredPartialResult:
    @staticmethod
    def _member(
        state: list[float],
        *,
        period: float | None,
        amplitude_km: float | None = None,
        sampling_fraction: float | None = None,
        point_count: int = 1,
    ) -> dict:
        return {
            "states": [state] * point_count,
            "times": list(np.arange(point_count, dtype=float)),
            "period": period,
            "closure_error": 1e-10 if period is not None else None,
            "amplitude_km": amplitude_km,
            "perilune_height_km": None,
            "sampling_fraction": sampling_fraction,
            "jacobi_drift": 1e-12 if period is not None else None,
            "newton_iterations": 3 if amplitude_km is not None else None,
            "tangent_system_rank": 4 if amplitude_km is not None else None,
            "tangent_system_condition": 10.0 if amplitude_km is not None else None,
            "augmented_system_rank": 5 if amplitude_km is not None else None,
            "augmented_system_condition": 20.0 if amplitude_km is not None else None,
            "step_size": 0.01 if amplitude_km is not None else None,
        }

    def test_planar_pal_stagnation_preserves_partial_family(self, dynamics, monkeypatch):
        from e2m2e.algorithm.family import rust_generation

        calls = 0

        def fake_generate(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["family_type"] == "spo"
            return {
                "family_type": "spo",
                "periodicity": "periodic",
                "status": "stagnated",
                "cause": "stagnation_detected",
                "message": "测试停滞",
                "requested_members": 5,
                "generated_members": 2,
                "members": [
                    self._member(
                        [0.49, 0.86, 0.0, 0.0, 0.01, 0.0], period=6.5, amplitude_km=5000.0
                    ),
                    self._member(
                        [0.48, 0.87, 0.0, 0.0, 0.02, 0.0], period=6.6, amplitude_km=10000.0
                    ),
                ],
            }

        monkeypatch.setattr(rust_generation, "generate_cr3bp_family_py", fake_generate)
        result = design_spo_family(4, 5000.0, 20000.0, n_orbits=5, dynamics=dynamics)

        assert calls == 1
        assert result.status is ConvergenceState.STAGNATED
        assert result.cause is FailureCause.STAGNATION_DETECTED
        assert result.generated_members == 2
        assert len(result.family) == 2

    def test_halo_soft_failure_preserves_partial_family(self, dynamics, monkeypatch):
        from e2m2e.algorithm.family import rust_generation

        monkeypatch.setattr(
            rust_generation,
            "generate_cr3bp_family_py",
            lambda **kwargs: {
                "family_type": "halo",
                "periodicity": "periodic",
                "status": "stagnated",
                "cause": "stagnation_detected",
                "message": "测试停滞",
                "requested_members": 3,
                "generated_members": 1,
                "members": [
                    self._member(
                        [0.85, 0.0, 0.01, 0.0, -0.13, 0.0],
                        period=2.7,
                    )
                ],
            },
        )
        result = design_halo_family(1, 3000.0, n_orbits=3, dynamics=dynamics)

        assert isinstance(result, FamilyGenerationResult)
        assert result.status is ConvergenceState.STAGNATED
        assert result.generated_members == 1
        assert len(result.family) == 1

    def test_lissajous_failure_preserves_completed_samples(self, dynamics, monkeypatch):
        from e2m2e.algorithm.family import rust_generation

        calls = 0

        def fake_generate(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["family_type"] == "lissajous"
            return {
                "family_type": "lissajous",
                "periodicity": "quasi-periodic",
                "status": "stagnated",
                "cause": "stagnation_detected",
                "message": "测试停滞",
                "requested_members": 3,
                "generated_members": 1,
                "members": [
                    self._member(
                        [0.84, 0.0, 0.01, 0.0, 0.1, 0.0],
                        period=3.0,
                        sampling_fraction=1.0 / 3.0,
                        point_count=3,
                    )
                ],
            }

        monkeypatch.setattr(rust_generation, "generate_cr3bp_family_py", fake_generate)
        result = design_lissajous_family(
            2, 2400.0, 7200.0, 0.01, 0.55, n_orbits=3, dynamics=dynamics
        )

        assert calls == 1
        assert result.status is ConvergenceState.STAGNATED
        assert result.cause is FailureCause.STAGNATION_DETECTED
        assert result.generated_members == 1
        assert len(result.family) == 1


class TestSpoLpoFamily:
    def test_spo_family_members_in_amplitude_range(self, dynamics):
        family = _converged_family(
            design_spo_family(4, 5000.0, 20000.0, n_orbits=3, dynamics=dynamics)
        )
        assert isinstance(family, OrbitFamily)
        assert family.family_type == "spo"
        assert family.periodicity == "periodic"
        assert not family.is_quasi_periodic
        assert 1 <= len(family) <= 3
        amplitudes = [o.parameters["amplitude_km"] for o in family]
        assert all(5000.0 <= a <= 20000.0 for a in amplitudes)
        for orbit in family:
            assert orbit.family_type == "spo"
            assert orbit.states.shape == (1, 6)
            assert orbit.closure_error <= 1e-8
            assert orbit.parameters["tangent_system_rank"] == 4
            assert orbit.parameters["augmented_system_rank"] == 5
            assert np.isfinite(orbit.parameters["tangent_system_condition"])
            assert np.isfinite(orbit.parameters["augmented_system_condition"])
            assert orbit.parameters["newton_iterations"] >= 0
            assert orbit.parameters["step_size"] > 0.0
            assert np.allclose(orbit.states[0, (2, 5)], 0.0, atol=1e-12)
        _assert_period_closure_and_jacobi(dynamics, family)

    def test_spo_increase_x0_initial_direction(self, dynamics):
        family = _converged_family(
            design_spo_family(
                4,
                1737.0,
                10000.0,
                n_orbits=2,
                continuation_direction="increase-x0",
                dynamics=dynamics,
            )
        )
        assert len(family) == 2
        assert family.metadata["continuation_direction"] == "increase-x0"

    def test_lpo_family_members_in_amplitude_range(self, dynamics):
        family = _converged_family(
            design_lpo_family(5, 5000.0, 30000.0, n_orbits=3, dynamics=dynamics)
        )
        assert family.family_type == "lpo"
        assert 1 <= len(family) <= 3
        assert all(5000.0 <= o.parameters["amplitude_km"] <= 30000.0 for o in family)
        _assert_period_closure_and_jacobi(dynamics, family)

    def test_horseshoe_family_retags_lpo_large_amplitude_members(self, dynamics):
        family = _converged_family(
            design_horseshoe_family(4, 50000.0, 110000.0, n_orbits=2, dynamics=dynamics)
        )
        assert family.family_type == "horseshoe"
        assert 1 <= len(family) <= 2
        assert all(o.family_type == "horseshoe" for o in family)
        assert all(o.parameters["amplitude_km"] >= 50000.0 for o in family)
        _assert_period_closure_and_jacobi(dynamics, family)


class TestNrhoFamily:
    def test_l2_family_perilune_below_threshold(self, dynamics):
        family = _converged_family(design_nrho_family(2, 2, 30000.0, n_orbits=3, dynamics=dynamics))
        assert family.family_type == "nrho"
        assert family.periodicity == "periodic"
        assert 1 <= len(family) <= 3
        perilunes = [o.parameters["perilune_height_km"] for o in family]
        assert all(p <= 30000.0 for p in perilunes)
        # 固定 x0 向月侧行走：近月点高度沿族单调下降
        assert np.all(np.diff(perilunes) < 0)
        assert all(o.parameters["libration_point"] == 2 for o in family)
        assert all(o.parameters["halo_class"] == 1 for o in family)
        assert all(np.allclose(o.states[0, (1, 3, 5)], 0.0, atol=1e-10) for o in family)
        _assert_period_closure_and_jacobi(dynamics, family)

    def test_l1_family_walks_pal_path(self, dynamics):
        """L1 固定 x0 在折叠点两侧失效，族行走必须走 PAL 路径。"""
        family = _converged_family(design_nrho_family(1, 1, 30000.0, n_orbits=2, dynamics=dynamics))
        assert 1 <= len(family) <= 2
        assert all(o.parameters["perilune_height_km"] <= 30000.0 for o in family)
        assert all(o.parameters["libration_point"] == 1 for o in family)
        assert all(np.allclose(o.states[0, (1, 3, 5)], 0.0, atol=1e-10) for o in family)
        _assert_period_closure_and_jacobi(dynamics, family)


class TestAxialFamily:
    @pytest.fixture(scope="class")
    def family(self, dynamics) -> OrbitFamily:
        return _converged_family(design_axial_family(2, 1500.0, n_orbits=3, dynamics=dynamics))

    def test_members_within_amplitude_cap(self, family):
        assert family.family_type == "axial"
        assert family.periodicity == "periodic"
        assert 1 <= len(family) <= 3
        assert all(o.parameters["amplitude_z_km"] <= 1500.0 for o in family)

    def test_walk_follows_vz0_family_parameter(self, family):
        """Type B 族参数是 vz0：成员 vz0 递增，且 z0=0（x 轴出发）。"""
        vz0s = [o.states[0, 5] for o in family]
        assert np.all(np.diff(vz0s) > 0)
        assert all(np.allclose(o.states[0, (1, 2, 3)], 0.0, atol=1e-12) for o in family)

    def test_period_closure_and_jacobi(self, dynamics, family):
        _assert_period_closure_and_jacobi(dynamics, family)


class TestLissajousFamily:
    def test_quasi_periodic_annotation(self, dynamics):
        family = _converged_family(
            design_lissajous_family(2, 2400.0, 7200.0, 0.01, 0.55, n_orbits=3, dynamics=dynamics)
        )
        assert family.family_type == "lissajous"
        # 契约 A：统一 OrbitFamily 容器 + 显式拟周期标注
        assert family.is_quasi_periodic
        assert family.periodicity == "quasi-periodic"
        assert family.metadata["periodicity"] == "quasi-periodic"
        assert len(family) == 3
        libration_point = dynamics.system.get_libration_point(LibrationPoint.L2)
        characteristic_length = dynamics.system.characteristic_length
        assert characteristic_length is not None
        for orbit in family:
            assert orbit.family_type == "lissajous"
            # 拟周期成员是有界多点轨迹，不是严格周期轨道
            assert not orbit.is_periodic
            assert orbit.states.shape[0] > 1
            assert orbit.states.shape[1] == 6
            assert orbit.times.shape == (len(orbit.states),)
            assert np.all(np.diff(orbit.times) > 0.0)
            assert np.all(np.isfinite(orbit.states))
            distance_km = (
                np.linalg.norm(orbit.states[:, :3] - libration_point, axis=1)
                * characteristic_length
            )
            requested_scale_km = np.hypot(
                orbit.parameters["amplitude_in_km"],
                orbit.parameters["amplitude_out_km"],
            )
            assert np.max(distance_km) < 2.5 * requested_scale_km

    def test_sampling_scales_amplitudes_with_fixed_phases(self, dynamics):
        family = _converged_family(
            design_lissajous_family(2, 2400.0, 7200.0, 0.01, 0.55, n_orbits=3, dynamics=dynamics)
        )
        amps_in = [o.parameters["amplitude_in_km"] for o in family]
        amps_out = [o.parameters["amplitude_out_km"] for o in family]
        # 线性采样：振幅递增且末端命中请求值；相位语义由 design_lissajous 承担
        assert np.all(np.diff(amps_in) > 0)
        assert np.all(np.diff(amps_out) > 0)
        assert amps_in[-1] == pytest.approx(2400.0)
        assert amps_out[-1] == pytest.approx(7200.0)
