"""DRO（远距逆行轨道族）族生成端到端 + 物理不变量测试。

覆盖 design_dro_family：窗口成员筛选与升序排列、周期闭合、Jacobi 守恒、
月心族契约（成员参数不含平动点）、无平动点请求的算法层参数守卫。
振幅定义与 design_dro 一致（距月心距离 min/max 均值，km）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_dro_family, earth_moon_system
from e2m2e.algorithm.family.cr3bp_orbits import _moon_distance_minmax
from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import OrbitFamily

pytestmark = pytest.mark.orchestration


@pytest.fixture(scope="module")
def dynamics() -> CR3BP_Dynamics:
    return CR3BP_Dynamics(earth_moon_system())


@pytest.fixture(scope="module")
def family(dynamics) -> OrbitFamily:
    """共享一族 DRO(5000~20000 km)，取 3 个成员。"""
    result = design_dro_family(5000.0, 20000.0, n_orbits=3, dynamics=dynamics)
    assert result.status is ConvergenceState.CONVERGED
    assert result.cause is FailureCause.NONE
    assert result.generated_members == len(result.family)
    return result.family


class TestDroFamilyContract:
    def test_family_type_and_member_count(self, family):
        assert family.family_type == "dro"
        assert len(family) == 3

    def test_members_stay_in_window_and_sorted(self, family):
        amplitudes = [orbit.parameters["amplitude_km"] for orbit in family]
        assert all(5000.0 <= amplitude <= 20000.0 for amplitude in amplitudes)
        assert amplitudes == sorted(amplitudes)

    def test_members_carry_no_libration_point(self, family):
        for orbit in family:
            assert "libration_point" not in orbit.parameters

    def test_metadata_records_amplitude_window(self, family):
        assert family.metadata["periodicity"] == "periodic"
        assert family.metadata["amplitude_range_km"] == [5000.0, 20000.0]

    def test_amplitude_definition_matches_design_dro(self, dynamics, family):
        """成员参数中的振幅应与距月心距离 min/max 均值（重新测量）一致。"""
        for orbit in family:
            d_min, d_max = _moon_distance_minmax(dynamics, orbit)
            measured = 0.5 * (d_min + d_max) * dynamics.system.characteristic_length
            assert orbit.parameters["amplitude_km"] == pytest.approx(measured, abs=1.0)


class TestDroFamilyPhysics:
    def test_periodic_closure_and_jacobi_conservation(self, dynamics, family):
        for orbit in family:
            assert orbit.period is not None
            propagated = dynamics.propagate(
                orbit.states[0],
                (0.0, orbit.period),
                t_eval=np.linspace(0.0, orbit.period, 200),
                with_jacobi=True,
            )
            closure = float(np.linalg.norm(propagated["states"][-1] - orbit.states[0], ord=np.inf))
            assert closure <= 1e-8
            jacobi = np.asarray(propagated["jacobi"])
            assert np.max(np.abs(jacobi - jacobi[0])) <= 1e-8

    def test_members_are_planar_retrograde(self, family):
        """DRO 成员应在 xy 平面内且近侧穿越速度 vy0 > 0（旋转系逆行）。"""
        for orbit in family:
            state = orbit.states[0]
            assert abs(state[2]) < 1e-12
            assert abs(state[5]) < 1e-12
            assert state[4] > 0.0


class TestDroFamilyInputValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: design_dro_family(20000.0, 20000.0, n_orbits=1),
            lambda: design_dro_family(30000.0, 20000.0, n_orbits=1),
            lambda: design_dro_family(0.0, 20000.0, n_orbits=1),
            lambda: design_dro_family(1000.0, 20000.0, n_orbits=0),
        ],
    )
    def test_rejects_invalid_parameters_before_numerical_work(self, call):
        with pytest.raises(ValueError):
            call()


class TestDroFamilyStructuredResult:
    def test_single_rust_call_wraps_soft_failure(self, dynamics, monkeypatch):
        """软失败经统一容器保留部分成员；Rust 只调用一次。"""
        from e2m2e.algorithm.family import rust_generation

        calls = 0

        def fake_generate(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["family_type"] == "dro"
            assert kwargs["min_amplitude_km"] == 5000.0
            assert kwargs["max_amplitude_km"] == 20000.0
            return {
                "family_type": "dro",
                "periodicity": "periodic",
                "status": "stagnated",
                "cause": "stagnation_detected",
                "message": "测试停滞",
                "requested_members": 5,
                "generated_members": 1,
                "members": [
                    {
                        "states": [[0.9, 0.0, 0.0, 0.0, 1.0, 0.0]],
                        "times": [0.0],
                        "period": 1.2,
                        "closure_error": 1e-10,
                        "amplitude_km": 8000.0,
                        "perilune_height_km": None,
                        "sampling_fraction": None,
                        "jacobi_drift": None,
                        "newton_iterations": None,
                        "tangent_system_rank": None,
                        "tangent_system_condition": None,
                        "augmented_system_rank": None,
                        "augmented_system_condition": None,
                        "step_size": None,
                    }
                ],
            }

        monkeypatch.setattr(rust_generation, "generate_cr3bp_family_py", fake_generate)
        result = design_dro_family(5000.0, 20000.0, n_orbits=5, dynamics=dynamics)

        assert calls == 1
        assert isinstance(result, FamilyGenerationResult)
        assert result.status is ConvergenceState.STAGNATED
        assert result.cause is FailureCause.STAGNATION_DETECTED
        assert result.generated_members == 1
        assert len(result.family) == 1
        assert result.family.orbits[0].parameters["amplitude_km"] == pytest.approx(8000.0)
        assert "libration_point" not in result.family.orbits[0].parameters
