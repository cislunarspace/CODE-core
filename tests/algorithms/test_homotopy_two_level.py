"""TDD: two-level inner method in correct_with_homotopy.

Issue #241 acceptance criteria:
- inner_method="two_level" is supported and reuses
  HomotopyEphemerisDynamics + base_bodies/lambda_steps validation +
  inner_method="homotopy" guard + lambda-step orchestration.
- Per lambda step, TwoLevelMultipleShooting.correct(...) replaces
  MultipleShooting.correct(...).
- The two-level inner dynamics is still pure ephemeris (J2000, ET sec,
  km, km/s) — no CR3BP, no coordinate transformation.
- Aggregated result for the two-level path:
    converged = final step's converged
    iterations = sum of (outer_iterations) per step
    max_residual = final step's final_position_residual
    residual_history = flattened position residual history
    velocity_residual = final step's final_velocity_residual
    velocity_residual_history = flattened velocity residual history
    t_patch/state_patch = final step's t_patch/state_patch
- Standard, two_level, and standard-homotopy paths coexist without
  interference.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithms import ephemeris_correction, homotopy_correction


def _fake_dynamics():
    return SimpleNamespace(
        system=SimpleNamespace(
            bodies=["EARTH", "MOON", "SUN"],
            origin="EARTH",
            frame="J2000",
            spice=object(),
        ),
        rtol=1e-9, atol=1e-9, max_step=60.0, integrator="DOP853",
    )


def test_two_level_inner_method_is_dispatched_to_two_level_solver():
    """inner_method='two_level' uses TwoLevelMultipleShooting, not MultipleShooting."""
    t_patch = np.array([0.0, 100.0, 200.0])
    state_patch = np.ones((3, 6))
    captured: list[dict] = []

    class FakeTwoLevelMS:
        def __init__(self, dynamics):
            self.dynamics = dynamics

        def correct(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                converged=True, status="converged",
                outer_iterations=2, level1_iterations=[1, 1],
                final_position_residual=1.0e-3,
                final_velocity_residual=1.0e-6,
                per_patch_position_residual=np.array([0.5e-3, 0.5e-3]),
                per_patch_velocity_residual=np.array([0.5e-6, 0.5e-6]),
                residual_history=[(1.0e-2, 1.0e-5), (1.0e-3, 1.0e-6)],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "TwoLevelMultipleShooting", FakeTwoLevelMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-3,  # 1e-3 is the two-level default position tolerance
            max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 1.0],
            inner_method="two_level",
        )

    # Two steps, two solver calls
    assert len(captured) == 2
    # All kwargs forwarded (we use the default velocity_tolerance; just check shape)
    assert "position_tolerance" in captured[0]
    assert "velocity_tolerance" in captured[0]
    assert captured[0]["boundary"] == "fixed_endpoints"

    # Aggregated result follows the two-level aggregation spec
    assert result.converged is True
    assert result.iterations == 4  # 2 outer iters * 2 steps
    assert result.max_residual == 1.0e-3
    # position residuals: only the position component of residual_history
    assert result.residual_history == [1.0e-2, 1.0e-3, 1.0e-2, 1.0e-3]
    assert result.velocity_residual == 1.0e-6
    assert result.velocity_residual_history == [1.0e-5, 1.0e-6, 1.0e-5, 1.0e-6]
    # Final step's t_patch / state_patch returned
    np.testing.assert_allclose(result.t_patch, t_patch + 0.2)  # 2 steps * 0.1
    np.testing.assert_allclose(result.state_patch, state_patch + 0.02)


def test_two_level_dispatches_via_caller_with_inner_method_kwarg():
    """correct_ephemeris_patch_points(method='homotopy', inner_method='two_level')

    must propagate inner_method to correct_with_homotopy.
    """
    fake_dynamics = _fake_dynamics()
    captured: dict = {}

    def fake_correct_with_homotopy(dynamics_arg, t_patch_arg, state_patch_arg, **kwargs):
        captured.update(kwargs)
        return ephemeris_correction.EphemerisCorrectionResult(
            converged=True, iterations=1, max_residual=1.0e-3,
            residual_history=[1.0e-3], velocity_residual=1.0e-6,
            velocity_residual_history=[1.0e-6],
            t_patch=t_patch_arg, state_patch=state_patch_arg,
        )

    with patch.object(homotopy_correction, "correct_with_homotopy", fake_correct_with_homotopy):
        import sys
        saved = sys.modules.get("e2m2e.algorithms.homotopy_correction")
        sys.modules["e2m2e.algorithms.homotopy_correction"] = homotopy_correction
        try:
            ephemeris_correction.correct_ephemeris_patch_points(
                "homotopy",
                dynamics=fake_dynamics,
                t_patch=np.array([0.0, 1.0, 2.0]),
                state_patch=np.zeros((3, 6)),
                tolerance=1e-3, max_iter=5, verbose=False,
                n_workers=1, kernel_dir="k",
                inner_method="two_level",
            )
        finally:
            if saved is None:
                sys.modules.pop("e2m2e.algorithms.homotopy_correction", None)
            else:
                sys.modules["e2m2e.algorithms.homotopy_correction"] = saved

    assert captured["inner_method"] == "two_level"


def test_two_level_rejects_recursive_homotopy_inner_method():
    """inner_method='homotopy' is still rejected (no recursion) on the two-level path."""
    with pytest.raises(ValueError, match="inner_method='homotopy' is not allowed"):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=np.array([0.0, 100.0, 200.0]),
            state_patch=np.ones((3, 6)),
            tolerance=1e-3, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            inner_method="homotopy",
            lambda_steps=[0.5, 1.0],
        )


def test_two_level_uses_velocity_tolerance_default():
    """When velocity_tolerance is not provided to two_level inner, default is 1e-6."""
    t_patch = np.array([0.0, 100.0, 200.0])
    state_patch = np.ones((3, 6))
    velocity_tolerances: list[float] = []

    class FakeTwoLevelMS:
        def __init__(self, dynamics):
            pass

        def correct(self, **kwargs):
            velocity_tolerances.append(kwargs["velocity_tolerance"])
            return SimpleNamespace(
                converged=True, status="converged", outer_iterations=1,
                level1_iterations=[1], final_position_residual=1.0e-3,
                final_velocity_residual=1.0e-6,
                per_patch_position_residual=np.array([0.5e-3, 0.5e-3]),
                per_patch_velocity_residual=np.array([0.5e-6, 0.5e-6]),
                residual_history=[(1.0e-3, 1.0e-6)],
                t_patch=kwargs["t_patch"], state_patch=kwargs["state_patch"],
            )

    with patch.object(homotopy_correction, "TwoLevelMultipleShooting", FakeTwoLevelMS):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-3, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[1.0],
            inner_method="two_level",
        )

    assert velocity_tolerances == [1.0e-6]


def test_two_level_failed_final_step_aggregates_correctly():
    """Nonconverged final two-level step propagates through aggregation."""
    t_patch = np.array([0.0, 100.0, 200.0])
    state_patch = np.ones((3, 6))

    class FakeTwoLevelMS:
        def __init__(self, dynamics):
            pass

        def correct(self, **kwargs):
            return SimpleNamespace(
                converged=False, status="max_iterations", outer_iterations=5,
                level1_iterations=[5, 5], final_position_residual=2.0e-3,
                final_velocity_residual=2.0e-6,
                per_patch_position_residual=np.array([1.0e-3, 1.0e-3]),
                per_patch_velocity_residual=np.array([1.0e-6, 1.0e-6]),
                residual_history=[(1.0e-2, 1.0e-5), (2.0e-3, 2.0e-6)],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "TwoLevelMultipleShooting", FakeTwoLevelMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-3, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 1.0],
            inner_method="two_level",
        )

    assert result.converged is False
    assert result.iterations == 10
    assert result.max_residual == 2.0e-3
    assert result.residual_history == [1.0e-2, 2.0e-3, 1.0e-2, 2.0e-3]
    assert result.velocity_residual == 2.0e-6
    assert result.velocity_residual_history == [1.0e-5, 2.0e-6, 1.0e-5, 2.0e-6]
