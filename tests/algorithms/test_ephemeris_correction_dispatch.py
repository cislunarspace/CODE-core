from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.algorithms import ephemeris_correction
from e2m2e.algorithms.ephemeris_correction import correct_ephemeris_patch_points


def test_standard_method_uses_multiple_shooting_and_normalizes_result(monkeypatch):
    calls: dict[str, object] = {}
    t_patch = np.array([0.0, 1.0])
    state_patch = np.ones((2, 6))

    class FakeMultipleShooting:
        def __init__(self, dynamics, n_workers, kernel_dir):
            calls["init"] = (dynamics, n_workers, kernel_dir)

        def correct(self, **kwargs):
            calls["correct"] = kwargs
            return SimpleNamespace(
                converged=True,
                iterations=3,
                max_residual=1.2,
                residual_history=[3, 2, 1.2],
                t_patch=t_patch + 1,
                state_patch=state_patch + 2,
            )

    monkeypatch.setattr(ephemeris_correction, "MultipleShooting", FakeMultipleShooting)

    result = correct_ephemeris_patch_points(
        "standard",
        dynamics="dynamics",
        t_patch=t_patch,
        state_patch=state_patch,
        tolerance=1e-3,
        max_iter=5,
        verbose=True,
        n_workers=2,
        kernel_dir="kernels",
    )

    assert calls["init"] == ("dynamics", 2, "kernels")
    assert calls["correct"] == {
        "t_patch": t_patch,
        "state_patch": state_patch,
        "var_time": True,
        "max_iter": 5,
        "tolerance": 1e-3,
        "verbose": True,
    }
    assert result.converged is True
    assert result.iterations == 3
    assert result.max_residual == 1.2
    assert result.residual_history == [3.0, 2.0, 1.2]
    assert result.velocity_residual is None


def test_two_level_method_uses_two_level_solver_and_preserves_velocity_diagnostics(monkeypatch):
    calls: dict[str, object] = {}
    t_patch = np.array([0.0, 1.0])
    state_patch = np.ones((2, 6))

    class FakeTwoLevelMultipleShooting:
        def __init__(self, dynamics):
            calls["init"] = dynamics

        def correct(self, **kwargs):
            calls["correct"] = kwargs
            return SimpleNamespace(
                converged=False,
                outer_iterations=4,
                final_position_residual=2.5,
                final_velocity_residual=0.4,
                residual_history=[(5, 1), (2.5, 0.4)],
                t_patch=t_patch + 2,
                state_patch=state_patch + 3,
            )

    monkeypatch.setattr(
        ephemeris_correction,
        "TwoLevelMultipleShooting",
        FakeTwoLevelMultipleShooting,
    )

    result = correct_ephemeris_patch_points(
        "two_level",
        dynamics="dynamics",
        t_patch=t_patch,
        state_patch=state_patch,
        tolerance=1e-3,
        velocity_tolerance=1e-6,
        max_iter=6,
        verbose=True,
        n_workers=2,
        kernel_dir="kernels",
    )

    assert calls["init"] == "dynamics"
    assert calls["correct"] == {
        "t_patch": t_patch,
        "state_patch": state_patch,
        "max_outer_iterations": 6,
        "position_tolerance": 1e-3,
        "velocity_tolerance": 1e-6,
        "boundary": "fixed_endpoints",
        "verbose": True,
    }
    assert result.converged is False
    assert result.iterations == 4
    assert result.max_residual == 2.5
    assert result.residual_history == [5.0, 2.5]
    assert result.velocity_residual == 0.4
    assert result.velocity_residual_history == [1.0, 0.4]


def test_homotopy_method_delegates_to_correct_with_homotopy(monkeypatch):
    """method='homotopy' routes to correct_with_homotopy (Issue #239)."""
    from e2m2e.algorithms import homotopy_correction

    captured: dict = {}
    t_patch = np.array([0.0, 1.0])
    state_patch = np.ones((2, 6))

    def fake(dynamics_arg, t_patch_arg, state_patch_arg, **kwargs):
        captured["dynamics"] = dynamics_arg
        captured["t_patch"] = t_patch_arg
        captured["state_patch"] = state_patch_arg
        captured.update(kwargs)
        return ephemeris_correction.EphemerisCorrectionResult(
            converged=True, iterations=2, max_residual=1.0e-9,
            residual_history=[1.0e-6, 1.0e-9],
            t_patch=t_patch_arg, state_patch=state_patch_arg,
        )

    monkeypatch.setattr(
        homotopy_correction, "correct_with_homotopy", fake,
    )

    # Inject patched module so the dispatch's lazy import resolves to it
    import sys
    saved = sys.modules.get("e2m2e.algorithms.homotopy_correction")
    sys.modules["e2m2e.algorithms.homotopy_correction"] = homotopy_correction
    try:
        result = correct_ephemeris_patch_points(
            "homotopy",
            dynamics="dynamics",
            t_patch=t_patch,
            state_patch=state_patch,
            tolerance=1e-8,
            max_iter=5,
            verbose=False,
            n_workers=1,
            kernel_dir="kernels",
        )
    finally:
        if saved is None:
            sys.modules.pop("e2m2e.algorithms.homotopy_correction", None)
        else:
            sys.modules["e2m2e.algorithms.homotopy_correction"] = saved

    assert captured["dynamics"] == "dynamics"
    assert captured["tolerance"] == 1e-8
    assert captured["max_iter"] == 5
    assert captured["n_workers"] == 1
    assert captured["kernel_dir"] == "kernels"
    assert result.converged is True
