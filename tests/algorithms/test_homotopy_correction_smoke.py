"""TDD: minimal smoke test for correct_with_homotopy.

Verifies the function exists in algorithms.homotopy, accepts the
MVP parameter set, and returns a (placeholder) EphemerisCorrectionResult
object with the documented fields.

Issue #239: 接入标准多重打靶固定步长同伦过渡 MVP.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from e2m2e.algorithms.ephemeris_correction import EphemerisCorrectionResult, homotopy


def test_correct_with_homotopy_returns_ephemeris_correction_result():
    """MVP: correct_with_homotopy exists and returns a result object.

    We monkeypatch MultipleShooting to avoid running real SPICE — the goal
    of this tracer bullet is to confirm the function signature and result
    shape, not to verify orbital convergence.
    """
    t_patch = np.array([0.0, 1000.0])
    state_patch = np.ones((2, 6))

    captured: dict = {}

    class FakeMultipleShooting:
        def __init__(self, dynamics, n_workers, kernel_dir):
            captured["init"] = (dynamics, n_workers, kernel_dir)

        def correct(self, **kwargs):
            captured["correct"] = kwargs
            return SimpleNamespace(
                converged=True,
                outer_iterations=2,
                max_residual=1.0e-9,
                residual_history=[1.0e-6, 1.0e-9],
                t_patch=t_patch + 0.1,
                state_patch=state_patch + 0.01,
            )

    fake_system = SimpleNamespace(
        bodies=["EARTH", "MOON", "SUN"],
        origin="EARTH",
        frame="J2000",
        spice=object(),
    )
    fake_dynamics = SimpleNamespace(
        system=fake_system,
        rtol=1e-9,
        atol=1e-9,
        max_step=60.0,
        integrator="DOP853",
    )

    with patch.object(homotopy, "MultipleShooting", FakeMultipleShooting):
        result = homotopy.correct_with_homotopy(
            dynamics=fake_dynamics,
            t_patch=t_patch,
            state_patch=state_patch,
            tolerance=1e-8,
            max_iter=10,
            n_workers=1,
            kernel_dir="kernels",
            base_bodies=["EARTH", "MOON"],
        )

    assert isinstance(result, EphemerisCorrectionResult)
    # The default inner_method is "standard"; the test stubs a FakeMultipleShooting
    # that converges on the very first call, so we should observe at least one
    # lambda step (0.25 by default) executed.
    assert captured  # the inner solver was instantiated
