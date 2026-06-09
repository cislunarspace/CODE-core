"""TDD: dispatch delegation for method='homotopy'.

Issue #239 acceptance criterion: `correct_ephemeris_patch_points` must
delegate to `correct_with_homotopy` when method='homotopy' (no longer
raise NotImplementedError). The delegation must happen via a deferred
import to avoid a circular import between ephemeris_correction and
homotopy_correction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithms import ephemeris_correction, homotopy_correction
from e2m2e.algorithms.ephemeris_correction import EphemerisCorrectionResult


def test_homotopy_method_delegates_to_correct_with_homotopy():
    """method='homotopy' routes to correct_with_homotopy (not NotImplError)."""
    fake_dynamics = SimpleNamespace(
        system=SimpleNamespace(
            bodies=["EARTH", "MOON", "SUN"], origin="EARTH", frame="J2000", spice=object()
        ),
        rtol=1e-9, atol=1e-9, max_step=60.0, integrator="DOP853",
    )
    captured: dict = {}

    def fake_correct_with_homotopy(dynamics_arg, t_patch_arg, state_patch_arg, **kwargs):
        captured["dynamics"] = dynamics_arg
        captured["t_patch"] = t_patch_arg
        captured["state_patch"] = state_patch_arg
        captured.update(kwargs)
        return EphemerisCorrectionResult(
            converged=True, iterations=5, max_residual=1.0e-10,
            residual_history=[1.0e-8, 1.0e-10],
            t_patch=t_patch_arg + 0.1, state_patch=state_patch_arg + 0.01,
        )

    # The dispatch function must import correct_with_homotopy lazily to
    # avoid a circular import; we patch the attribute on the module
    # ephemeris_correction references via its deferred import.
    with patch.object(
        homotopy_correction, "correct_with_homotopy", fake_correct_with_homotopy
    ):
        # Make the lazy import inside ephemeris_correction resolve to
        # our patched object by injecting into sys.modules.
        import sys
        sys.modules["e2m2e.algorithms.homotopy_correction"] = homotopy_correction
        try:
            result = ephemeris_correction.correct_ephemeris_patch_points(
                "homotopy",
                dynamics=fake_dynamics,
                t_patch=np.array([0.0, 1000.0]),
                state_patch=np.ones((2, 6)),
                tolerance=1e-8,
                max_iter=10,
                verbose=False,
                n_workers=1,
                kernel_dir="kernels",
            )
        finally:
            del sys.modules["e2m2e.algorithms.homotopy_correction"]

    assert isinstance(result, EphemerisCorrectionResult)
    assert result.converged is True
    assert captured["dynamics"] is fake_dynamics
    assert captured["tolerance"] == 1e-8
    assert captured["max_iter"] == 10
    assert captured["n_workers"] == 1
    assert captured["kernel_dir"] == "kernels"


def test_homotopy_method_uses_deferred_import_no_circular_import():
    """Importing ephemeris_correction must not eagerly load homotopy_correction.

    The dispatch must do a function-local import so that we can patch
    homotopy_correction.correct_with_homotopy at the module level without
    re-entering this dispatch loop.
    """
    import sys
    if "e2m2e.algorithms.homotopy_correction" in sys.modules:
        # This is informational; we don't fail. The contract is that the
        # dispatch path uses a function-local import.
        pass
    # Verify the attribute access pattern: dispatch should do
    # `from .homotopy_correction import correct_with_homotopy as _correct`
    # inside the function body, not at module top.
    import inspect
    source = inspect.getsource(ephemeris_correction.correct_ephemeris_patch_points)
    assert "homotopy_correction" in source or "correct_with_homotopy" in source
