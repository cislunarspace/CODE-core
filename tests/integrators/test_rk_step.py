"""Tests for the Rust integrator extension."""

import math

import numpy as np
import pytest


def test_hello_integrators_smoke():
    """Smoke test: the Rust extension module imports and responds."""
    from e2m2e._integrators import hello_integrators

    assert hello_integrators() == "hello from e2m2e-integrators"


def test_rk_step_imports():
    """The RK stepper symbols are importable from the extension module."""
    from e2m2e._integrators import RkMethod, rk_step

    assert RkMethod.PD45 is not None
    assert callable(rk_step)


def test_rk_step_harmonic_oscillator():
    """PD45 approximates the harmonic oscillator on a small step."""
    from e2m2e._integrators import RkMethod, rk_step

    def f(t, y):
        return np.array([y[1], -y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    h = 1e-4
    result = rk_step(RkMethod.PD45, 0.0, y0, h, 1e-12, f)

    expected = np.array([math.cos(h), -math.sin(h)], dtype=float)
    assert np.linalg.norm(np.asarray(result.y_new) - expected) < 1e-10
    assert result.error < 1e-10
    assert result.h_next > 0.0


def test_rk_step_invalid_step_size_raises():
    """A non-positive step size is rejected before integration."""
    from e2m2e._integrators import RkMethod, rk_step

    def f(t, y):
        return np.zeros_like(y)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 0.0, 1e-12, f)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, -1e-3, 1e-12, f)


def test_rk_step_invalid_tolerance_raises():
    """A non-positive tolerance is rejected before integration."""
    from e2m2e._integrators import RkMethod, rk_step

    def f(t, y):
        return np.zeros_like(y)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 0.0, f)


def test_rk_step_callback_dimension_mismatch_raises():
    """A callback returning the wrong dimension is rejected."""
    from e2m2e._integrators import RkMethod, rk_step

    def f(t, y):
        return np.array([0.0], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        rk_step(RkMethod.PD45, 0.0, y0, 1e-3, 1e-12, f)
