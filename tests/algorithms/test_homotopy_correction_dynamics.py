"""TDD: HomotopyEphemerisDynamics lambda-weighted acc/Jacobian semantics.

Issue #240 acceptance criteria:
- FakeSpice implements ``get_gm(body)`` and
  ``get_body_position(target, et, frame, observer)``.
- Use ``EphemerisSystem(spice=fake_spice, ...)`` to build the system.
- Verify lambda=0 → acc/Jacobian == base EphemerisDynamics.
- Verify lambda=1 → acc/Jacobian == full EphemerisDynamics.
- Verify intermediate lambda → linear interpolation
    a_lambda = a_base + lambda * (a_full - a_base)
    J_lambda = J_base + lambda * (J_full - J_base)
- No real SPICE kernel or network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithms.homotopy_correction import HomotopyEphemerisDynamics
from e2m2e.core.ephemeris_dynamics import EphemerisDynamics
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.mbse.data.enums import ReferenceFrame


class FakeSpice:
    """Deterministic SPICE stand-in: small bodies, constant GM, simple analytic positions."""

    _GM = {
        "EARTH": 398600.435507,
        "MOON": 4902.800118,
        "SUN": 1.32712440018e11,
    }

    def get_gm(self, body: str) -> float:
        return self._GM[body.upper()]

    def get_body_position(self, target: str, et: float, frame: str, observer: str) -> np.ndarray:
        # Position is a function of et; distinct observers/bodies get distinct values.
        # For MOON relative to EARTH: roughly circular orbit ~384_400 km.
        # We use a tiny model: position[0] = 384_400 + 100*sin(et/86400) km etc.
        # Determinism and rough realism suffice for testing the linear
        # interpolation contract — we only need distinct, non-degenerate values.
        t = float(et)
        if observer.upper() == "EARTH" and target.upper() == "MOON":
            return np.array([384_400.0, 0.0, 0.0]) + 100.0 * np.sin(t / 86_400.0)
        if observer.upper() == "EARTH" and target.upper() == "SUN":
            return np.array([1.496e8, 0.0, 0.0]) + 1.0e6 * np.cos(t / 86_400.0)
        return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def fake_spice() -> FakeSpice:
    return FakeSpice()


@pytest.fixture
def full_dynamics(fake_spice) -> EphemerisDynamics:
    system = EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=fake_spice,
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )
    return EphemerisDynamics(system=system)


@pytest.fixture
def base_dynamics(fake_spice) -> EphemerisDynamics:
    system = EphemerisSystem(
        bodies=["EARTH", "MOON"],
        spice=fake_spice,
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )
    return EphemerisDynamics(system=system)


def _r_sc() -> np.ndarray:
    # Pick a non-origin point so the third-body terms are not trivially zero.
    return np.array([7000.0, 100.0, 50.0])  # km


def test_lambda_zero_matches_base_dynamics(full_dynamics, base_dynamics):
    hom = HomotopyEphemerisDynamics(
        system=full_dynamics.system,
        base_bodies=["EARTH", "MOON"],
        lambda_weight=0.0,
    )
    r = _r_sc()
    t = 1.0e7  # ET seconds
    acc_h, jac_h = hom._compute_acc_and_jacobian(t, r, need_jacobian=True)
    acc_b, jac_b = base_dynamics._compute_acc_and_jacobian(t, r, need_jacobian=True)
    np.testing.assert_allclose(acc_h, acc_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jac_h, jac_b, rtol=0, atol=1e-12)


def test_lambda_one_matches_full_dynamics(full_dynamics, base_dynamics):
    hom = HomotopyEphemerisDynamics(
        system=full_dynamics.system,
        base_bodies=["EARTH", "MOON"],
        lambda_weight=1.0,
    )
    r = _r_sc()
    t = 1.0e7
    acc_h, jac_h = hom._compute_acc_and_jacobian(t, r, need_jacobian=True)
    acc_f, jac_f = full_dynamics._compute_acc_and_jacobian(t, r, need_jacobian=True)
    np.testing.assert_allclose(acc_h, acc_f, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jac_h, jac_f, rtol=0, atol=1e-12)


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.75])
def test_intermediate_lambda_is_linear_interpolation(
    full_dynamics, base_dynamics, lam
):
    hom = HomotopyEphemerisDynamics(
        system=full_dynamics.system,
        base_bodies=["EARTH", "MOON"],
        lambda_weight=lam,
    )
    r = _r_sc()
    t = 1.0e7
    acc_h, jac_h = hom._compute_acc_and_jacobian(t, r, need_jacobian=True)
    acc_b, jac_b = base_dynamics._compute_acc_and_jacobian(t, r, need_jacobian=True)
    acc_f, jac_f = full_dynamics._compute_acc_and_jacobian(t, r, need_jacobian=True)
    expected_acc = acc_b + lam * (acc_f - acc_b)
    expected_jac = jac_b + lam * (jac_f - jac_b)
    np.testing.assert_allclose(acc_h, expected_acc, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jac_h, expected_jac, rtol=0, atol=1e-12)


def test_lambda_weight_outside_unit_interval_raises(full_dynamics):
    with pytest.raises(ValueError, match="lambda_weight must be in"):
        HomotopyEphemerisDynamics(
            system=full_dynamics.system,
            base_bodies=["EARTH", "MOON"],
            lambda_weight=1.5,
        )
    with pytest.raises(ValueError, match="lambda_weight must be in"):
        HomotopyEphemerisDynamics(
            system=full_dynamics.system,
            base_bodies=["EARTH", "MOON"],
            lambda_weight=-0.1,
        )


def test_homotopy_dynamics_keeps_propagate_working(fake_spice):
    """End-to-end: a short propagation through HomotopyEphemerisDynamics succeeds."""
    system = EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=fake_spice,
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )
    hom = HomotopyEphemerisDynamics(
        system=system, base_bodies=["EARTH", "MOON"], lambda_weight=0.5
    )
    initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    result = hom.propagate(
        initial_state=initial_state,
        t_span=(0.0, 100.0),
        t_eval=np.linspace(0.0, 100.0, 101),
        with_stm=True,
    )
    assert result["states"].shape == (101, 6)
    assert result["stm"].shape == (101, 6, 6)
    # Numerical sanity: position must stay finite
    assert np.all(np.isfinite(result["states"]))
