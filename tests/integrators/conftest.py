"""Shared helpers for integrator tests."""

import numpy as np


def kepler_analytic_state(r0: np.ndarray, v0: np.ndarray, t: float, mu: float = 1.0) -> np.ndarray:
    """Propagate a two-body initial state analytically using Kepler's equation.

    Parameters
    ----------
    r0 : np.ndarray
        Initial position vector (length 3).
    v0 : np.ndarray
        Initial velocity vector (length 3).
    t : float
        Propagation time.
    mu : float, optional
        Gravitational parameter. Default is 1.0 for normalized units.

    Returns
    -------
    np.ndarray
        State vector [x, y, z, vx, vy, vz] at time t.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)

    # Specific orbital energy and angular momentum
    energy = 0.5 * v0_norm**2 - mu / r0_norm
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)

    # Semi-major axis
    a = -mu / (2.0 * energy)

    # Eccentricity vector
    e_vec = ((v0_norm**2 - mu / r0_norm) * r0 - np.dot(r0, v0) * v0) / mu
    e = np.linalg.norm(e_vec)

    # Mean motion
    n = np.sqrt(mu / a**3)

    # Handle near-circular orbit directly to avoid division by eccentricity
    if e < 1e-12:
        # Circular motion in the orbital plane
        z_hat = h_vec / h
        x_hat = r0 / r0_norm
        y_hat = np.cross(z_hat, x_hat)

        theta0 = np.arctan2(r0[1], r0[0])
        theta = theta0 + n * t

        r_t = a * (np.cos(theta) * x_hat + np.sin(theta) * y_hat)
        v_t = n * a * (-np.sin(theta) * x_hat + np.cos(theta) * y_hat)
        return np.concatenate([r_t, v_t])

    # Eccentric anomaly from initial conditions
    E0 = np.arccos(np.clip((1.0 - r0_norm / a) / e, -1.0, 1.0))
    if np.dot(r0, v0) < 0.0:
        E0 = 2.0 * np.pi - E0

    # Mean anomaly at epoch
    M0 = E0 - e * np.sin(E0)

    # Mean anomaly at time t
    M = M0 + n * t
    M = np.mod(M, 2.0 * np.pi)

    # Solve Kepler's equation for E
    E = solve_kepler(M, e)

    # True anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    # Orbital plane basis vectors
    z_hat = h_vec / h
    x_hat = e_vec / e if e > 1e-12 else r0 / r0_norm
    y_hat = np.cross(z_hat, x_hat)

    # Position and velocity in orbital plane
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))

    r_orbit = r * np.array([np.cos(nu), np.sin(nu)])
    v_orbit = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu)])

    r_t = r_orbit[0] * x_hat + r_orbit[1] * y_hat
    v_t = v_orbit[0] * x_hat + v_orbit[1] * y_hat

    return np.concatenate([r_t, v_t])


def solve_kepler(M: float, e: float, tol: float = 1e-14, max_iter: int = 100) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if abs(dE) < tol:
            break
    return E


# ---------------------------------------------------------------------------
# LEO J2 problem + adaptive propagation helper (shared by RK method tests).
# ---------------------------------------------------------------------------

# Earth gravitational constants (km, s).
EARTH_MU = 398600.4418  # km^3 / s^2
EARTH_RE = 6378.137  # km, equatorial radius
EARTH_J2 = 1.0826261e-3


def j2_rhs(
    mu: float = EARTH_MU, re: float = EARTH_RE, j2: float = EARTH_J2
):
    """Two-body + J2 acceleration, state vector [x, y, z, vx, vy, vz] in km/s."""

    def f(t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG001
        r = state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        r2 = r_norm**2
        a_2body = -mu * r / r_norm**3
        k = 1.5 * j2 * mu * re**2 / r_norm**5
        z2_over_r2 = r[2] ** 2 / r2
        a_j2 = -k * np.array(
            [
                r[0] * (1.0 - 5.0 * z2_over_r2),
                r[1] * (1.0 - 5.0 * z2_over_r2),
                r[2] * (3.0 - 5.0 * z2_over_r2),
            ]
        )
        return np.concatenate([v, a_2body + a_j2])

    return f


def leo_initial_state(altitude_km: float = 400.0, mu: float = EARTH_MU, re: float = EARTH_RE):
    """Circular LEO initial state in the xy-plane: [r, 0, 0, 0, v, 0] (km/s)."""
    r = re + altitude_km
    v = np.sqrt(mu / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def propagate_rk(method, rhs, y0, t_span, tol: float = 1e-12, h0: float = 1.0):
    """Adaptive propagation via ``rk_step`` from t_span[0] to t_span[1].

    ``tol`` is a *relative* tolerance: the per-step acceptance threshold is
    ``tol * max(1, ||y||)`` so the controller behaves sensibly across state
    scales (e.g. normalised two-body vs LEO in km). Steps whose local error
    estimate exceeds the threshold are rejected and retried at the smaller
    step size suggested by ``rk_step``.

    Returns ``(t_final, y_final, n_steps)``.
    """
    from e2m2e.integrators import rk_step

    t0, tf = t_span
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    h = float(h0)
    n_steps = 0
    while t < tf:
        abs_tol = tol * max(1.0, float(np.linalg.norm(y)))
        h_step = min(h, tf - t)
        result = rk_step(method, t, y, h_step, abs_tol, rhs)
        if result.error <= abs_tol:
            # Accept; cap step growth at 2x to avoid error-overshoot on accept.
            y = np.asarray(result.y_new, dtype=float)
            t += h_step
            h = min(result.h_next, h_step * 2.0)
        else:
            # Reject: leave y/t unchanged, retry with the smaller suggested step.
            h = result.h_next
        n_steps += 1
    return t, y, n_steps


def normalized_leo_j2(altitude_du: float = 400.0 / EARTH_RE, days: float = 1.0):
    """Normalised LEO + J2 problem (length unit = Earth radius, time unit set so mu = 1).

    Normalisation makes ||y|| ~ O(1) so that the relative tolerance in
    :func:`propagate_rk` controls the error directly rather than being scaled
    up by the ~7000 km state magnitude of a km/s formulation.

    Returns ``(rhs, y0, t_span)`` with ``t_span`` in normalised time units
    (1 day ≈ 107.2 TU).
    """
    tu_per_second = np.sqrt(EARTH_MU / EARTH_RE**3)  # TU per second
    t_span = (0.0, days * 86400.0 * tu_per_second)
    rhs = j2_rhs(mu=1.0, re=1.0, j2=EARTH_J2)
    r = 1.0 + altitude_du
    v = np.sqrt(1.0 / r)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    return rhs, y0, t_span
