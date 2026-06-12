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
