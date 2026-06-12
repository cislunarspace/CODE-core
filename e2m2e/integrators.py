"""Public Python shim for the Rust integrator extension."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from e2m2e._integrators import RkMethod
from e2m2e._integrators import rk_step as _rk_step

__all__ = ["rk_step", "RkMethod"]


def rk_step(
    method: RkMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
):
    """Take a single Runge-Kutta step using the Rust integrator core.

    Parameters
    ----------
    method : RkMethod
        Runge-Kutta method to use.
    t : float
        Current time.
    y : array_like
        Current state vector.
    h : float
        Step size to attempt. Must be positive.
    tol : float
        Error tolerance for step-size suggestion. Must be positive.
    f : callable
        Right-hand side function ``f(t, y) -> dy/dt``. It receives ``y`` as a
        NumPy ndarray and should return a NumPy ndarray of the same length.

    Returns
    -------
    StepResult
        Named result with ``y_new``, ``error``, and ``h_next`` attributes.
    """
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return result.tolist()

    return _rk_step(method, t, y.tolist(), h, tol, _adapt)
