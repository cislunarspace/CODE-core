"""Public Python shim for the Rust integrator extension."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from e2m2e._integrators import MultistepMethod
from e2m2e._integrators import MultistepResult
from e2m2e._integrators import RkMethod
from e2m2e._integrators import rk_step as _rk_step
from e2m2e._integrators import multistep_step as _multistep_step

__all__ = [
    "rk_step",
    "RkMethod",
    "multistep_step",
    "MultistepMethod",
    "MultistepResult",
    "initialize_abm_history",
]


def rk_step(
    method: RkMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
):
    """Take a single Runge-Kutta step using the Rust integrator core.

    The callback ``f`` receives a NumPy ndarray and must return one of the same
    length. Returns a ``StepResult`` with ``y_new``, ``error``, ``h_next``.
    """
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    return _rk_step(method, t, y.tolist(), h, tol, _adapt)


def multistep_step(
    method: MultistepMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    history: list[npt.ArrayLike],
):
    """Take a single multistep predictor-corrector step.

    ``history`` must hold ``method.steps()`` derivative samples (oldest first),
    each the same length as ``y``, at equal spacing ``h``. The callback ``f``
    has the same signature as for :func:`rk_step`. Returns a
    ``MultistepResult`` whose ``history`` is the rolled buffer for the next step.

    The step size is assumed fixed; changing ``h`` requires re-initialising the
    history (see :func:`initialize_abm_history`).
    """
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    hist_lists = [np.asarray(hi, dtype=float).tolist() for hi in history]
    return _multistep_step(method, t, y.tolist(), h, tol, _adapt, hist_lists)


def initialize_abm_history(
    t0: float,
    y0: npt.ArrayLike,
    h: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    n_stages: int = 3,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, list[list[float]]]:
    """Bootstrap the ABM history by running ``n_stages`` RK89 steps.

    The ABM method consumes 4 derivative samples; with the default
    ``n_stages=3`` this returns ``(t0 + 3h, y(3h), [f_0, f_1, f_2, f_3])``.
    The returned history is ready to feed into :func:`multistep_step`.
    """
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    history: list[list[float]] = [np.asarray(f(t, y), dtype=float).tolist()]
    for _ in range(n_stages):
        result = rk_step(RkMethod.RK89, t, y, h, tol, f)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        history.append(np.asarray(f(t, y), dtype=float).tolist())
    return t, y, history
