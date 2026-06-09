"""Fixed-step homotopy ephemeris correction.

Implements `correct_with_homotopy`, which transitions a patched-point
trajectory from a base body set (e.g. ``["EARTH", "MOON"]``) to a full
ephemeris body set (e.g. ``["EARTH", "MOON", "SUN"]``) by a fixed sequence
of lambda weights. At each lambda step the inner corrector (standard
multiple shooting, or two-level multiple shooting) is invoked with the
previous step's `(t_patch, state_patch)` as initial guess, regardless of
whether that step converged.

The dynamics weighting itself is performed by `HomotopyEphemerisDynamics`,
a subclass of `EphemerisDynamics` whose `_compute_acc_and_jacobian`
linearly interpolates the per-body acceleration/Jacobian between the
base set and the full set:

    a_lambda = a_base + lambda * (a_full - a_base)
    J_lambda = J_base + lambda * (J_full - J_base)

The interpolation is purely in the ephemeris body-grouping dimension; no
coordinate transformation or CR3BP dynamics is involved.

Issue: #239 (MVP), extended in #240/#241.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.ephemeris_dynamics import EphemerisDynamics
from ..core.ephemeris_system import EphemerisSystem
from .ephemeris_correction import EphemerisCorrectionResult
from .multiple_shooting import MultipleShooting
from .two_level_multiple_shooting import TwoLevelMultipleShooting

DEFAULT_LAMBDA_STEPS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


class HomotopyEphemerisDynamics(EphemerisDynamics):
    """Ephemeris dynamics with a single linear mixing weight ``lambda``.

    The acceleration and Jacobian are computed as
    ``a_base + lambda * (a_full - a_base)``. The base set is the subset
    `base_bodies` of the full body list; the full set is the dynamics'
    normal body list. Both share the same spice/origin/frame, so the
    call structure of `_compute_acc_and_jacobian` is reused.
    """

    def __init__(
        self,
        system: EphemerisSystem,
        base_bodies: list[str],
        lambda_weight: float,
    ) -> None:
        # Lambda must be 0 <= lambda <= 1 by contract; it is set on the
        # full-dynamics object so that the parent class's EoM routine
        # reuses our interpolation.
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")
        super().__init__(system)
        self.base_bodies = list(base_bodies)
        self.lambda_weight = float(lambda_weight)

        # Build the parallel base-dynamics object on the same SPICE/origin/frame
        # but with bodies reduced to `base_bodies`. origin must be a member of
        # base_bodies (validation lives in correct_with_homotopy).
        self.base_dynamics = EphemerisDynamics(
            EphemerisSystem(
                bodies=list(base_bodies),
                spice=system.spice,
                origin=system.origin,
                frame=system.frame,
            )
        )
        # Mirror integration parameters onto the base dynamics so that any
        # propagation that goes through it matches tolerances/steps.
        self.base_dynamics.rtol = self.rtol
        self.base_dynamics.atol = self.atol
        self.base_dynamics.max_step = self.max_step
        self.base_dynamics.integrator = self.integrator

    def _compute_acc_and_jacobian(
        self,
        t: float,
        r_sc: npt.NDArray[np.floating],
        need_jacobian: bool = False,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """Linear interpolation of acc/Jacobian between base and full sets."""
        acc_base, jac_base = self.base_dynamics._compute_acc_and_jacobian(
            t, r_sc, need_jacobian
        )
        acc_full, jac_full = super()._compute_acc_and_jacobian(
            t, r_sc, need_jacobian
        )
        lam = self.lambda_weight
        acc = acc_base + lam * (acc_full - acc_base)
        jac: np.ndarray | None = None
        if need_jacobian:
            assert jac_base is not None and jac_full is not None
            jac = jac_base + lam * (jac_full - jac_base)
        return acc, jac


def _validate_base_bodies(dynamics: EphemerisDynamics, base_bodies: list[str]) -> None:
    """Ensure base_bodies is a subset of full bodies and contains origin."""
    full_bodies = list(dynamics.system.bodies)
    full_set = set(full_bodies)
    base_set = set(base_bodies)
    if not base_set.issubset(full_set):
        missing = sorted(base_set - full_set)
        raise ValueError(
            f"base_bodies {base_bodies} must be a subset of system.bodies "
            f"{full_bodies}; unknown: {missing}"
        )
    if dynamics.system.origin not in base_set:
        raise ValueError(
            f"base_bodies {base_bodies} must include origin "
            f"{dynamics.system.origin!r}"
        )


def _validate_lambda_steps(lambda_steps: list[float]) -> None:
    if not lambda_steps:
        raise ValueError("lambda_steps must not be empty")
    for value in lambda_steps:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"lambda_steps values must be in [0, 1], got {value}")
    for prev, curr in zip(lambda_steps, lambda_steps[1:]):
        if curr <= prev:
            raise ValueError(
                f"lambda_steps must be strictly increasing, got {lambda_steps}"
            )
    if lambda_steps[-1] != 1.0:
        raise ValueError(
            f"lambda_steps must end at 1.0 (final step is the full dynamics), "
            f"got {lambda_steps[-1]}"
        )


def correct_with_homotopy(
    dynamics: EphemerisDynamics,
    t_patch: np.ndarray,
    state_patch: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
    n_workers: int,
    kernel_dir: str,
    base_bodies: list[str],
    lambda_steps: list[float] | None = None,
    inner_method: str = "standard",
    velocity_tolerance: float | None = None,
    verbose: bool = False,
) -> EphemerisCorrectionResult:
    """Drive a base->full body-set transition via fixed lambda steps.

    Each step invokes the inner corrector (MultipleShooting by default,
    or TwoLevelMultipleShooting when ``inner_method="two_level"``) with
    ``(t_patch, state_patch)`` seeded from the previous step's output.
    Intermediate steps use ``tolerance * 10``; the final ``lambda=1.0``
    step uses the strict tolerance.

    Aggregated ``EphemerisCorrectionResult`` semantics:
      - standard: ``converged`` from last step, ``iterations`` summed,
        ``max_residual`` from last step, ``residual_history`` flattened.
      - two_level: same plus ``velocity_residual`` and
        ``velocity_residual_history`` from the two-level history pairs.
    """
    if inner_method == "homotopy":
        raise ValueError("inner_method='homotopy' is not allowed (would recurse)")
    if inner_method not in ("standard", "two_level"):
        raise ValueError(
            f"unsupported inner_method: {inner_method!r}; "
            "expected 'standard' or 'two_level'"
        )

    steps = list(lambda_steps) if lambda_steps is not None else list(DEFAULT_LAMBDA_STEPS)
    _validate_base_bodies(dynamics, base_bodies)
    _validate_lambda_steps(steps)

    t_work = np.asarray(t_patch, dtype=float).copy()
    state_work = np.asarray(state_patch, dtype=float).copy()
    position_histories: list[float] = []
    velocity_histories: list[float] = []
    iterations_total = 0
    final_converged = False
    final_max_residual = float("inf")
    final_velocity_residual: float | None = None
    last_t = t_work
    last_state = state_work

    for step_index, lam in enumerate(steps):
        step_tol = tolerance if lam == 1.0 else tolerance * 10.0
        step_dynamics = HomotopyEphemerisDynamics(
            system=dynamics.system,
            base_bodies=base_bodies,
            lambda_weight=lam,
        )
        # Mirror integration parameters from the supplied dynamics.
        step_dynamics.rtol = dynamics.rtol
        step_dynamics.atol = dynamics.atol
        step_dynamics.max_step = dynamics.max_step
        step_dynamics.integrator = dynamics.integrator

        try:
            if inner_method == "standard":
                solver = MultipleShooting(
                    dynamics=step_dynamics,
                    n_workers=n_workers,
                    kernel_dir=kernel_dir,
                )
                step_result = solver.correct(
                    t_patch=t_work,
                    state_patch=state_work,
                    var_time=True,
                    max_iter=max_iter,
                    tolerance=step_tol,
                    verbose=verbose,
                )
                position_histories.extend(float(v) for v in step_result.residual_history)
                iterations_total += int(step_result.iterations)
                final_converged = bool(step_result.converged)
                final_max_residual = float(step_result.max_residual)
                last_t = step_result.t_patch
                last_state = step_result.state_patch
            else:  # two_level
                solver = TwoLevelMultipleShooting(step_dynamics)
                vel_tol = velocity_tolerance if velocity_tolerance is not None else 1e-6
                step_result = solver.correct(
                    t_patch=t_work,
                    state_patch=state_work,
                    max_outer_iterations=max_iter,
                    position_tolerance=step_tol,
                    velocity_tolerance=vel_tol,
                    boundary="fixed_endpoints",
                    verbose=verbose,
                )
                pos_hist, vel_hist = _split_residual_history(step_result.residual_history)
                position_histories.extend(pos_hist)
                velocity_histories.extend(vel_hist)
                iterations_total += int(step_result.outer_iterations)
                final_converged = bool(step_result.converged)
                final_max_residual = float(step_result.final_position_residual)
                final_velocity_residual = float(step_result.final_velocity_residual)
                last_t = step_result.t_patch
                last_state = step_result.state_patch
        except Exception as exc:
            raise RuntimeError(
                f"homotopy lambda step {step_index} (lambda={lam}, "
                f"inner_method={inner_method}) failed: {exc}"
            ) from exc

        # Use the latest result as the next step's initial guess, even if
        # the current step did not converge — the next lambda is closer to
        # 1.0 and may pull the trajectory back into the basin.
        t_work = np.asarray(step_result.t_patch, dtype=float).copy()
        state_work = np.asarray(step_result.state_patch, dtype=float).copy()

    return EphemerisCorrectionResult(
        converged=final_converged,
        iterations=iterations_total,
        max_residual=final_max_residual,
        residual_history=position_histories,
        t_patch=last_t,
        state_patch=last_state,
        velocity_residual=final_velocity_residual,
        velocity_residual_history=velocity_histories if velocity_histories else None,
    )


def _split_residual_history(
    residual_history: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Split a two-level ``residual_history`` (pos, vel) into two flat lists."""
    pos: list[float] = []
    vel: list[float] = []
    for pair in residual_history:
        p, v = pair
        pos.append(float(p))
        vel.append(float(v))
    return pos, vel
