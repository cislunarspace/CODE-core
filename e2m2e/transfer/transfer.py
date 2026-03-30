"""
Transfer class for DRO-RO transfer trajectory optimization.

This module provides a simplified interface for transfer trajectory optimization
using NLP methods (Cui et al. 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit
from ..core.system import CR3BP_System

from . import transfer_optimization
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    optimize_with_copt,
)

_HAVE_COPT = transfer_optimization.coptpy is not None


DU = 3.84405000e5


@dataclass
class TransferConfig:
    """Configuration for transfer optimization.

    Attributes:
        alpha_min: Minimum tangential velocity ratio
        alpha_max: Maximum tangential velocity ratio
        earth_radius: Earth radius for collision checking (in DU)
        moon_radius: Moon radius for collision checking (in DU)
        use_relaxed_velocity: Whether to use relaxed velocity constraint
        velocity_angle_tol: Velocity angle tolerance for relaxed constraint (radians)
        use_copt: Whether to use COPT optimizer if available
        fallback_to_scipy: Whether to fallback to SciPy if COPT fails
    """

    alpha_min: float = 0.5
    alpha_max: float = 2.5
    earth_radius: float = 200.0 / DU
    moon_radius: float = 100.0 / DU
    use_relaxed_velocity: bool = True
    velocity_angle_tol: float = 0.05
    use_copt: bool = False
    fallback_to_scipy: bool = True


@dataclass
class TransferOptimizationResult:
    """Result of transfer optimization.

    Attributes:
        success: Whether optimization succeeded
        message: Solver message
        departure_state: Departure state [x, y, z, vx, vy, vz]
        departure_alpha: Tangential velocity ratio at departure
        departure_beta: Normal velocity ratio at departure
        insertion_state: Insertion state on RO [x, y, z, vx, vy, vz]
        final_state: Final state after insertion [x, y, z, vx, vy, vz]
        delta_v1: Departure impulse magnitude
        delta_v2: Insertion impulse magnitude
        total_delta_v: Total delta-v (delta_v1 + delta_v2)
        transfer_time: Transfer duration
        t_ins: Insertion time on RO
        transfer_trajectory: Full transfer trajectory [n_steps, 6]
        transfer_trajectory_times: Time values for trajectory [n_steps]
        constraints_violation: Maximum constraint violation
    """

    success: bool = False
    message: str = ""
    departure_state: Optional[np.ndarray] = None
    departure_alpha: float = 0.0
    departure_beta: float = 0.0
    insertion_state: Optional[np.ndarray] = None
    final_state: Optional[np.ndarray] = None
    delta_v1: float = 0.0
    delta_v2: float = 0.0
    total_delta_v: float = 0.0
    transfer_time: float = 0.0
    t_ins: float = 0.0
    transfer_trajectory: Optional[np.ndarray] = None
    transfer_trajectory_times: Optional[np.ndarray] = None
    constraints_violation: float = 0.0


class Transfer:
    """DRO-RO transfer trajectory optimizer.

    Provides a simplified interface for optimizing transfer trajectories
    between DRO (Distant Retrograde Orbit) and RO (Rectilinear Orbit)
    using NLP methods.

    Example:
        >>> from e2m2e.transfer import Transfer, TransferConfig
        >>> from e2m2e.core import CR3BP_System, CR3BP_Dynamics
        >>> from scripts.utils.common import MU
        >>>
        >>> system = CR3BP_System(mu=MU, primary="earth", secondary="moon")
        >>> dynamics = CR3BP_Dynamics(system=system)
        >>> transfer = Transfer(dynamics)
        >>> transfer.set_orbit(start=dro_orbit, end=ro_orbit)
        >>> result = transfer.optimize(
        ...     initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
        ...     alpha_range=(0.5, 2.5),
        ...     use_relaxed_velocity=True,
        ...     velocity_angle_tol=0.05,
        ... )
    """

    def __init__(self, dynamics: CR3BP_Dynamics):
        """Initialize Transfer optimizer.

        Args:
            dynamics: CR3BP dynamics instance for propagation
        """
        self.dynamics = dynamics
        self.system = dynamics.system
        self.mu = self.system.mu

        self._departure_orbit: Optional[Orbit] = None
        self._arrival_orbit: Optional[Orbit] = None
        self._config = TransferConfig()
        self._result: Optional[TransferOptimizationResult] = None

    @property
    def departure_orbit(self) -> Optional[Orbit]:
        """Departure orbit (DRO)."""
        return self._departure_orbit

    @property
    def arrival_orbit(self) -> Optional[Orbit]:
        """Arrival orbit (RO)."""
        return self._arrival_orbit

    @property
    def config(self) -> TransferConfig:
        """Transfer configuration."""
        return self._config

    @property
    def result(self) -> Optional[TransferOptimizationResult]:
        """Latest optimization result."""
        return self._result

    def set_orbit(self, start: Orbit, end: Orbit) -> "Transfer":
        """Set departure and arrival orbits.

        Args:
            start: Departure orbit (DRO)
            end: Arrival orbit (RO)

        Returns:
            self for method chaining
        """
        self._departure_orbit = start
        self._arrival_orbit = end
        return self

    def optimize(
        self,
        initial_guess: Dict[str, float],
        alpha_range: Tuple[float, float],
        departure_state: Optional[np.ndarray] = None,
        t_ins_range: Optional[Tuple[float, float]] = None,
        use_relaxed_velocity: Optional[bool] = None,
        velocity_angle_tol: Optional[float] = None,
    ) -> TransferOptimizationResult:
        """Optimize transfer trajectory.

        Args:
            initial_guess: Initial guess containing 'alpha', 'transfer_time', 't_ins'
            alpha_range: Range for alpha parameter (min, max)
            departure_state: Manual departure state [6], if None auto-samples from DRO
            t_ins_range: Range for insertion time on RO, defaults to full RO period
            use_relaxed_velocity: Override config use_relaxed_velocity
            velocity_angle_tol: Override config velocity_angle_tol

        Returns:
            TransferOptimizationResult with optimization details
        """
        if self._departure_orbit is None or self._arrival_orbit is None:
            raise ValueError("Must call set_orbit() before optimize()")

        if use_relaxed_velocity is None:
            use_relaxed_velocity = self._config.use_relaxed_velocity
        if velocity_angle_tol is None:
            velocity_angle_tol = self._config.velocity_angle_tol

        if departure_state is None:
            departure_state = self._sample_departure_state_from_dro()
        else:
            departure_state = np.asarray(departure_state)

        if t_ins_range is None:
            t0 = self._arrival_orbit.times[0]
            period = self._get_ro_period()
            t_ins_range = (t0, t0 + period)

        ig = NLPOptimizationVariables(
            alpha=initial_guess["alpha"],
            transfer_time=initial_guess["transfer_time"],
            t_ins=initial_guess["t_ins"],
        )

        optimizer = DROTRONLPOptimizer(
            system=self.system,
            dynamics=self.dynamics,
            departure_orbit=self._departure_orbit,
            arrival_orbit=self._arrival_orbit,
            departure_state=departure_state,
        )

        optimizer.alpha_range = alpha_range
        optimizer.earth_radius = self._config.earth_radius
        optimizer.moon_radius = self._config.moon_radius
        optimizer.velocity_angle_tol = velocity_angle_tol
        optimizer.t_ins_range = t_ins_range

        if self._config.use_copt and _HAVE_COPT:
            nlp_result = optimize_with_copt(
                optimizer,
                initial_guess=ig,
                fallback_to_scipy=self._config.fallback_to_scipy,
            )
        else:
            nlp_result = optimizer.optimize(
                initial_guess=ig,
                alpha_range=alpha_range,
                t_ins_range=t_ins_range,
                use_relaxed_velocity_constraint=use_relaxed_velocity,
                velocity_angle_constraint=velocity_angle_tol,
                verbose=False,
            )

        self._result = self._convert_nlp_result(nlp_result, departure_state)
        return self._result

    def _sample_departure_state_from_dro(self) -> np.ndarray:
        """Sample a departure state from the DRO.

        Returns the first state of the DRO orbit.
        """
        if self._departure_orbit is None:
            raise ValueError("Departure orbit not set")

        return self._departure_orbit.states[0].copy()

    def _get_ro_period(self) -> float:
        """Get RO orbit period.

        Returns:
            RO period, or default if not available
        """
        if self._arrival_orbit is None:
            return 10.0

        period = getattr(self._arrival_orbit, "period", None)
        if period is not None:
            return float(period)

        if hasattr(self._arrival_orbit, "times") and len(self._arrival_orbit.times) > 1:
            return float(self._arrival_orbit.times[-1] - self._arrival_orbit.times[0])

        return 10.0

    def _convert_nlp_result(
        self, nlp_result, departure_state: np.ndarray
    ) -> TransferOptimizationResult:
        """Convert NLPOptimizationResult to TransferOptimizationResult.

        Args:
            nlp_result: NLP optimization result from DROTRONLPOptimizer
            departure_state: The actual departure state used

        Returns:
            TransferOptimizationResult
        """
        max_violation = 0.0
        if nlp_result.constraints_violation:
            max_violation = (
                max(nlp_result.constraints_violation.values())
                if nlp_result.constraints_violation
                else 0.0
            )

        return TransferOptimizationResult(
            success=nlp_result.success,
            message=nlp_result.message,
            departure_state=nlp_result.departure_state,
            departure_alpha=nlp_result.alpha,
            departure_beta=0.0,
            insertion_state=nlp_result.insertion_state,
            final_state=nlp_result.final_state,
            delta_v1=nlp_result.delta_v1,
            delta_v2=nlp_result.delta_v2,
            total_delta_v=nlp_result.objective_value,
            transfer_time=nlp_result.transfer_time,
            t_ins=nlp_result.t_ins,
            transfer_trajectory=nlp_result.transfer_trajectory,
            transfer_trajectory_times=nlp_result.transfer_times,
            constraints_violation=max_violation,
        )
