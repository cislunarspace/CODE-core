"""Base configuration dataclass for differential correction strategies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorrectionConfig:
    """Immutable configuration for a differential correction strategy.

    Encapsulates all correction parameters that were previously scattered
    across individual setup_* method bodies in DifferentialCorrection.

    Attributes:
        setup_type: Identifier string for the correction setup type.
        symmetry_condition: Symmetry exploited by the correction (e.g. 'x_axis').
        fixed_parameters: Parameter values held constant during correction.
        free_variables: Names of variables the Newton solver adjusts.
        free_variable_indices: State-vector indices corresponding to free variables.
        target_conditions: Constraint names mapped to their target values.
        constraint_indices: State-vector indices for constraint evaluation.
        constraint_weights: Per-constraint weighting factors for the Jacobian.
        constraint_types: Per-constraint classification (e.g. 'equality').
    """

    setup_type: str
    symmetry_condition: str
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    free_variables: list[str] = field(default_factory=list)
    free_variable_indices: list[int] = field(default_factory=list)
    target_conditions: dict[str, float] = field(default_factory=dict)
    constraint_indices: list[int] = field(default_factory=list)
    constraint_weights: dict[str, float] = field(default_factory=dict)
    constraint_types: dict[str, str] = field(default_factory=dict)
