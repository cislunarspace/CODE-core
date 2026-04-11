"""2D symmetric correction strategies (planar CR3BP)."""

from __future__ import annotations

from .base import CorrectionConfig


def symmetric_2d_fixed_x0(x0: float = 0.0) -> CorrectionConfig:
    """Fixed x0: free variables are y_dot0 and T_half.

    Used for planar symmetric periodic orbits that cross the x-axis
    perpendicularly at both start and half-period.

    Args:
        x0: Fixed initial x coordinate.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_x_fixed_x0",
        symmetry_condition="x_axis",
        fixed_parameters={"x0": x0},
        free_variables=["y_dot0", "T_half"],
        free_variable_indices=[4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 3],
        constraint_weights={"y": 1.0, "x_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality"},
    )


def symmetric_2d_fixed_t(t_half: float) -> CorrectionConfig:
    """Fixed half-period: free variables are x0 and y_dot0.

    Args:
        t_half: Fixed half-period value.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_x_fixed_t",
        symmetry_condition="x_axis",
        fixed_parameters={"T_half": t_half},
        free_variables=["x0", "y_dot0"],
        free_variable_indices=[0, 4],
        target_conditions={"y": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 3],
        constraint_weights={"y": 1.0, "x_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality"},
    )


def symmetric_2d_fixed_y0(y0: float = 0.0) -> CorrectionConfig:
    """Fixed y0 (y-axis symmetric): free variables are x_dot0 and T_half.

    Suitable for resonant orbits that depart from the y-axis.

    Args:
        y0: Fixed initial y coordinate.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="2D_symmetric_y_fixed_y0",
        symmetry_condition="y_axis",
        fixed_parameters={"y0": y0},
        free_variables=["x_dot0", "T_half"],
        free_variable_indices=[3, 6],
        target_conditions={"x": 0.0, "x_dot": 0.0},
        constraint_indices=[0, 3],
        constraint_weights={"x": 1.0, "x_dot": 1.0},
        constraint_types={"x": "equality", "x_dot": "equality"},
    )
