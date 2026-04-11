"""3D symmetric correction strategies (spatial CR3BP)."""

from __future__ import annotations

from .base import CorrectionConfig


def symmetric_3d_fixed_x0(x0: float) -> CorrectionConfig:
    """Fixed x0 in 3D: free variables are z0, y_dot0, T_half.

    Used for spatially symmetric periodic orbits such as Halo orbits
    with the x-axis symmetry condition.

    Args:
        x0: Fixed initial x coordinate.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_x_fixed_x0",
        symmetry_condition="x_axis",
        fixed_parameters={"x0": x0},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
        constraint_weights={"y": 1.0, "x_dot": 1.0, "z_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality", "z_dot": "equality"},
    )


def symmetric_xz_fixed_x0(x0: float) -> CorrectionConfig:
    """Fixed x0 with XZ-plane symmetry: free variables are z0, y_dot0, T_half.

    Args:
        x0: Fixed initial x coordinate.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_xz_fixed_x0",
        symmetry_condition="xz_plane",
        fixed_parameters={"x0": x0},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )


def symmetric_xz_fixed_z0(z0: float) -> CorrectionConfig:
    """Fixed z0 with XZ-plane symmetry: free variables are x0, y_dot0, T_half.

    Args:
        z0: Fixed initial z coordinate.

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="3D_symmetric_xz_fixed_z0",
        symmetry_condition="xz_plane",
        fixed_parameters={"z0": z0},
        free_variables=["x0", "y_dot0", "T_half"],
        free_variable_indices=[0, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )
