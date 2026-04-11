"""Halo orbit correction strategies."""

from __future__ import annotations

from .base import CorrectionConfig


def halo_fixed_z0(z0: float, libration_point: int = 1) -> CorrectionConfig:
    """Halo orbit correction with fixed z0 (XZ-plane symmetry).

    Free variables are x0, y_dot0, and T_half.  The libration point
    is recorded as a fixed parameter for bookkeeping.

    Args:
        z0: Fixed initial z coordinate.
        libration_point: Lagrange point number (1=L1, 2=L2).

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="halo_orbit_fixed_z0",
        symmetry_condition="xz_plane",
        fixed_parameters={"z0": z0, "libration_point": libration_point},
        free_variables=["x0", "y_dot0", "T_half"],
        free_variable_indices=[0, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )


def halo_fixed_x0(x0: float, libration_point: int = 1) -> CorrectionConfig:
    """Halo orbit correction with fixed x0 (XZ-plane symmetry).

    Free variables are z0, y_dot0, and T_half.

    Args:
        x0: Fixed initial x coordinate.
        libration_point: Lagrange point number (1=L1, 2=L2).

    Returns:
        CorrectionConfig with the corresponding correction parameters.
    """
    return CorrectionConfig(
        setup_type="halo_orbit_fixed_x0",
        symmetry_condition="xz_plane",
        fixed_parameters={"x0": x0, "libration_point": libration_point},
        free_variables=["z0", "y_dot0", "T_half"],
        free_variable_indices=[2, 4, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0, "z_dot": 0.0},
        constraint_indices=[1, 3, 5],
    )
