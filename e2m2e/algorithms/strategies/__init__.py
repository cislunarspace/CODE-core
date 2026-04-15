"""Differential correction strategy functions.

Each strategy function returns an immutable CorrectionConfig that fully
describes the correction setup (symmetry, free variables, constraints, etc.).
The DifferentialCorrection class delegates to these functions so that
configuration logic is separated from the iterative solver.
"""

from .base import CorrectionConfig
from .halo import halo_fixed_x0, halo_fixed_z0
from .symmetric_2d import (
    symmetric_2d_fixed_t,
    symmetric_2d_fixed_x0,
    symmetric_2d_fixed_y0,
)
from .symmetric_3d import (
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)

__all__ = [
    "CorrectionConfig",
    "symmetric_2d_fixed_x0",
    "symmetric_2d_fixed_t",
    "symmetric_2d_fixed_y0",
    "symmetric_3d_fixed_x0",
    "symmetric_xz_fixed_x0",
    "symmetric_xz_fixed_z0",
    "halo_fixed_z0",
    "halo_fixed_x0",
]
