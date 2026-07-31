"""DFH 功能对齐层。

以原生数值路径复现 DFH（``DFH_DAC.exe``）的功能，参数语义对齐 MATLAB
??? inputs-dac.txt ???????????? 1????????
????DRO/NRHO/Halo/Lissajous/L4/L5????? 2??????????????
??????? #261?
"""

from .control_orbit import ControlOrbitResult, control_orbit
from .cr3bp_orbits import (
    MOON_RADIUS_KM,
    Cr3bpOrbitError,
    design_dro,
    design_halo,
    design_lissajous,
    design_nrho,
    design_triangular,
    earth_moon_system,
)
from .design_orbit import (
    DesignNotConvergedError,
    OrbitDesignResult,
    default_kernel_dir,
    design_orbit,
    load_design_kernels,
)

__all__ = [
    "control_orbit",
    "ControlOrbitResult",
    "design_orbit",
    "OrbitDesignResult",
    "DesignNotConvergedError",
    "Cr3bpOrbitError",
    "design_dro",
    "design_halo",
    "design_lissajous",
    "design_nrho",
    "design_triangular",
    "earth_moon_system",
    "default_kernel_dir",
    "load_design_kernels",
    "MOON_RADIUS_KM",
]
