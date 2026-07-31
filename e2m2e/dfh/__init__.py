"""DFH 功能对齐层。

以原生数值路径复现 DFH（``DFH_DAC.exe``）的功能，参数语义对齐 MATLAB
封装与 inputs-dac.txt 输入规范。本包当前覆盖功能码 1（任务轨道设计）
的 DRO/NRHO/Halo 三类与功能码 2（任务轨道控制，轨道保持）；Lissajous/
L4/L5 归属 issue #255，角动量管理归属 #261。
"""

from .control_orbit import ControlOrbitResult, control_orbit
from .cr3bp_orbits import (
    MOON_RADIUS_KM,
    Cr3bpOrbitError,
    design_dro,
    design_halo,
    design_nrho,
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
    "design_nrho",
    "earth_moon_system",
    "default_kernel_dir",
    "load_design_kernels",
    "MOON_RADIUS_KM",
]
