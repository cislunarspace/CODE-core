"""CR3BP 周期轨道生成：``design_orbit`` 链路的初猜段，按族拆分。

原单文件 ``cr3bp_orbits.py`` 按轨道族拆为子模块：``walk``（族行走、
错误类型与共享度量）、``dro``、``ro``、``halo``、``nrho``、``axial``、
``lissajous``、``lyapunov``、``triangular``（SPO/LPO/Horseshoe）。
本包入口 eager re-export 全部公开设计入口，外部导入面与拆分前一致。
"""

from .axial import design_axial, design_axial_family
from .dro import design_dpo, design_dro, design_dro_family
from .halo import design_halo, design_halo_family
from .lissajous import design_lissajous, design_lissajous_family
from .lyapunov import design_lyapunov
from .nrho import design_nrho, design_nrho_family
from .ro import design_ro, design_ro_family
from .triangular import (
    design_horseshoe,
    design_horseshoe_family,
    design_lpo,
    design_lpo_family,
    design_spo,
    design_spo_family,
    design_triangular,
)
from .walk import Cr3bpOrbitError, earth_moon_system

__all__ = [
    "Cr3bpOrbitError",
    "design_axial",
    "design_axial_family",
    "design_dpo",
    "design_dro",
    "design_dro_family",
    "design_halo",
    "design_halo_family",
    "design_horseshoe",
    "design_horseshoe_family",
    "design_lissajous",
    "design_lissajous_family",
    "design_lpo",
    "design_lpo_family",
    "design_lyapunov",
    "design_nrho",
    "design_nrho_family",
    "design_ro",
    "design_ro_family",
    "design_spo",
    "design_spo_family",
    "design_triangular",
    "earth_moon_system",
]
