"""动力学：System（数据上下文）+ Dynamics（传播编排）。

System + Dynamics 都归 algorithm/dynamics/（ADR 0011 迁移，源：
``core/system.py``、``core/cr3bp_system.py``、``core/ephemeris_system.py``、
``core/dynamics.py``、``core/bcr4bp_system.py``、``core/bcr4bp_dynamics.py``、
``core/ephemeris_dynamics.py``、``core/potential.py``）。System 描述模型
（坐标系、单位、引力参数、天体列表），Dynamics 用 system 传播（模板方法
模式 ADR 0002）。标准参数数据（μ/DU/TU/平动点值）在
``data/templates/systems.py``。

``CR3BP_System``/``CR3BP_Dynamics`` 等类名保持原样（三体问题文献惯例，
不统一 PascalCase/snake_case，ADR 0011 规则 3）。
"""

from __future__ import annotations

from .bcr4bp_dynamics import BCR4BP_Dynamics
from .bcr4bp_system import BCR4BPSystem
from .cr3bp_system import CR3BP_System, LibrationPoint
from .dynamics import CR3BP_Dynamics, Dynamics, propagate_state_at_orbit_time
from .ephemeris_dynamics import EphemerisDynamics
from .ephemeris_system import EphemerisSystem
from .potential import pseudo_potential_hessian
from .system import System

__all__ = [
    "System",
    "CR3BP_System",
    "LibrationPoint",
    "Dynamics",
    "CR3BP_Dynamics",
    "EphemerisSystem",
    "EphemerisDynamics",
    "BCR4BPSystem",
    "BCR4BP_Dynamics",
    "propagate_state_at_orbit_time",
    "pseudo_potential_hessian",
]
