"""轨道保持（轨道控制）。

站保控制律留 Python（领域知识，ADR 0011 迁移，源：``dfh/control_orbit.py``
编排 + ``algorithms/station_keeping/``）：特征点控制（special_point）、目标
点严格/宽松控制（target_point）、误差模型（error_models）、角动量管理
（momentum_management）、三轨道蒙特卡洛（monte_carlo）。``controller.py``
是编排器（读输入星历、选控制律、配双力模型、汇总输出）。
"""

from __future__ import annotations

from .controller import ControlOrbitResult, control_orbit
from .error_models import (
    BoxMullerSampler,
    NavigationErrorModel,
    SrpErrorModel,
    ThrustExecutionError,
)
from .momentum_management import (
    EngineLayout,
    compute_delta_m,
    compute_srp_torque,
    solve_joint_control,
    solve_momentum_unload,
    validate_engine_layout,
)
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .special_point import SpecialPointLaw, StmPropagator
from .target_point import LooseTargetPointLaw, NominalOrbitView, StrictTargetPointLaw

__all__ = [
    "ControlOrbitResult",
    "control_orbit",
    "run_monte_carlo",
    "MonteCarloResult",
    "SpecialPointLaw",
    "StmPropagator",
    "LooseTargetPointLaw",
    "StrictTargetPointLaw",
    "NominalOrbitView",
    "NavigationErrorModel",
    "ThrustExecutionError",
    "SrpErrorModel",
    "BoxMullerSampler",
    "EngineLayout",
    "validate_engine_layout",
    "compute_srp_torque",
    "compute_delta_m",
    "solve_momentum_unload",
    "solve_joint_control",
]
