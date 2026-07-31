"""轨道保持（轨道控制）。

站保控制律留 Python（领域知识，ADR 0011 迁移，源：``dfh/control_orbit.py``
编排 + ``algorithms/station_keeping/``）：特征点控制（special_point）、目标
点严格/宽松控制（target_point）、误差模型（error_models）、三轨道蒙特卡洛
（monte_carlo）。``controller.py`` 是编排器（读输入星历、选控制律、配双力
模型、汇总输出）。

未实现（对外承诺能力）：角动量管理（原 #261），占位函数抛
``NotImplementedError``。
"""

from __future__ import annotations

from .controller import ControlOrbitResult, control_orbit
from .error_models import (
    BoxMullerSampler,
    NavigationErrorModel,
    SrpErrorModel,
    ThrustExecutionError,
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
]


def momentum_management(
    input_ephemeris: object,
    *,
    engine_layout: object = None,
    **kwargs,
) -> ControlOrbitResult:
    """角动量管理联合控制（原功能码 4-6，原 #261）。

    实现状态：未实现（对外承诺能力，占位）。

    Raises:
        NotImplementedError: 角动量管理未实现。
    """
    raise NotImplementedError(
        "角动量管理未实现（原 #261）：姿态发动机 E/E_r 矩阵 + 联合控制待补"
    )
