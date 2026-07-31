"""轨道保持（轨道控制）。

站保控制律留 Python（领域知识，ADR 0011）：特征点控制（special_point）、目标点
严格/宽松控制（target_point）、误差模型（error_models）、三轨道蒙特卡洛
（monte_carlo）。``controller.py`` 是编排器（读输入星历、选控制律、配双力模型、
汇总输出）。

实现状态：骨架。控制律/误差模型/蒙特卡洛待从 ``algorithms/station_keeping/``
迁入；controller.py 待从 ``dfh/control_orbit.py`` 迁入。

未实现（对外承诺能力）：角动量管理（原 #261），占位函数抛 ``NotImplementedError``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ControlOrbitResult", "control_orbit"]


@dataclass
class ControlOrbitResult:
    """轨道保持仿真结果。

    Attributes:
        sk_statistic: SK_STATISTIC 表（总 Δv/最大 Δv）。
        num_failed: 蒙特卡洛失败样本数。
        maneuvers: 机动序列（MJD(TDB) + Δv）。
        controlled_ephemeris: 最后一次样本的受控真实轨道星历。
    """

    sk_statistic: Any
    num_failed: int
    maneuvers: Any
    controlled_ephemeris: Any = None
    raw: Any = field(default=None, repr=False)


def control_orbit(
    input_ephemeris: Any,
    *,
    control_mode: int = 1,
    special_mode: int = 1,
    num_controls: int = 120,
    num_monte_carlo: int = 5,
    **kwargs,
) -> ControlOrbitResult:
    """端到端轨道保持仿真。

    实现状态：骨架。完整实现待从 ``dfh/control_orbit.py`` 迁入。

    Args:
        input_ephemeris: 标称轨道星历（通用星历容器）。
        control_mode: 1=目标点宽松、2=目标点严格、3=特征点。
        special_mode: 特征点模式 1=Lissajous、2=Halo/NRHO。
        num_controls: 控制次数。
        num_monte_carlo: 蒙特卡洛样本数。

    Raises:
        NotImplementedError: 实现未完成（骨架）。
    """
    raise NotImplementedError(
        "control_orbit 实现未完成（待从 dfh/control_orbit.py 迁入），能力在规划中"
    )


def momentum_management(
    input_ephemeris: Any,
    *,
    engine_layout: Any = None,
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
