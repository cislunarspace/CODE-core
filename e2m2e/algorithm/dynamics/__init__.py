"""动力学：System（数据上下文）+ Dynamics（传播编排）。

System + Dynamics 都归 algorithm/dynamics/（ADR 0011）：System 描述模型（坐标系、
单位、引力参数、天体列表），Dynamics 用 system 传播（模板方法模式 ADR 0002）。
标准参数数据（μ/DU/TU/平动点值）在 ``data/templates/systems.py``。

实现状态：骨架。System/CR3BP_System/EphemerisSystem + Dynamics/CR3BP_Dynamics/
EphemerisDynamics/BCR4BP_Dynamics 待从 ``core/`` 迁入。
"""

from __future__ import annotations

__all__: list[str] = []
