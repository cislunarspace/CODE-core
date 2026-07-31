"""Facade 门面：唯一公开顶级入口，粗粒度任务方法。

两层粒度（ADR 0014）：Facade 暴露粗粒度任务方法（人类/Agent 常用），算法层保留
细粒度 API（专家用）。MCP 工具 = Facade 方法全集（纯派生），方法带
``mcp_exposed`` 元数据控制是否对 MCP 暴露。

实现状态：骨架。方法签名已定，实现待接入 algorithm/ 各编排器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Facade"]


@dataclass
class _ToolMeta:
    """Facade 方法元数据（纯派生 + 元数据标记，ADR 0014）。"""

    mcp_exposed: bool = True


class Facade:
    """e2m2e 唯一公开入口。

    ``Facade(config=Config(...))`` 构造注入配置（ADR 0014）。方法对应任务级能力，
    一档任务（稳定骨架，会增）：orbit_design / orbit_control / transfer_design /
    orbit_propagation / spacetime_transform。二档子任务（会增）标 ``mcp_exposed=True``，
    三档辅助标 ``False``。
    """

    def __init__(self, config: Any = None) -> None:
        """构造 Facade。

        Args:
            config: 运行配置（api/config.py Config），缺省从环境变量读。
        """
        self._config = config
        # 待接入：algorithm/ 各编排器

    def design_orbit(self, **params) -> Any:
        """任务轨道设计（一档）。"""
        raise NotImplementedError("Facade.design_orbit 待接入 algorithm/design/")

    def control_orbit(self, **params) -> Any:
        """轨道保持（一档）。"""
        raise NotImplementedError("Facade.control_orbit 待接入 algorithm/station_keeping/")

    def transfer_design(self, **params) -> Any:
        """转移轨道设计（一档）。"""
        raise NotImplementedError("Facade.transfer_design 待接入 algorithm/transfer/")

    def orbit_propagation(self, **params) -> Any:
        """轨道预报（一档）。"""
        raise NotImplementedError("Facade.orbit_propagation 待接入 algorithm/propagation.py")

    def spacetime_transform(self, **params) -> Any:
        """时空坐标转换（一档）。"""
        raise NotImplementedError("Facade.spacetime_transform 待接入 algorithm/coordinate/")

    def orbit_family_generation(self, **params) -> Any:
        """轨道族生成（二档）。"""
        raise NotImplementedError("Facade.orbit_family_generation 待接入 algorithm/family/")

    def orbit_stability(self, **params) -> Any:
        """稳定性分析（二档）。"""
        raise NotImplementedError("Facade.orbit_stability 待接入 algorithm/stability.py")

    def transfer_search(self, **params) -> Any:
        """转移网格搜索（二档）。"""
        raise NotImplementedError("Facade.transfer_search 待接入 algorithm/transfer/")

    def low_thrust_design(self, **params) -> Any:
        """小推力转移设计（二档）。"""
        raise NotImplementedError("Facade.low_thrust_design 待接入 algorithm/transfer/")

    def manifold_analysis(self, **params) -> Any:
        """不变流形分析（二档）。"""
        raise NotImplementedError("Facade.manifold_analysis 待接入 algorithm/manifold/")

    def low_energy_transfer(self, **params) -> Any:
        """低能转移（二档）。"""
        raise NotImplementedError("Facade.low_energy_transfer 待接入 algorithm/transfer/")

    def relative_motion(self, **params) -> Any:
        """相对运动（二档）。"""
        raise NotImplementedError("Facade.relative_motion 待接入 algorithm/proximity/")
