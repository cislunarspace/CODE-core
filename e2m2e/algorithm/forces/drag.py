"""大气阻力力模型。"""

from __future__ import annotations

from typing import Any

from .atmosphere import ExponentialAtmosphere
from .physical_model import PhysicalModel


class DragModel(PhysicalModel):
    """大气阻力力模型。

    在 ITRF（地固系）中计算大气密度与相对速度，求得阻力加速度后转换回
    参考系。大气在 ITRF 中静止，因此相对速度等于航天器 ITRF 速度。

    加速度计算全部由 Rust 编译路径承载（``("drag", ...)`` 力元组，
    ``crates/e2m2e-forces/src/forces/drag.rs``），Python 侧不保留参考实现
    （issue #378）。``to_rust_spec`` 需 system 提供 SPICE（ITRF93 pxform
    帧旋转）；不满足时返回 ``None``，``ForceModel.propagate`` 据此显式报
    能力错误（不静默回退）。

    Args:
        atmosphere: 大气密度模型（依赖注入）。
        body: 中心天体名称，默认 ``'EARTH'``。
        cd: 阻力系数，默认 2.2。
        area: 航天器迎风截面积，单位 m²。
        mass: 航天器质量，单位 kg。
    """

    def __init__(
        self,
        atmosphere: ExponentialAtmosphere,
        area: float,
        mass: float,
        body: str = "EARTH",
        cd: float = 2.2,
    ) -> None:
        self._atmosphere = atmosphere
        self._body = body.upper()
        self._cd = float(cd)
        self._area = float(area)
        self._mass = float(mass)
        if self._area <= 0:
            raise ValueError("area must be positive")
        if self._mass <= 0:
            raise ValueError("mass must be positive")
        if self._cd <= 0:
            raise ValueError("cd must be positive")

    @property
    def atmosphere(self) -> ExponentialAtmosphere:
        """大气密度模型。"""
        return self._atmosphere

    @property
    def body(self) -> str:
        """中心天体名称。"""
        return self._body

    @property
    def cd(self) -> float:
        """阻力系数 Cd。"""
        return self._cd

    @property
    def area(self) -> float:
        """迎风截面积，单位 m²。"""
        return self._area

    @property
    def mass(self) -> float:
        """航天器质量，单位 kg。"""
        return self._mass

    @property
    def ballistic_coefficient(self) -> float:
        """弹道系数 ``Cd·A/m``，单位 m²/kg。"""
        return self._cd * self._area / self._mass

    def to_rust_spec(self, system: Any) -> tuple | None:
        """序列化为 Rust ``("drag", area, mass, cd, propagation_frame, f107, ap)`` 元组。

        f107/ap 从注入的大气模型取出，确保 Rust 路径与配置用同一组太阳活动
        参数（issue #315 的 drag 静默分歧先例，Rust 与配置同源）。

        需要 system 提供 SPICE 以做 ITRF93 pxform 帧旋转。若 system 未暴露
        spice 属性、或中心天体非 EARTH，返回 None——由 ``ForceModel.propagate``
        显式报能力错误，不静默回退 Python 路径。
        """
        if getattr(system, "spice", None) is None:
            return None
        if self._body.upper() != "EARTH":
            return None  # 仅支持地球阻力（ITRF93 帧旋转专用于地球）
        return (
            "drag",
            self._area,
            self._mass,
            self._cd,
            "J2000",
            self._atmosphere.f107,
            self._atmosphere.ap,
        )
