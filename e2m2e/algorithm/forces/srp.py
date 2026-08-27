"""太阳光压力模型（cannonball / 基础 Cr 系数）。

实现 Montenbruck & Gill eq. 3.75 的 cannonball SRP 模型：

    a = flux_factor · P · (1 AU / r)² · Cr · A / m · û

其中 ``û`` 为 Sun→SC 单位向量（指向远离太阳），``P = 4.56e-6 N/m²`` 为 1 AU
处的太阳光压常数。``flux_factor ∈ [0, 1]`` 由阴影模型给出（全光照=1，本影=0）。

加速度计算全部由 Rust 编译路径承载（``("srp", ...)`` 力元组，
``crates/e2m2e-forces/src/forces/srp.rs``），Python 侧不保留参考实现。

References:
    - Montenbruck & Gill, *Satellite Orbits*, eq. 3.75
    - GMAT R2026a ``SolarRadiationPressure`` (Spherical 模型)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from .shadow import ConicalShadowModel


class SolarRadiationPressure(PhysicalModel):
    """基础 Cr 系数太阳光压模型（cannonball）。

    Args:
        area: 航天器迎风截面积，单位 m²。
        mass: 航天器质量，单位 kg。
        cr: 辐射反射系数（1=全吸收，2=全反射），默认 1.5。
        shadow: 阴影模型（注入）。``None`` 表示全光照（flux_factor 恒为 1）。
    """

    def __init__(
        self,
        area: float,
        mass: float,
        cr: float = 1.5,
        shadow: ConicalShadowModel | None = None,
    ) -> None:
        self._area = float(area)
        self._mass = float(mass)
        self._cr = float(cr)
        self._shadow = shadow
        if self._area <= 0:
            raise ValueError("area must be positive")
        if self._mass <= 0:
            raise ValueError("mass must be positive")

    @property
    def area(self) -> float:
        """截面积（m²）。"""
        return self._area

    @property
    def mass(self) -> float:
        """质量（kg）。"""
        return self._mass

    @property
    def cr(self) -> float:
        """辐射反射系数 Cr。"""
        return self._cr

    @property
    def shadow(self) -> ConicalShadowModel | None:
        """注入的阴影模型，``None`` 表示全光照。"""
        return self._shadow

    def to_rust_spec(self, system) -> tuple | None:
        """序列化为 ``("srp", area, mass, cr, shadow_bodies)``。"""
        shadow_bodies = list(self._shadow.bodies) if self._shadow is not None else []
        return ("srp", self._area, self._mass, self._cr, shadow_bodies)


class VariableMassSolarRadiationPressure(PhysicalModel):
    """质量由增广状态提供的 cannonball 光压模型。

    小推力任务质量在线衰减，固定质量的 :class:`SolarRadiationPressure` 会
    在整个传播区间用同一初始质量。本类只存截面积 ``area``（m²）与 ``cr``，
    质量在每个增广状态里取出，因此 ``a = flux·P·(1AU/r)²·cr·area/m`` 随
    推进耗质量自动更新。

    Args:
        area: 航天器迎风截面积，单位 m²。
        cr: 辐射反射系数，默认 1.5。
        shadow: 阴影模型（注入）；``None`` 表示全光照。
    """

    def __init__(
        self,
        area: float,
        cr: float = 1.5,
        shadow: ConicalShadowModel | None = None,
    ) -> None:
        self._area = float(area)
        self._cr = float(cr)
        self._shadow = shadow
        if self._area <= 0:
            raise ValueError("area must be positive")

    @property
    def area(self) -> float:
        return self._area

    @property
    def cr(self) -> float:
        return self._cr

    @property
    def shadow(self) -> ConicalShadowModel | None:
        return self._shadow

    def to_rust_spec(self, system) -> tuple | None:
        """序列化为 ``("srp_variable_mass", area, cr, shadow_bodies)``。"""
        shadow_bodies = list(self._shadow.bodies) if self._shadow is not None else []
        return ("srp_variable_mass", self._area, self._cr, shadow_bodies)
