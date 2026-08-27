"""ECOM 光压模型（DFH 兼容 9 系数 DYB 参数化）。

ECOM（Empirical CODE Orbit Model）将光压加速度分解到卫星本体坐标系的
D（太阳方向）、Y（太阳帆板法向）、B（D×Y）三轴，每个方向用常量+周期项展开。

DFH 的 DYB 9 系数含义：
- dyb[0] = 等效面质比 (m²/kg)
- dyb[1:5] = D 方向周期项（cos(u), sin(u), cos(2u), sin(2u)）
- dyb[5:7] = Y 方向（cos(u), sin(u)）
- dyb[7:9] = B 方向（常量, cos(u)）

当仅 dyb[0] 非零时，模型退化为标准 cannonball SRP。

加速度计算全部由 Rust 编译路径承载（``("ecom_srp", dyb, shadow_bodies)``
力元组，``crates/e2m2e-forces/src/forces/ecom.rs``），Python 侧不保留参考实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from ..dynamics.system import System
    from .shadow import ConicalShadowModel


class EcomSolarRadiationPressure(PhysicalModel):
    """ECOM 光压模型（DFH 兼容 9 系数 DYB 参数化）。

    Args:
        dyb: DYB 系数列表（长度 9）。
            - dyb[0] = 等效面质比 (m²/kg)
            - dyb[1:5] = D 方向周期项（cos(u), sin(u), cos(2u), sin(2u)）
            - dyb[5:7] = Y 方向（cos(u), sin(u)）
            - dyb[7:9] = B 方向（常量, cos(u)）
        shadow: 阴影模型（注入）。``None`` 表示全光照（flux_factor 恒为 1）。
    """

    def __init__(
        self,
        dyb: list[float],
        shadow: ConicalShadowModel | None = None,
    ) -> None:
        if len(dyb) != 9:
            raise ValueError(f"dyb must have 9 elements, got {len(dyb)}")
        self._dyb = [float(v) for v in dyb]
        self._shadow = shadow

    @property
    def dyb(self) -> list[float]:
        """DYB 系数列表副本。"""
        return list(self._dyb)

    @property
    def shadow(self) -> ConicalShadowModel | None:
        """注入的阴影模型，``None`` 表示全光照。"""
        return self._shadow

    def to_rust_spec(self, system: System | None = None) -> tuple:
        """序列化为 ``("ecom_srp", dyb, shadow_bodies)``。"""
        shadow_bodies = list(self._shadow.bodies) if self._shadow is not None else []
        return ("ecom_srp", self._dyb, shadow_bodies)

    def to_config(self) -> dict:
        """序列化为配置字典。"""
        shadow_cfg = None
        if self._shadow is not None:
            shadow_cfg = {
                "type": "ConicalShadowModel",
                "params": {"bodies": list(self._shadow.bodies), "radii": self._shadow.radii},
            }
        return {"dyb": self._dyb, "shadow": shadow_cfg}

    @classmethod
    def from_config(cls, config: dict) -> EcomSolarRadiationPressure:
        """从配置字典构造实例。"""
        shadow = None
        shadow_cfg = config.get("shadow")
        if shadow_cfg is not None:
            from .shadow import ConicalShadowModel

            shadow = ConicalShadowModel(**shadow_cfg.get("params", {}))
        return cls(dyb=list(config["dyb"]), shadow=shadow)
