"""太阳光压力模型（cannonball / 基础 Cr 系数）。

实现 Montenbruck & Gill eq. 3.75 的 cannonball SRP 模型：

    a = flux_factor · P · (1 AU / r)² · Cr · A / m · û

其中 ``û`` 为 Sun→SC 单位向量（指向远离太阳），``P = 4.56e-6 N/m²`` 为 1 AU
处的太阳光压常数。``flux_factor ∈ [0, 1]`` 由阴影模型给出（全光照=1，本影=0）。

力模型接口约定：输入状态与输出加速度均在 ``system.coordinate_system`` 下；
系统感知路径（``compute_acceleration``）从 SPICE 取太阳位置并调用阴影模型，
纯函数路径（``_compute_srp_acceleration``）可在无 SPICE 环境下直接测试。

References:
    - Montenbruck & Gill, *Satellite Orbits*, eq. 3.75
    - GMAT R2026a ``SolarRadiationPressure`` (Spherical 模型)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from .shadow import ShadowModel

# 1 AU 处太阳光压常数（N/m²）。等价于 GMAT flux/c = 1367 / 2.998e8。
_P_SRP_1AU = 4.56e-6
# 1 天文单位（km），GMAT nominalSun。
_AU_KM = 149597870.691
_KM_TO_M = 1000.0


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
        shadow: ShadowModel | None = None,
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
    def shadow(self) -> ShadowModel | None:
        """注入的阴影模型，``None`` 表示全光照。"""
        return self._shadow

    def _compute_srp_acceleration(
        self,
        sun_to_sc_vec: npt.ArrayLike,
        flux_factor: float,
    ) -> npt.NDArray[np.floating]:
        """计算光压加速度（纯函数，免 SPICE）。

        Args:
            sun_to_sc_vec: Sun→SC 向量，单位 km，形状 (3,)。
            flux_factor: 光照份额 ∈ [0, 1]，由阴影模型给出。

        Returns:
            加速度向量，单位 km/s²，沿远离太阳方向。
        """
        vec = np.asarray(sun_to_sc_vec, dtype=float)
        r = float(np.linalg.norm(vec))
        # 1/r² 标度（相对 1 AU）。
        pressure = _P_SRP_1AU * (_AU_KM / r) ** 2  # N/m²
        mag_si = flux_factor * pressure * self._cr * self._area / self._mass  # m/s²
        mag_km = mag_si / _KM_TO_M  # km/s²
        return mag_km * (vec / r)

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: object,
    ) -> npt.NDArray[np.floating]:
        """系统感知路径（待 Phase F 实现）。"""
        raise NotImplementedError(
            "SolarRadiationPressure.compute_acceleration (system-aware) "
            "is implemented in a later phase."
        )
