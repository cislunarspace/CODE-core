"""大气阻力力模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.atmosphere import AtmosphereModel
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ITRFApproxAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

from .exceptions import CoordinateTransformError
from .physical_model import PhysicalModel

# WGS84 地球赤道半径（km），用于地心高度计算。
_EARTH_EQUATORIAL_RADIUS_KM = 6378.137

_KM_TO_M = 1000.0


class DragModel(PhysicalModel):
    """大气阻力力模型。

    在 ITRF（地固系）中计算大气密度与相对速度，求得阻力加速度后转换回
    传播坐标系。大气在 ITRF 中静止，因此相对速度等于航天器 ITRF 速度。

    力模型接口约定：输入状态与输出加速度均在 ``system.coordinate_system``
    下；本类内部负责转换到 ITRF 并转回。

    Args:
        atmosphere: 大气密度模型（依赖注入）。
        body: 中心天体名称，默认 ``'EARTH'``。
        cd: 阻力系数，默认 2.2。
        area: 航天器迎风截面积，单位 m²。
        mass: 航天器质量，单位 kg。
    """

    def __init__(
        self,
        atmosphere: AtmosphereModel,
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
    def atmosphere(self) -> AtmosphereModel:
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

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """计算大气阻力加速度。

        Args:
            t: SPICE et 时间（秒）。
            state: 状态向量，在 ``system.coordinate_system`` 下。
            system: 动力学系统；若为 ``None`` 则假设状态已在 ITRF 下
                （仅用于隔离测试）。

        Returns:
            加速度向量，形状 ``(3,)``，单位 km/s²。
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape[0] < 6:
            raise ValueError("state must have at least 6 elements")

        if system is None:
            r_itrf = state_arr[:3].copy()
            v_itrf = state_arr[3:6].copy()
            a_drag = self._compute_drag_in_itrf(r_itrf, v_itrf)
            return a_drag

        state_itrf = self._transform_state_to_itrf(t, state_arr, system)
        r_itrf = state_itrf[:3]
        v_itrf = state_itrf[3:6]
        a_drag_itrf = self._compute_drag_in_itrf(r_itrf, v_itrf)
        return self._transform_vector_from_itrf(t, a_drag_itrf, system)

    def _compute_drag_in_itrf(
        self,
        r_itrf: npt.NDArray[np.floating],
        v_itrf: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """在 ITRF 中计算阻力加速度，返回 km/s²。"""
        altitude_km = float(np.linalg.norm(r_itrf)) - _EARTH_EQUATORIAL_RADIUS_KM
        rho = self._atmosphere.density(altitude_km)  # kg/m³

        # 大气在 ITRF 中静止，相对速度 = ITRF 速度；转 SI（m/s）
        v_rel = v_itrf * _KM_TO_M  # m/s
        v_rel_mag = float(np.linalg.norm(v_rel))

        if rho == 0.0 or v_rel_mag == 0.0:
            return np.zeros(3)

        bc = self.ballistic_coefficient  # m²/kg

        # a_drag [m/s²] = -0.5 · ρ · BC · |v_rel|² · v̂_rel
        # 等价于 -0.5 · ρ · BC · |v_rel| · v_rel
        a_drag_si = -0.5 * rho * bc * v_rel_mag * v_rel
        return a_drag_si / _KM_TO_M  # 转回 km/s²

    def _transform_state_to_itrf(
        self, t: float, state: npt.NDArray[np.floating], system: Any
    ) -> npt.NDArray[np.floating]:
        """把状态转换到 ITRF。"""
        input_cs = self._get_itrf_coordinate_system(system)
        try:
            state_itrf = system.coordinate_system.transform_state(
                state, from_cs=system.coordinate_system, to_cs=input_cs, et=t
            )
        except Exception as exc:
            raise CoordinateTransformError(
                f"Failed to transform state to ITRF for body {self._body}"
            ) from exc
        return state_itrf

    def _transform_vector_from_itrf(
        self,
        t: float,
        vector: npt.NDArray[np.floating],
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """把 ITRF 中的加速度矢量转换回传播坐标系。"""
        input_cs = self._get_itrf_coordinate_system(system)
        try:
            return system.coordinate_system.transform_vector(
                vector, from_cs=input_cs, to_cs=system.coordinate_system, et=t
            )
        except Exception as exc:
            raise CoordinateTransformError(
                "Failed to transform drag acceleration from ITRF"
            ) from exc

    def _get_itrf_coordinate_system(self, system: Any) -> CoordinateSystem:
        """构造 ITRF 坐标系（地固系）。"""
        spice = getattr(system, "spice", None)
        if spice is None:
            raise CoordinateTransformError(
                "system must expose a 'spice' attribute for ITRF transforms"
            )
        axes = ITRFApproxAxes()
        origin = CelestialBodyOrigin(body=self._body, spice=spice)
        return CoordinateSystem(axes=axes, origin=origin)
