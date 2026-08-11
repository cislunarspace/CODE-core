"""ECOM 光压模型（DFH 兼容 9 系数 DYB 参数化）。

ECOM（Empirical CODE Orbit Model）将光压加速度分解到卫星本体坐标系的
D（太阳方向）、Y（太阳帆板法向）、B（D×Y）三轴，每个方向用常量+周期项展开。

DFH 的 DYB 9 系数含义：
- dyb[0] = 等效面质比 (m²/kg)
- dyb[1:5] = D 方向周期项（cos(u), sin(u), cos(2u), sin(2u)）
- dyb[5:7] = Y 方向（cos(u), sin(u)）
- dyb[7:9] = B 方向（常量, cos(u)）

当仅 dyb[0] 非零时，模型退化为标准 cannonball SRP。

力模型接口约定：输入状态与输出加速度均在 ``system.coordinate_system`` 下；
系统感知路径（``compute_acceleration``）从 SPICE 取太阳位置并调用阴影模型，
纯函数路径（``_compute_ecom_acceleration``）可在无 SPICE 环境下直接测试。
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.constants import AU_KM, KM_TO_M, SOLAR_PRESSURE_1AU
from .physical_model import PhysicalModel, require_inertial_frame

if TYPE_CHECKING:
    from ..system import System
    from .shadow import ConicalShadowModel


def _cross(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """三维向量叉积。"""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _build_dyb_frame(
    sc_pos: npt.NDArray, sun_to_sc: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """构建 D-Y-B 坐标系三轴单位向量。

    Args:
        sc_pos: 航天器位置（相对传播原点），形状 (3,)，km。
        sun_to_sc: Sun→SC 向量，形状 (3,)，km。

    Returns:
        (d_hat, y_hat, b_hat) 三个单位向量。
    """
    r = float(np.linalg.norm(sun_to_sc))
    d_hat = sun_to_sc / r

    # Y = sc_pos × D（轨道面法向近似）
    y_raw = _cross(sc_pos, d_hat)
    y_norm = float(np.linalg.norm(y_raw))
    if y_norm > 1e-10:
        y_hat = y_raw / y_norm
    else:
        # 退化：D 与 SC 平行，用 z 轴构造
        z = np.array([0.0, 0.0, 1.0])
        y2 = _cross(z, d_hat)
        y2n = float(np.linalg.norm(y2))
        if y2n > 1e-10:
            y_hat = y2 / y2n
        else:
            x = np.array([1.0, 0.0, 0.0])
            y3 = _cross(x, d_hat)
            y3n = float(np.linalg.norm(y3))
            y_hat = y3 / y3n

    b_hat = _cross(d_hat, y_hat)
    return d_hat, y_hat, b_hat


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

    def _compute_ecom_acceleration(
        self,
        sc_pos: npt.ArrayLike,
        sun_to_sc_vec: npt.ArrayLike,
        flux_factor: float,
    ) -> npt.NDArray[np.floating]:
        """计算 ECOM 光压加速度（纯函数，免 SPICE）。

        Args:
            sc_pos: 航天器位置（相对传播原点），形状 (3,)，km。
            sun_to_sc_vec: Sun→SC 向量，形状 (3,)，km。
            flux_factor: 光照份额 ∈ [0, 1]，由阴影模型给出。

        Returns:
            加速度向量，单位 km/s²。
        """
        sc = np.asarray(sc_pos, dtype=float)
        vec = np.asarray(sun_to_sc_vec, dtype=float)
        r = float(np.linalg.norm(vec))
        if r == 0.0:
            return np.zeros(3)

        d_hat, y_hat, b_hat = _build_dyb_frame(sc, vec)

        # 基础加速度幅值
        pressure = SOLAR_PRESSURE_1AU * (AU_KM / r) ** 2  # N/m²
        a0 = flux_factor * pressure * self._dyb[0] / KM_TO_M  # km/s²

        # 太阳平近点角（简化：u=0）
        u = 0.0

        # ECOM 三向分量
        d_comp = (
            1.0
            + self._dyb[1] * np.cos(u)
            + self._dyb[2] * np.sin(u)
            + self._dyb[3] * np.cos(2.0 * u)
            + self._dyb[4] * np.sin(2.0 * u)
        )
        y_comp = self._dyb[5] * np.cos(u) + self._dyb[6] * np.sin(u)
        b_comp = self._dyb[7] + self._dyb[8] * np.cos(u)

        return a0 * (d_comp * d_hat + y_comp * y_hat + b_comp * b_hat)

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """系统感知 ECOM 光压加速度。

        从 ``system`` 读取传播原点与 SPICE，查询太阳相对原点的 J2000 位置，
        取阴影模型的光照份额（无阴影时为全光照），调用纯函数
        ``_compute_ecom_acceleration``。要求参考系为惯性系。
        """
        warnings.warn(
            f"{self.__class__.__name__}.compute_acceleration 走 Python 回退路径，"
            "应优先走 Rust 编译路径。",
            DeprecationWarning,
            stacklevel=2,
        )
        _cs, spice, origin = require_inertial_frame(system, t)
        sc_pos = np.asarray(state, dtype=float)[:3]
        sun_pos = spice.get_body_state("SUN", t, "J2000", origin)[:3]

        flux = self._shadow.flux_factor(t, state, system) if self._shadow is not None else 1.0
        return self._compute_ecom_acceleration(sc_pos, sc_pos - sun_pos, flux)

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
