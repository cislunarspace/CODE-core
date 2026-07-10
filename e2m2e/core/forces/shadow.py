"""圆锥阴影模型（地影/月影）。

实现 GMAT ``ShadowState`` 的圆锥阴影算法（Montenbruck & Gill §3.4.2 的
"Shadow Function"）：从航天器看太阳与遮挡体的视角径 (a, b) 与角距 c，分四
分支判定全光照 / 本影 / 半影 / 环形食，半影区用 M&G eq. 3.92-3.94 的精确
圆面重叠面积。多遮挡体合成遵循 GMAT GMT-6543 规范。

References:
    - Montenbruck & Gill, *Satellite Orbits*, §3.4.2 (eq. 3.85-3.94)
    - GMAT R2026a ``ShadowState`` / ``SolarRadiationPressure::GetShadowStateFromAllBodies``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..constants import R_EARTH
from .physical_model import require_inertial_frame

# 默认遮挡体赤道半径（km），取 GMAT PCK 值。可通过 radii 覆盖参数扩展。
_BODY_RADII_KM: dict[str, float] = {
    "EARTH": R_EARTH,
    "MOON": 1737.4,
    "SUN": 695700.0,
}


class ConicalShadowModel:
    """圆锥阴影模型（本影 + 半影 + 环形食）。

    Args:
        bodies: 遮挡体名称列表（大写），默认 ``["EARTH"]``。需显式列出所有
            想计算阴影的天体（含传播原点天体，若其阴影相关）。
        radii: 天体半径覆盖字典（km），用于补充默认表外的天体。
    """

    def __init__(
        self,
        bodies: list[str] | tuple[str, ...] = ("EARTH",),
        radii: dict[str, float] | None = None,
    ) -> None:
        self._bodies: tuple[str, ...] = tuple(b.upper() for b in bodies)
        self._radii_arg: dict[str, float] | None = (
            {k.upper(): float(v) for k, v in radii.items()} if radii else None
        )
        self._radii: dict[str, float] = dict(_BODY_RADII_KM)
        if radii:
            self._radii.update({k.upper(): float(v) for k, v in radii.items()})
        for body in self._bodies:
            if body not in self._radii:
                raise ValueError(f"unknown shadow body {body!r}; provide a radii override")

    @property
    def bodies(self) -> tuple[str, ...]:
        """遮挡体列表（大写）。"""
        return self._bodies

    @property
    def radii(self) -> dict[str, float] | None:
        """用户传入的天体半径覆盖（大写键）；``None`` 表示全用默认值。"""
        return dict(self._radii_arg) if self._radii_arg is not None else None

    def body_radius(self, body: str) -> float:
        """返回天体半径（km）。"""
        return self._radii[body.upper()]

    def _body_flux_factor(
        self,
        sc_pos: npt.ArrayLike,
        body_pos: npt.ArrayLike,
        sun_pos: npt.ArrayLike,
        body_radius: float,
        sun_radius: float,
    ) -> float:
        """单遮挡体光照份额（纯几何，免 SPICE）。

        Args:
            sc_pos: 航天器位置（公共惯性系），km。
            body_pos: 遮挡体位置（公共惯性系），km。
            sun_pos: 太阳位置（公共惯性系），km。
            body_radius: 遮挡体半径，km。
            sun_radius: 太阳半径，km。

        Returns:
            光照份额 ∈ [0, 1]。
        """
        sc = np.asarray(sc_pos, dtype=float)
        body = np.asarray(body_pos, dtype=float)
        sun = np.asarray(sun_pos, dtype=float)

        sc_to_body = body - sc
        sc_to_sun = sun - sc
        d_body = float(np.linalg.norm(sc_to_body))
        d_sun = float(np.linalg.norm(sc_to_sun))

        # GMAT 守卫：避免 arcsin 定义域错误。
        if sun_radius >= d_sun:
            return 1.0
        if body_radius >= d_body:
            return 0.0

        a = np.arcsin(sun_radius / d_sun)  # 太阳视角径
        b = np.arcsin(body_radius / d_body)  # 遮挡体视角径
        cos_c = float(np.dot(sc_to_body, sc_to_sun) / (d_body * d_sun))
        c = np.arccos(np.clip(cos_c, -1.0, 1.0))  # 太阳-遮挡体角距

        if a + b <= c:
            return 1.0  # 全光照
        if c <= abs(a - b):
            # 一盘完全包含另一盘（含内切边界）
            if b >= a:
                return 0.0  # 本影：遮挡体大于太阳，全遮
            return 1.0 - (b / a) ** 2  # 环形食：遮挡体小于太阳，annular
        # 半影：|a-b| < c < a+b，M&G eq. 3.92-3.94 精确圆面重叠
        a2 = a * a
        b2 = b * b
        x = (c * c + a2 - b2) / (2.0 * c)
        y = np.sqrt(max(0.0, a2 - x * x))
        area = a2 * np.arccos(x / a) + b2 * np.arccos((c - x) / b) - c * y
        return 1.0 - area / (np.pi * a2)

    def _combine_body_fluxes(
        self,
        factors: list[float],
        body_angular_radii: list[float],
        body_directions: list[npt.NDArray[np.floating]],
    ) -> float:
        """多遮挡体光照份额合成（GMAT GMT-6543 规范，纯函数）。

        - 任一遮挡体本影（factor=0）→ 0
        - 恰两体部分阴影且日盘上不重叠（a1+a2 < c12）→ f1+f2−1（包容排斥）
        - 恰两体部分阴影且日盘上重叠 → min(f1, f2)
        - 其余（1 体或 3+ 体）→ min(所有 factor)

        Args:
            factors: 各遮挡体的单体光照份额，与下两参数同序。
            body_angular_radii: 各遮挡体对 SC 的视角径 asin(R/d)。
            body_directions: 各遮挡体相对 SC 的方向单位向量。

        Returns:
            合成光照份额 ∈ [0, 1]。
        """
        if any(f == 0.0 for f in factors):
            return 0.0

        partial = [i for i, f in enumerate(factors) if 0.0 < f < 1.0]

        if len(partial) == 2:
            i, j = partial
            a_i = body_angular_radii[i]
            a_j = body_angular_radii[j]
            cos_c = float(np.dot(body_directions[i], body_directions[j]))
            c_ij = np.arccos(np.clip(cos_c, -1.0, 1.0))
            if a_i + a_j < c_ij:
                return factors[i] + factors[j] - 1.0  # 包容排斥
            return min(factors[i], factors[j])  # 重叠 → 保守

        return min(factors)

    def flux_factor(
        self,
        t: float,
        state: npt.ArrayLike,
        system: Any,
    ) -> float:
        """系统感知光照份额。

        从 ``system`` 读取传播原点与 SPICE，查询太阳及各遮挡体相对原点的 J2000
        位置，调用纯几何 ``_body_flux_factor`` 与 ``_combine_body_fluxes``。
        要求参考系为惯性系（轴旋转矩阵为单位阵）。
        """
        _cs, spice, origin = require_inertial_frame(system, t)
        sc_pos = np.asarray(state, dtype=float)[:3]
        sun_pos = spice.get_body_state("SUN", t, "J2000", origin)[:3]
        sun_radius = self._radii["SUN"]

        factors: list[float] = []
        angular_radii: list[float] = []
        directions: list[npt.NDArray[np.floating]] = []
        for body in self._bodies:
            body_pos = spice.get_body_state(body, t, "J2000", origin)[:3]
            r_body = self._radii[body]
            factors.append(self._body_flux_factor(sc_pos, body_pos, sun_pos, r_body, sun_radius))
            sc_to_body = body_pos - sc_pos
            d_body = float(np.linalg.norm(sc_to_body))
            angular_radii.append(np.arcsin(min(1.0, r_body / d_body)))
            directions.append(sc_to_body / d_body)

        return self._combine_body_fluxes(factors, angular_radii, directions)
