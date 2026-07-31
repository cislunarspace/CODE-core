"""相对论修正力模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ...data.templates.systems import R_EARTH
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes, ITRFSpiceAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from .exceptions import RelativisticCorrectionError
from .physical_model import PhysicalModel


class RelativisticCorrection(PhysicalModel):
    """相对论修正力模型。

    实现 Schwarzschild、Lense-Thirring 与 de Sitter（geodesic）三项相对论
    加速度修正，公式与 GMAT R2026a ``RelativisticCorrection`` 对齐。
    """

    # 常用天体赤道半径 (km)，与 GMAT 默认值/IAU 标准一致。
    _DEFAULT_BODY_RADII_KM: dict[str, float] = {
        "EARTH": R_EARTH,
        "MOON": 1737.4,
        "SUN": 696000.0,
        "MARS": 3396.19,
        "JUPITER": 71492.0,
    }

    def __init__(
        self,
        central_body: str,
        *,
        primary_body: str | None = "SUN",
        enable_schwarzschild: bool = True,
        enable_lense_thirring: bool = True,
        enable_de_sitter: bool = True,
        angular_momentum_vector: npt.ArrayLike | None = None,
        body_radius: float | None = None,
        c: float = 299792.458,
        gamma: float = 1.0,
    ) -> None:
        self._central_body = central_body.upper()
        self._primary_body = primary_body.upper() if primary_body is not None else None
        self._enable_schwarzschild = bool(enable_schwarzschild)
        self._enable_lense_thirring = bool(enable_lense_thirring)
        self._enable_de_sitter = bool(enable_de_sitter)
        self._angular_momentum_vector = (
            None
            if angular_momentum_vector is None
            else np.asarray(angular_momentum_vector, dtype=float).copy()
        )
        self._body_radius = body_radius
        self._c = float(c)
        self._gamma = float(gamma)

    @property
    def central_body(self) -> str:
        """中心天体名称（大写）。"""
        return self._central_body

    @property
    def primary_body(self) -> str | None:
        """de Sitter 项主天体名称（大写），可能为 ``None``。"""
        return self._primary_body

    @property
    def enable_schwarzschild(self) -> bool:
        """Schwarzschild 项开关。"""
        return self._enable_schwarzschild

    @property
    def enable_lense_thirring(self) -> bool:
        """Lense-Thirring 项开关。"""
        return self._enable_lense_thirring

    @property
    def enable_de_sitter(self) -> bool:
        """de Sitter 项开关。"""
        return self._enable_de_sitter

    @property
    def angular_momentum_vector(self) -> npt.NDArray[np.floating] | None:
        """Lense-Thirring 角动量矢量（覆盖值），单位 km²/s。

        注意：这里的 ``J`` 与 GMAT 约定一致，是 ``(2/5) * R² * spin_rate``
        形式的归一化量，不是 SI 物理角动量（kg·m²/s）。
        """
        return self._angular_momentum_vector

    @property
    def body_radius(self) -> float | None:
        """中心天体赤道半径（覆盖值），单位 km。"""
        return self._body_radius

    @property
    def c(self) -> float:
        """光速，单位 km/s。"""
        return self._c

    @property
    def gamma(self) -> float:
        """后牛顿参数 gamma。"""
        return self._gamma

    def to_rust_spec(self, system) -> tuple | None:
        """序列化为 ``("relativistic", ...)`` 元组。

        - LT 项需要 sxform + body-fixed frame；本仓库已实测 NRHO 上 LT 量级 < 1m
          （#343 排查），但完整移植已实现（含 sxform via cspice-sys FFI）。
        - 如果 LT 启用但 angular_momentum_vector 未传，Rust 侧会每步 sxform
          自动算（与 Python 一致）；如需避免 sxform 开销，可在 Python 侧
          预先算好 J 向量并传入 angular_momentum_vector。
        """
        mu_central = float(system.gravitational_parameter(self._central_body))
        mu_primary = (
            float(system.gravitational_parameter(self._primary_body))
            if self._primary_body is not None
            else None
        )
        # angular_momentum_vector：None 时 Rust 自动 sxform
        j_vec = (
            list(self._angular_momentum_vector)
            if self._angular_momentum_vector is not None
            else None
        )
        return (
            "relativistic",
            self._central_body,
            self._primary_body,
            mu_central,
            mu_primary,
            self._enable_schwarzschild,
            self._enable_lense_thirring,
            self._enable_de_sitter,
            j_vec,
            self._body_radius,
            self._gamma,
        )

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """返回相对论修正加速度，km/s²。"""
        state_arr = np.asarray(state, dtype=float)
        rv = state_arr[:3].copy()
        vv = state_arr[3:6].copy()

        mu = float(system.gravitational_parameter(self._central_body))
        c = self._c
        c2 = c * c
        r = float(np.linalg.norm(rv))
        v = float(np.linalg.norm(vv))

        acc = np.zeros(3, dtype=float)

        if self._enable_schwarzschild:
            s1 = mu / (c2 * r * r * r)
            s2 = ((4.0 * mu / r) - (v * v)) * rv
            rv_dot_vv = float(np.dot(rv, vv))
            s3 = 4.0 * rv_dot_vv * vv
            acc = acc + self._gamma * s1 * (s2 + s3)

        if self._enable_lense_thirring:
            j_vec = self._angular_momentum_vector
            if j_vec is None:
                j_vec = self._compute_angular_momentum(t, system)
            rv_cross_vv = np.cross(rv, vv)
            vv_cross_j = np.cross(vv, j_vec)
            lt1 = 2.0 * mu / (c2 * r * r * r)
            lt2 = (3.0 / (r * r)) * np.dot(rv, j_vec)
            acc = acc + lt1 * (lt2 * rv_cross_vv + vv_cross_j)

        if self._enable_de_sitter and self._primary_body is not None:
            omega = self._compute_de_sitter_omega(t, system)
            acc = acc + 2.0 * np.cross(omega, vv)

        return acc

    def _compute_de_sitter_omega(self, t: float, system: Any) -> npt.NDArray[np.floating]:
        """计算 de Sitter（geodesic）项的 omega 矢量。"""
        spice = system.spice
        primary = self._primary_body
        central_state = spice.get_body_state(
            self._central_body, t, "J2000", "SOLAR SYSTEM BARYCENTER"
        )
        primary_state = spice.get_body_state(primary, t, "J2000", "SOLAR SYSTEM BARYCENTER")
        rel_state = central_state - primary_state
        r_vec = rel_state[:3]
        v_vec = rel_state[3:6]
        r = float(np.linalg.norm(r_vec))
        mu_primary = float(system.gravitational_parameter(primary))
        c2 = self._c * self._c
        factor = -mu_primary / (c2 * r * r * r)
        pos = factor * r_vec
        vel = 1.5 * v_vec
        return np.cross(vel, pos)

    def _compute_angular_momentum(self, t: float, system: Any) -> npt.NDArray[np.floating]:
        """通过 bodyFixed -> inertial 旋转矩阵实时计算角动量矢量 J。"""
        try:
            spice = system.spice
            # body_inertial 备用：GMAT 同时构造 fixed/inertial 两个
            # 坐标系；这里只使用 fixed 系，如需反算惯系轴可参考展开。
            body_inertial = CoordinateSystem(  # noqa: F841
                axes=ICRSAxes(),
                origin=CelestialBodyOrigin(body=self._central_body, spice=spice),
            )
            body_fixed = CoordinateSystem(
                axes=ITRFSpiceAxes(),
                origin=CelestialBodyOrigin(body=self._central_body, spice=spice),
            )
            R, Rdot = body_fixed.axes.rotation_and_rate(t)
        except Exception as exc:
            raise RelativisticCorrectionError(
                "Automatic angular momentum computation requires a body-fixed "
                "coordinate system and SPICE binary PCK kernels. Provide "
                "angular_momentum_vector explicitly, or load the required kernels."
            ) from exc

        # GMAT: bodySpinVector 从 fixed -> inertial 的 R 和 Rdot 提取
        body_spin_vector = np.array(
            [
                -R[0, 2] * Rdot[0, 1] - R[1, 2] * Rdot[1, 1] - R[2, 2] * Rdot[2, 1],
                R[0, 2] * Rdot[0, 0] + R[1, 2] * Rdot[1, 0] + R[2, 2] * Rdot[2, 0],
                -R[0, 1] * Rdot[0, 0] - R[1, 1] * Rdot[1, 0] - R[2, 1] * Rdot[2, 0],
            ]
        )
        body_spin_rate = float(np.linalg.norm(body_spin_vector))

        radius = self._resolve_body_radius()
        J1 = np.array([0.0, 0.0, (2.0 / 5.0) * radius * radius * body_spin_rate])
        return R @ J1

    def _resolve_body_radius(self) -> float:
        """返回中心天体赤道半径（km）。"""
        if self._body_radius is not None:
            return float(self._body_radius)
        try:
            return self._DEFAULT_BODY_RADII_KM[self._central_body]
        except KeyError as exc:
            raise RelativisticCorrectionError(
                f"No default body radius for {self._central_body!r}. "
                "Provide body_radius explicitly."
            ) from exc

