"""相对论修正力模型。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.constants import SPEED_OF_LIGHT_KMS
from .physical_model import PhysicalModel


class RelativisticCorrection(PhysicalModel):
    """相对论修正力模型。

    实现 Schwarzschild、Lense-Thirring 与 de Sitter（geodesic）三项相对论
    加速度修正，公式与 GMAT R2026a ``RelativisticCorrection`` 对齐。

    加速度计算全部由 Rust 编译路径承载（``("relativistic", ...)`` 力元组，
    ``crates/e2m2e-forces/src/forces/relativistic.rs``），Python 侧不保留参考
    实现。
    """

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
        c: float = SPEED_OF_LIGHT_KMS,
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

        - LT 项需要 sxform + body-fixed frame；NRHO 上 LT 量级 < 1m，
          但完整移植已实现（含 sxform via cspice-sys FFI）。
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
