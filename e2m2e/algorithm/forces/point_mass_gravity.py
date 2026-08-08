"""点质量引力模型。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from ..system import System


class PointMassGravity(PhysicalModel):
    """点质量引力加速度模型。

    返回 :math:`-\\mu / r^3 \\cdot \\mathbf{r}`，即中心天体二体引力加速度。

    Args:
        body: 中心天体名称（如 ``'EARTH'``）。
        mu: 引力参数（km³/s²）。为 ``None`` 时，
            在 ``compute_acceleration`` 中从 ``system.gravitational_parameter(body)`` 获取。
    """

    def __init__(self, body: str, mu: float | None = None) -> None:
        self._body = body.upper()
        self._mu = float(mu) if mu is not None else None

    @property
    def body(self) -> str:
        """中心天体名称。"""
        return self._body

    @property
    def mu(self) -> float | None:
        """显式设置的引力参数；``None`` 表示从 system 获取。"""
        return self._mu

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """返回引力加速度，km/s²。

        ``r=0`` 时返回零向量，避免除零。
        """
        warnings.warn(
            f"{self.__class__.__name__}.compute_acceleration 走 Python 回退路径，"
            "应优先走 Rust 编译路径。",
            DeprecationWarning,
            stacklevel=2,
        )
        mu = self._resolve_mu(system)

        r = np.asarray(state, dtype=float)[:3]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-15:
            return np.zeros(3)
        return -mu / (r_norm**3) * r

    def compute_jacobian(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating] | None:
        """返回中心引力加速度对位置的偏导 ∂a/∂r（3×3）。

        公式：``-μ(I/r³ - 3 r rᵀ/r⁵)``，与 ``EphemerisDynamics`` 中心天体
        分支逐字一致。``r=0`` 时返回零矩阵。
        """
        mu = self._resolve_mu(system)

        r = np.asarray(state, dtype=float)[:3]
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-15:
            return np.zeros((3, 3))
        mu_r3 = mu / r_norm**3
        return -mu_r3 * (np.eye(3) - 3.0 * np.outer(r, r) / (r_norm**2))

    def to_rust_spec(self, system: System) -> tuple | None:
        """序列化为 Rust ``propagate_compiled`` 接受的 ``("point_mass", mu)`` 元组。

        与 ``GravityField``（degree=0 等价点质量）的 Rust 路径对齐，但更轻量
        （不查 body-fixed 轴、不查星历）。``mu`` 为 ``None`` 时从 system 解析。
        """
        return ("point_mass", self._resolve_mu(system))
