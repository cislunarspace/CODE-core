"""点质量引力模型。"""

from __future__ import annotations

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
        mu = self._mu
        if mu is None:
            if system is None:
                raise ValueError(
                    "mu is None and system is None; cannot resolve gravitational_parameter"
                )
            mu = system.gravitational_parameter(self._body)

        r = np.asarray(state, dtype=float)[:3]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-15:
            return np.zeros(3)
        return -mu / (r_norm**3) * r
