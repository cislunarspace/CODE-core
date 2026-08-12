"""点质量引力模型。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from ..system import System


class PointMassGravity(PhysicalModel):
    """点质量引力加速度模型。

    返回 :math:`-\\mu / r^3 \\cdot \\mathbf{r}`，即中心天体二体引力加速度。

    Args:
        body: 中心天体名称（如 ``'EARTH'``）。
        mu: 引力参数（km³/s²）。为 ``None`` 时，
            在 ``to_rust_spec`` 中从 ``system.gravitational_parameter(body)`` 获取。
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

    def to_rust_spec(self, system: System) -> tuple | None:
        """序列化为 Rust ``propagate_compiled`` 接受的 ``("point_mass", mu)`` 元组。

        与 ``GravityField`` （degree=0 等价点质量）的 Rust 路径对齐，但更轻量
        （不查 body-fixed 轴、不查星历）。``mu`` 为 ``None`` 时从 system 解析。
        """
        return ("point_mass", self._resolve_mu(system))
