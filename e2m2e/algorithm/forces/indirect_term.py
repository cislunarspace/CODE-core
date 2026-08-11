"""第三体引力间接项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    pass


class IndirectTerm(PhysicalModel):
    """第三体引力的间接项（geocentric 加速系所需）。

    在以某天体（如地球）为原点的非惯性系下传播时，运动方程需对每个摄动
    天体 :math:`i` 补一项 ``-μ_i · r_i / |r_i|³``（间接项），扣除摄动天体
    对原点的引力（见 ``EphemerisDynamics`` 的 N 体闭式公式）。

    ``ThirdBodyGravity`` 内部已自带间接项，但 ``GravityField`` 只算球谐
    直接引力（含中心项 degree=0），不带间接项。所以用 ``GravityField``
    模拟月球（中心+非球形）时，必须单独补月球间接项——既不能用
    ``ThirdBodyGravity("MOON")``（会与 ``GravityField`` 的 degree=0 中心项
    重复算月球点质量），也不能省略（地心系下物理不正确）。

    加速度：``-μ_body · r_body / |r_body|³``，其中 ``r_body`` 为摄动天体相对
    ``system.origin`` 的位置（由 ``system.get_body_position`` 自动以 origin
    为观察者计算）。与 ``ThirdBodyGravity`` 的间接项逐字一致。

    Args:
        body: 摄动天体名称（如 ``'MOON'``）。
        mu: 引力参数（km³/s²）。为 ``None`` 时，
            在 ``to_rust_spec`` 中从 ``system.gravitational_parameter(body)`` 获取。
    """

    def __init__(self, body: str, mu: float | None = None) -> None:
        self._body = body.upper()
        self._mu = float(mu) if mu is not None else None

    @property
    def body(self) -> str:
        """摄动天体名称。"""
        return self._body

    @property
    def mu(self) -> float | None:
        """显式设置的引力参数；``None`` 表示从 system 获取。"""
        return self._mu

    def to_rust_spec(self, system) -> tuple | None:
        """序列化为 ``("indirect", naif_id_str, mu)``。"""
        from .third_body_gravity import ThirdBodyGravity

        mu = self._resolve_mu(system)
        naif_id = ThirdBodyGravity._name_or_id(self._body)
        return ("indirect", naif_id, float(mu))
