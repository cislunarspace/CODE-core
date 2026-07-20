"""第三体引力间接项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    from ..system import System


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
            在 ``compute_acceleration`` 中从 ``system.gravitational_parameter(body)`` 获取。
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

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """返回间接项加速度，km/s²。

        ``r_body=0`` 时返回零向量，避免除零。
        """
        mu = self._resolve_mu(system)

        # 走 Rust cspice 路径（spice feature 启用时）；否则走 Python spiceypy。
        # 两路数值一致（机器精度），Rust 版避免跨界 + numpy 数组分配。
        # 只在 system 暴露 spice 属性（真实 EphemerisSystem）时走 Rust，
        # 桩 system（如单元测试 mock）回退到 Python 路径。
        if getattr(system, "spice", None) is not None:
            try:
                from e2m2e._integrators import indirect_term_acceleration  # noqa: F401
                from .third_body_gravity import ThirdBodyGravity
                observer = getattr(system, "origin", "EARTH")
                observer_id = ThirdBodyGravity._name_or_id(observer)
                target_id = ThirdBodyGravity._name_or_id(self._body)
                a = indirect_term_acceleration(float(t), target_id, observer_id, float(mu))
                return np.asarray(a, dtype=float)
            except ImportError:
                pass

        r_ob = np.asarray(system.get_body_position(self._body, t), dtype=float)
        n = float(np.linalg.norm(r_ob))
        if n < 1e-6:
            return np.zeros(3)
        return -mu * r_ob / n**3

    def compute_jacobian(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating] | None:
        """间接项不依赖航天器位置，∂a/∂r 恒为零矩阵。"""
        return np.zeros((3, 3))
