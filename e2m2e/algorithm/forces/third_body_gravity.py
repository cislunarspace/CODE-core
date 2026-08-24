"""第三体引力摄动模型。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physical_model import PhysicalModel

if TYPE_CHECKING:
    pass


class ThirdBodyGravity(PhysicalModel):
    """第三体引力摄动加速度模型。

    对一个摄动天体 :math:`i`，返回相对原点天体 :math:`P_0` 的第三体摄动加速度：

    .. math::

        a = -\\mu_i \\left[
              \\frac{r - r_i}{|r - r_i|^3} + \\frac{r_i}{|r_i|^3}
            \\right]

    其中 :math:`r` 为航天器相对原点的位置，:math:`r_i` 为摄动天体相对原点
    的位置（由 ``system.get_body_position`` 自动以 ``system.origin`` 为观察者
    计算）。第一项为直接项（摄动天体对航天器的引力），第二项为间接项
    （扣除摄动天体对原点的引力），与 ``EphemerisDynamics`` 的第三体分支
    逐字对齐。

    Args:
        body: 摄动天体名称（如 ``'MOON'``、``'SUN'``）。
        mu: 引力参数（km³/s²）。为 ``None`` 时，在 ``to_rust_spec`` 中
            从 ``system.gravitational_parameter(body)`` 获取。
    """

    #: 防止除零的最小距离钳位（km，约 1 米），与 EphemerisDynamics.MIN_DISTANCE 一致。
    MIN_DISTANCE = 1e-6

    @staticmethod
    def _name_or_id(name: str) -> str:
        """把天体名转 NAIF ID 字符串（cspice 0.1 无 boddef）。

        优先用 spiceypy.bods2c（识别 boddef 注册过的天体）；未注册（spiceypy
        抛 ``SpiceyError``）则原样返回（DE430 内置名 MOON/EARTH/SUN 等仍
        可用）。只 catch spiceypy 错误（``SpiceyError``），不吞编程错误
        （#352）：bods2c 的意外错误需原样上抛，不被 ``except Exception`` 吞掉。
        """
        try:
            import spiceypy as _spiceypy
        except ImportError:
            # spiceypy 未安装：原样返回（cspice 0.1 无 boddef，名字直传即可）
            return name
        try:
            naif_id = _spiceypy.bods2c(name)
        except _spiceypy.utils.exceptions.SpiceyError:
            # 名字未在 boddef 注册：原样返回
            return name
        if naif_id > 0:
            return str(naif_id)
        return name

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
        """序列化为 ``("third_body", naif_id_str, mu)``。"""
        mu = self._resolve_mu(system)
        naif_id = self._name_or_id(self._body)
        return ("third_body", naif_id, float(mu))
