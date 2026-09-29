"""逐 leg 传播内核（conic / ephemeris 双档，ADR 0050 → ADR 0053 框架移交）。

半段传播与 TOF 灵敏度链 RHS 原属 ``sims_flanagan.py``（#740/#741/#727），
#726 抽象为通用多段框架时整体搬移至此；:class:`LegKernel` 把两档封装为
逐 leg 可独立选择的传播内核（逐 leg 混档 = 每 leg 一个实例）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from e2m2e.integrators import propagate_kepler_py

from ...dynamics import EphemerisDynamics

__all__ = ["LegKernel", "propagate_half", "tier_rhs"]


def propagate_half(
    x: npt.NDArray[np.floating],
    t_start: float | None,
    half: float,
    *,
    mu_km3_s2: float,
    dyn: EphemerisDynamics | None,
    with_stm: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
    """半段传播内核：conic 封闭解 / 星历 N 体数值传播（#727 双档）。

    conic 档忽略 ``t_start``（二体时间不变）；星历档在绝对历元
    ``[t_start, t_start + half]`` 上数值传播（``half < 0`` 即后向）。
    返回 ``(x_out, Φ(6,6) 或 None)``。
    """
    if dyn is None:
        out = propagate_kepler_py(x.tolist(), [half], mu_km3_s2, with_stm=with_stm)
        x_out = np.asarray(out["states"][0], dtype=float)
        phi = np.asarray(out["stm"][0], dtype=float).reshape(6, 6) if with_stm else None
    else:
        if t_start is None:
            raise ValueError("ephemeris 档传播必须提供绝对历元 t_start")
        out = dyn.propagate(
            x, (t_start, t_start + half), t_eval=[t_start + half], with_stm=with_stm
        )
        x_out = np.asarray(out["states"][-1], dtype=float)
        phi = np.asarray(out["stm"][-1], dtype=float) if with_stm else None
    return x_out, phi


def tier_rhs(
    t: float | None,
    x: npt.NDArray[np.floating],
    *,
    mu_km3_s2: float,
    dyn: EphemerisDynamics | None,
) -> npt.NDArray[np.floating]:
    """TOF 灵敏度链 RHS：conic 二体 ``[v; −μr/|r|³]`` / 星历 ``equations_of_motion(t, x)``。"""
    if dyn is None:
        r = x[:3]
        return np.concatenate([x[3:], -mu_km3_s2 * r / float(np.linalg.norm(r)) ** 3])
    if t is None:
        raise ValueError("ephemeris 档 RHS 必须提供绝对历元 t")
    return np.asarray(dyn.equations_of_motion(t, x), dtype=float)


@dataclass(frozen=True)
class LegKernel:
    """单 leg 的传播内核与保真度档（conic 只给 ``mu``；星历再给 ``dyn``）。

    星历档 ``dyn`` 按鸭子类型使用（``propagate`` / ``equations_of_motion``），
    测试可用解析 stub 替换；conic 档 ``dyn=None`` 走闭式 Kepler。

    Attributes:
        mu_km3_s2: 中心天体引力常数（km³/s²），必须为正。
        dyn: 星历 N 体动力学；``None`` 为 conic 档（二体封闭解）。
    """

    mu_km3_s2: float
    dyn: EphemerisDynamics | None = None

    @property
    def absolute_time(self) -> bool:
        """是否依赖绝对历元（星历档 True；conic 档时间不变，False）。"""
        return self.dyn is not None

    def propagate_half(
        self,
        x: npt.NDArray[np.floating],
        t_start: float | None,
        half: float,
        *,
        with_stm: bool,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """转发 :func:`propagate_half`（本 leg 的 mu/dyn 绑定）。"""
        return propagate_half(
            x, t_start, half, mu_km3_s2=self.mu_km3_s2, dyn=self.dyn, with_stm=with_stm
        )

    def rhs(self, t: float | None, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """转发 :func:`tier_rhs`（本 leg 的 mu/dyn 绑定）。"""
        return tier_rhs(t, x, mu_km3_s2=self.mu_km3_s2, dyn=self.dyn)
