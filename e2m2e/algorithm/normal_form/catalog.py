"""表征参数目录变换器（issue #174，切片 5）。

把切片 #171–#173 的三个预计算结果（动力学替代 ``W_poly``、quasi-Floquet
``B(t)``、中心流形 ``W_series``）绑定到一个**不可变聚合句柄**
:class:`LibrationCatalogData`，并提供面向对象的访问入口
:class:`LibrationCatalogTransformer`，实现 qiao ``rho2param`` /
``param2rho`` 的完整坐标变换链 ``rho ↔ param``。

变换数学完全委托给 :mod:`.coord_trans` 子包（5 段函数式叶子函数 +
端到端链式组合）。本模块的职责是：

1. **聚合**：把 ``DynamicalSubstituteResult`` / ``QuasiFloquetResult`` /
   ``CenterManifoldResult`` 三个句柄 + :class:`NormalFormContext` 打包成
   一个数据类，避免调用方每次变换都要传 4 个参数；
2. **时间插值封装**：``rho_to_param`` / ``param_to_rho`` 内部对 ``W`` /
   ``B`` / ``W_series`` 做时刻 ``t`` 的插值（由叶子函数完成），调用方只
   传状态和时间。

与 #175（``pipeline.py`` 最终 ``NormalFormResult``）的衔接取舍：

- :mod:`.types.NormalFormResult` （切片 0）是一个**通用流水线结果容器**
  （Hamiltonian 系数、变换矩阵、残差等），字段语义偏"化简结果"，不适合
  直接当坐标变换的系数聚合器；
- 本切片定义独立的 :class:`LibrationCatalogData` 作为**坐标变换专用聚合
  句柄**（只持有三个子结果 + context），命名上避开 ``NormalFormResult``
  以免与 #175 冲突；
- **#175 可以直接复用本类**：``LibrationCatalogTransformer`` 的构造只
  依赖 ``context`` / ``ds_result`` / ``qf_result`` / ``cm_result`` 四个
  访问器，#175 的 ``NormalFormResult`` 只需暴露这四个属性即可无缝构造
  ``LibrationCatalogData``（或直接传 ``LibrationCatalogData`` 当字段）。
  本切片不在 ``types.NormalFormResult`` 上加字段，保持切片 0 容器稳定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .coord_trans import param_to_rho, rho_to_param

if TYPE_CHECKING:
    from .center_manifold import CenterManifoldResult
    from .context import NormalFormContext
    from .dynamical_substitution import DynamicalSubstituteResult
    from .quasi_floquet import QuasiFloquetResult


# ---------------------------------------------------------------------------
# 聚合句柄
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibrationCatalogData:
    """表征参数目录变换所需的预计算系数聚合句柄。

    不可变：把切片 #171–#173 的三个结果 + 上下文打包，供
    :class:`LibrationCatalogTransformer` 绑定使用。

    设计取舍见模块 docstring：本类是**坐标变换专用**聚合器，不复用
    :class:`~e2m2e.algorithms.normal_form.types.NormalFormResult`（后者是
    通用流水线结果容器）。#175 的最终 ``NormalFormResult`` 可暴露
    ``context``/``ds_result``/``qf_result``/``cm_result`` 四个属性后直接
    构造本类，或反过来把本类当字段嵌入。

    Attributes:
        context: 归一化上下文。
        ds_result: 动力学替代结果（提供 ``W_poly``、``tlist``）。
        qf_result: quasi-Floquet 结果（提供 ``B_at(t)``）。
        cm_result: 中心流形化简结果（提供 ``W_series``）。
    """

    context: NormalFormContext
    ds_result: DynamicalSubstituteResult
    qf_result: QuasiFloquetResult
    cm_result: CenterManifoldResult


# ---------------------------------------------------------------------------
# 变换器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibrationCatalogTransformer:
    """表征参数目录变换器（上下文 + 系数绑定）。

    把 :class:`LibrationCatalogData` 绑定到方法，提供面向对象的
    ``rho_to_param`` / ``param_to_rho`` 入口。所有时刻 ``t`` 的插值由
    :mod:`.coord_trans` 叶子函数在内部完成，调用方只传状态与时间。

    Args:
        data: 预计算系数聚合句柄（context + 三个子结果）。
    """

    data: LibrationCatalogData

    # 便捷透传 ----------------------------------------------------------
    @property
    def context(self) -> NormalFormContext:
        """绑定上下文（透传到 ``data.context``）。"""
        return self.data.context

    # ------------------------------------------------------------------
    # 公开变换
    # ------------------------------------------------------------------

    def rho_to_param(self, X_rho: npt.ArrayLike, t: float) -> npt.NDArray[np.floating]:
        """rho 坐标 → 表征参数（完整逆链）。

        对应 qiao ``rho2param``：``rho → EM → DS → QF → CM → param``。

        Args:
            X_rho: ``(6,)`` rho 状态 ``[ρ, ρ̇]``，无量纲。
            t: 归一化时间 TU。

        Returns:
            ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。
        """
        return rho_to_param(
            X_rho,
            t,
            self.data.context,
            self.data.ds_result,
            self.data.qf_result,
            self.data.cm_result,
        )

    def param_to_rho(self, X_param: npt.ArrayLike, t: float) -> npt.NDArray[np.floating]:
        """表征参数 → rho 坐标（完整正链）。

        对应 qiao ``param2rho``：``param → CM → QF → DS → EM → rho``。
        是 :meth:`rho_to_param` 的精确逆。

        Args:
            X_param: ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。
            t: 归一化时间 TU。

        Returns:
            ``(6,)`` rho 状态 ``[ρ, ρ̇]``，无量纲。
        """
        return param_to_rho(
            X_param,
            t,
            self.data.context,
            self.data.ds_result,
            self.data.qf_result,
            self.data.cm_result,
        )


__all__ = [
    "LibrationCatalogData",
    "LibrationCatalogTransformer",
]
