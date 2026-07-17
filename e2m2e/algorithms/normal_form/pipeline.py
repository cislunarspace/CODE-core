"""标准形化简一键式流水线（issue #175）。

把前面四个切片的 reducer 串成一条完整路径：

    星历轨道初值
        │  DynamicalSubstituteCorrector  （切片 #171）
        ▼
    动力学替代轨道 + 生成函数 W
        │  QuasiFloquetReducer           （切片 #172）
        ▼
    quasi-Floquet 变换 B(t) + 实标准形 D
        │  CenterManifoldReducer         （切片 #173）
        ▼
    中心流形化简 W_series
        │  LibrationCatalogTransformer   （切片 #174）
        ▼
    表征参数目录变换器（rho ↔ param）

外部用户一行代码即可完成"星历轨道 → 表征参数"：

.. code-block:: python

   result = NormalFormPipeline(context).reduce(x0)
   param = result.catalog_transformer.rho_to_param(x0, t=0.0)

返回的 :class:`NormalFormResult` 既是通用化简诊断容器（Hamiltonian 系数、
残差、收敛标志），也把四个子结果句柄作为一等公民字段暴露，使下游无需自行
重组装 :class:`LibrationCatalogData` 即可做坐标变换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .center_manifold import CenterManifoldReducer
from .dynamical_substitution import (
    DEFAULT_DENSE_STEP,
    DEFAULT_NODE_STEP,
    DEFAULT_TOLERANCE,
    DEFAULT_TOTAL_TU,
    DynamicalSubstituteCorrector,
)
from .quasi_floquet import QuasiFloquetReducer
from .types import NormalFormResult

if TYPE_CHECKING:
    from .catalog import LibrationCatalogTransformer
    from .center_manifold import CenterManifoldResult
    from .context import NormalFormContext
    from .dynamical_substitution import DynamicalSubstituteResult
    from .quasi_floquet import QuasiFloquetResult

__all__ = ["NormalFormPipeline"]


@dataclass(frozen=True)
class NormalFormPipeline:
    """标准形化简一键式流水线（上下文绑定）。

    通过 :meth:`reduce` 把一条 ``(6,)`` rho 坐标初值依次送进动力学替代
    corrector、quasi-Floquet reducer、中心流形 reducer，最后绑定成表征
    参数目录变换器，返回聚合的 :class:`NormalFormResult`。

    只暴露用户真正需要的旋钮（quasi-Floquet 求解法、中心流形截断阶与步
    序列）；动力学替代的打靶窗口/容差沿用 :class:`DynamicalSubstituteCorrector`
    的默认值（与 qiao ``Code05`` 一致），需要覆盖时传 ``dynamical_kwargs``。

    Args:
        context: 归一化上下文（提供平动点、频率、历元等）。
        quasi_floquet_method: quasi-Floquet 求解法，``"matrix"``（默认，
            36 维直接积分 + 辛投影）或 ``"lie_algebra"``（21 维 sp(6)
            参数化，自动保辛）。
        center_max_order: 中心流形 Lie 变换截断阶数，默认 ``10``（与 qiao
            ``Code10``/``Code11`` 一致）。
        center_steps: 中心流形化简步骤元组，默认 ``("invariant", "center")``。
        dynamical_kwargs: 透传给 :class:`DynamicalSubstituteCorrector` 的
            覆盖项（如 ``{"t_total": 8.0, "node_step": 0.8}``）。``None``
            时用该 corrector 的全部默认值。
    """

    context: NormalFormContext
    quasi_floquet_method: str = "matrix"
    center_max_order: int = 10
    center_steps: tuple[str, ...] = ("invariant", "center")
    dynamical_kwargs: dict[str, object] = field(default_factory=dict)

    def reduce(self, orbit: npt.ArrayLike) -> NormalFormResult:
        """对 rho 坐标初值跑完整标准形化简流水线。

        Args:
            orbit: ``(6,)`` rho 坐标初始状态 ``[ρ, ρ̇]``（无量纲），作为
                动力学替代 corrector 的种子。SPICE 内核不可用时底层自动
                降级到纯 CR3BP（仅供烟雾测试）。

        Returns:
            :class:`NormalFormResult`：聚合了四个子结果句柄与通用化简诊断
            字段。``catalog_transformer`` 字段在四步全部成功后非 ``None``。

        Notes:
            任一步抛异常时，流水线不向上传播，而是把已完成的子结果填进
            :class:`NormalFormResult`，置 ``success=False``、``message``
            记录失败步骤与原因——这样部分失败也保留诊断价值。输入校验
            （如 orbit 形状非法）仍直接抛 :class:`ValueError`，因为这是
            调用方错误而非流水线内部失败。
        """
        x0 = self._normalize_orbit(orbit)

        ds_result: DynamicalSubstituteResult | None = None
        qf_result: QuasiFloquetResult | None = None
        cm_result: CenterManifoldResult | None = None
        catalog_transformer: LibrationCatalogTransformer | None = None

        # —— 步骤 1：动力学替代 ——
        ds_kwargs: dict[str, Any] = {
            "t_total": DEFAULT_TOTAL_TU,
            "node_step": DEFAULT_NODE_STEP,
            "dense_step": DEFAULT_DENSE_STEP,
            "tolerance": DEFAULT_TOLERANCE,
            "spice_optional": True,
        }
        ds_kwargs.update(self.dynamical_kwargs)
        try:
            ds_corrector = DynamicalSubstituteCorrector(context=self.context, **ds_kwargs)
            ds_result = ds_corrector.reduce(seed=x0)
        except Exception as exc:
            return self._failure("dynamical_substitution", exc, ds_result, qf_result, cm_result)

        # —— 步骤 2：quasi-Floquet ——
        try:
            qf_reducer = QuasiFloquetReducer(context=self.context, method=self.quasi_floquet_method)
            qf_result = qf_reducer.reduce(ds_result)
        except Exception as exc:
            return self._failure("quasi_floquet", exc, ds_result, qf_result, cm_result)

        # —— 步骤 3：中心流形化简 ——
        try:
            cm_reducer = CenterManifoldReducer(
                context=self.context, max_order=self.center_max_order
            )
            cm_result = cm_reducer.reduce(qf_result, steps=self.center_steps)
        except Exception as exc:
            return self._failure("center_manifold", exc, ds_result, qf_result, cm_result)

        # —— 步骤 4：表征参数目录变换器 ——
        try:
            from .catalog import LibrationCatalogData, LibrationCatalogTransformer

            data = LibrationCatalogData(
                context=self.context,
                ds_result=ds_result,
                qf_result=qf_result,
                cm_result=cm_result,
            )
            catalog_transformer = LibrationCatalogTransformer(data=data)
        except Exception as exc:
            return self._failure("catalog", exc, ds_result, qf_result, cm_result)

        return NormalFormResult(
            context=self.context,
            order=int(self.context.order),
            substitute_residual=float(ds_result.residual_norm),
            success=True,
            message="流水线四步全部完成",
            metadata={
                "quasi_floquet_method": self.quasi_floquet_method,
                "center_max_order": int(self.center_max_order),
                "center_steps": tuple(self.center_steps),
                "spice_available": bool(ds_result.spice_available),
                "ds_backend": ds_result.backend,
                "qf_symplectic_error": float(qf_result.max_symplectic_error),
                "cm_hyperbolic_coupling": float(cm_result.max_hyperbolic_coupling),
            },
            ds_result=ds_result,
            qf_result=qf_result,
            cm_result=cm_result,
            catalog_transformer=catalog_transformer,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _normalize_orbit(self, orbit: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """把入参归一化为 ``(6,)`` rho 状态数组。

        接受 ``(6,)`` 数组；Orbit-like 对象（带 ``.states``）取首帧。
        形状非法时抛 :class:`ValueError`（调用方错误，直接上抛）。
        """
        if hasattr(orbit, "states"):
            arr = np.asarray(orbit.states, dtype=float)
            if arr.ndim == 2:
                arr = arr[0]
        else:
            arr = np.asarray(orbit, dtype=float).ravel()
        if arr.shape != (6,):
            raise ValueError(f"orbit 必须是形状 (6,) 的 rho 状态，得到 {arr.shape}")
        return arr

    def _failure(
        self,
        step: str,
        exc: BaseException,
        ds_result: DynamicalSubstituteResult | None,
        qf_result: QuasiFloquetResult | None,
        cm_result: CenterManifoldResult | None,
    ) -> NormalFormResult:
        """构造失败结果：保留已完成的子结果，记录失败原因。"""
        return NormalFormResult(
            context=self.context,
            order=int(self.context.order),
            success=False,
            message=f"步骤 {step!r} 失败：{type(exc).__name__}: {exc}",
            metadata={"failed_step": step, "exception_type": type(exc).__name__},
            ds_result=ds_result,
            qf_result=qf_result,
            cm_result=cm_result,
            catalog_transformer=None,
        )
