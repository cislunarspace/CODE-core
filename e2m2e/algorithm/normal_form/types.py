"""Normal-form 流水线的结果类型。

``NormalFormResult`` 是 :class:`NormalFormPipeline` （issue #175）输出的统一
载体：把前四个切片（动力学替代 / quasi-Floquet / 中心流形 / 表征参数目录）
的结果聚合到一个不可变句柄里。通用化简诊断字段（Hamiltonian 系数、变换
矩阵、残差）与各子结果句柄并存——前者供算法诊断，后者让外部用户一行代码
拿到 ``catalog_transformer`` 完成坐标变换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ..results import ResultStatus

if TYPE_CHECKING:
    from .catalog import LibrationCatalogTransformer
    from .center_manifold import CenterManifoldResult
    from .context import NormalFormContext
    from .dynamical_substitution import DynamicalSubstituteResult
    from .quasi_floquet import QuasiFloquetResult


@dataclass(frozen=True)
class NormalFormResult:
    """标准形化简流水线统一结果容器。

    不可变：所有字段在构造后只读，确保下游消费方按值传递。

    字段分两组：

    - **通用化简诊断** （``substitute_residual``、``success``、``message``、
      ``metadata``）：跨切片稳定，描述整条流水线的收敛情况。保留给仅关心
      "是否收敛、残差多大"的诊断调用方。
    - **子结果句柄** （``ds_result`` / ``qf_result`` / ``cm_result`` /
      ``catalog_transformer``）：issue #175 新增。指向四个子 reducer 的产物；
      ``catalog_transformer`` 一等公民字段使外部用户能直接
      ``result.catalog_transformer.rho_to_param(X_rho, t)`` 完成完整坐标变换，
      无需自己重组装 :class:`LibrationCatalogData`。

    所有新增字段都带默认值，保证仅关心通用字段的既有调用方
    （``NormalFormResult(context, order, ...)``）不受影响。

    Attributes:
        context: 关联的 ``NormalFormContext``。
        order: 实际展开阶数（一般等于 ``context.order``）。
        substitute_residual: 动力学替代步（多重打靶）的段间连续性残差
            （来自 ``ds_result.residual_norm``）。注意它不是 Lie 变换的
            截断残差——后者当前写入 ``metadata["cm_hyperbolic_coupling"]``。
            ``residual`` 是其向后兼容别名。
        success: 流水线是否在容差内收敛。
        message: 人类可读的终止原因。
        metadata: 自由扩展字段；保留供后续切片写入诊断数据。
        ds_result: 动力学替代结果（切片 #171）；流水线未跑到该步时为 ``None``。
        qf_result: quasi-Floquet 结果（切片 #172）；同上。
        cm_result: 中心流形化简结果（切片 #173）；同上。
        catalog_transformer: 表征参数目录变换器（切片 #174），
            绑定 ds_result、qf_result、cm_result 三个子
            结果与 context；流水线跑完四步后非 ``None``，是外部用户做
            ``rho ↔ param`` 坐标变换的入口。
    """

    context: NormalFormContext
    order: int
    substitute_residual: float = 0.0
    status: ConvergenceState = ConvergenceState.FAILED
    cause: FailureCause = FailureCause.UNKNOWN
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    ds_result: DynamicalSubstituteResult | None = None
    qf_result: QuasiFloquetResult | None = None
    cm_result: CenterManifoldResult | None = None
    catalog_transformer: LibrationCatalogTransformer | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    @property
    def residual(self) -> float:
        """``substitute_residual`` 的向后兼容别名。

        早期版本该字段名 ``residual``，但 docstring 描述的"截断残差"与实际
        写入的"动力学替代连续性残差"语义不符（见 #224）。新代码应用
        ``substitute_residual``；本 property 仅为不破坏既有消费者保留。
        """
        return self.substitute_residual

    # ------------------------------------------------------------------
    # 序列化：save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """把结果存为 ``.npz`` 文件。

        所有子结果（DS / QF / CM）及 context 参数一并序列化；
        ``catalog_transformer`` 反序列化时由三个子结果重建，不单独存储。

        Args:
            path: 输出 ``.npz`` 文件路径；父目录自动创建。
        """
        from ._serial import result_to_npz_dict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **result_to_npz_dict(self))

    @classmethod
    def load(cls, path: str | Path) -> NormalFormResult:
        """从 ``.npz`` 文件重建 :class:`NormalFormResult`。

        反序列化后 ``catalog_transformer`` 自动由三个子结果重建，
        与原始对象在 ``rho_to_param`` / ``param_to_rho`` 上数值等价。

        Args:
            path: ``.npz`` 文件路径。

        Returns:
            重建的 :class:`NormalFormResult`。
        """
        from ._serial import result_from_npz_dict

        data = np.load(path, allow_pickle=False)
        d = {k: data[k] for k in data.files}
        return result_from_npz_dict(d)


__all__ = ["NormalFormResult"]
