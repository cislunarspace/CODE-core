"""DS ↔ QF：动力学替代坐标 ↔ quasi-Floquet 坐标。

迁移自 qiao ``Subfunction/coord_trans/qpDS2qpQF.py`` /
``qpQF2qpDS.py``。对应变换链第三段：DS 坐标与 quasi-Floquet 坐标
之间的**矩阵变换**。

数学关系（qiao ``CONTEXT.md`` §三"quasi-Floquet 变换"）：

    X_QF = B(t)⁻¹ · X_DS        （正向 DS → QF）
    X_DS = B(t) · X_QF           （反向 QF → DS）

其中 ``B(t)`` 是 quasi-Floquet 变换求解得到的变换矩阵（6×6 辛矩阵），
把受迫时变线性化系统化为常系数的 ``Ẏ = D·Y``。本段是纯线性变换，
无量纲、无单位换算。

与 qiao 的差异：

- qiao 通过 ``get_QFmat(t, QFtrans_mat)`` 在 ``globalparam.data_array``
  上插值 ``B(t)``；本仓库的 :class:`QuasiFloquetResult.B_at` 已提供线性
  插值访问器，本模块直接接收插值后的 ``B_at_t``，把"何时插值"的决策
  上浮到 :class:`LibrationCatalogTransformer` （与 EM/CM 段保持一致）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ds_to_qf(
    X_ds: npt.ArrayLike,
    B_at_t: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """动力学替代坐标 → quasi-Floquet 坐标（矩阵变换）。

    ``X_QF = B⁻¹ · X_DS``。对应 qiao ``qpDS2qpQF``。

    Args:
        X_ds: ``(6,)`` DS 状态 ``[Q, P]``，无量纲。
        B_at_t: ``(6, 6)`` 在时刻 ``t`` 插值后的 quasi-Floquet 矩阵
            ``B(t)`` （辛）。

    Returns:
        ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲。

    Raises:
        numpy.linalg.LinAlgError: ``B(t)`` 奇异。
    """
    X = np.asarray(X_ds, dtype=float).ravel()
    B = np.asarray(B_at_t, dtype=float)
    X_qf = np.linalg.solve(B, X)
    return X_qf


def qf_to_ds(
    X_qf: npt.ArrayLike,
    B_at_t: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """quasi-Floquet 坐标 → 动力学替代坐标（矩阵变换）。

    ``X_DS = B · X_QF``。对应 qiao ``qpQF2qpDS``。是 :func:`ds_to_qf`
    的精确逆。

    Args:
        X_qf: ``(6,)`` QF 状态 ``[Q_qf, P_qf]``，无量纲。
        B_at_t: ``(6, 6)`` 在时刻 ``t`` 插值后的 quasi-Floquet 矩阵
            ``B(t)`` （辛）。

    Returns:
        ``(6,)`` DS 状态 ``[Q, P]``，无量纲。
    """
    X = np.asarray(X_qf, dtype=float).ravel()
    B = np.asarray(B_at_t, dtype=float)
    X_ds = B @ X
    return X_ds


__all__ = ["ds_to_qf", "qf_to_ds"]
