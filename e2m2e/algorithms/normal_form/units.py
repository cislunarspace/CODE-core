"""SI ↔ qiao 归一化单位转换。

约定：

- SI 状态 ``[x, y, z, vx, vy, vz]`` 单位为 ``km`` + ``km/s`` （与
  ``CR3BP_System.physical_to_dimensionless`` 物理接口一致）；
- qiao 归一化状态使用 ``LU`` （位置）与 ``VU = LU/TU`` （速度），与
  ``Global_File.py`` 中无量纲状态保持一致；
- 时间方向上 SI 秒 ↔ 归一化 TU 通过 ``NormalFormContext.seconds_to_tu``
  / ``tu_to_seconds`` 完成。

``to_normalized`` 与 ``from_normalized`` 互为精确逆运算（数值精度内
无截断），后续切片中的所有归一化 Hamilton 量都基于此接口之上。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .context import NormalFormContext


def to_normalized(
    state_si: npt.ArrayLike,
    context: NormalFormContext,
) -> npt.NDArray[np.floating]:
    """SI 物理状态 → qiao 归一化状态。

    位置除以 ``context.LU``（km），速度除以 ``context.VU``（km/s）。
    支持 ``(6,)`` 单状态与 ``(n, 6)`` 批量状态。

    Args:
        state_si: 物理状态向量 ``[x, y, z, vx, vy, vz]``，单位 km + km/s。
        context: 归一化上下文。

    Returns:
        与输入同形状的归一化状态数组。

    Raises:
        ValueError: 最后一维不是 6。
    """
    arr = np.asarray(state_si, dtype=float)
    if arr.shape[-1] != 6:
        raise ValueError(f"状态向量必须包含 6 个分量，最后一维为 {arr.shape[-1]}")
    out = arr.copy()
    out[..., :3] = out[..., :3] / context.LU
    out[..., 3:] = out[..., 3:] / context.VU
    return out


def from_normalized(
    state_norm: npt.ArrayLike,
    context: NormalFormContext,
) -> npt.NDArray[np.floating]:
    """qiao 归一化状态 → SI 物理状态。

    位置乘以 ``context.LU``（km），速度乘以 ``context.VU``（km/s）。
    支持 ``(6,)`` 单状态与 ``(n, 6)`` 批量状态。

    Args:
        state_norm: 归一化状态向量 ``[x, y, z, vx, vy, vz]``，无量纲。
        context: 归一化上下文。

    Returns:
        与输入同形状的物理状态数组，单位 km + km/s。

    Raises:
        ValueError: 最后一维不是 6。
    """
    arr = np.asarray(state_norm, dtype=float)
    if arr.shape[-1] != 6:
        raise ValueError(f"状态向量必须包含 6 个分量，最后一维为 {arr.shape[-1]}")
    out = arr.copy()
    out[..., :3] = out[..., :3] * context.LU
    out[..., 3:] = out[..., 3:] * context.VU
    return out


__all__ = ["to_normalized", "from_normalized"]
