"""RTN 三轴常值加速度（匀加速度）力模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .physical_model import PhysicalModel


class UniformAcceleration(PhysicalModel):
    """RTN 三轴常值加速度力模型（匀加速度）。

    加速度与航天器质量无关：每个 RHS 求值处以当前状态的 RTN 基把
    ``(aR, aT, aN)`` 旋转到传播坐标系。RTN 定义与 LVLH 方向帧同基
    （ADR 0007 的 RSW 约定）：R = r/|r|，N = (r×v)/|r×v|，T = N×R。
    三分量全零合法（等价于无该力）；退化状态（|r|≈0、|v|≈0，或 r∥v 使
    |r×v|≈0）在传播中显式报错，不静默丢弃分量（ADR 0020）。对应 CE-5
    精密定轨策略的姿轨控匀加速度项。

    加速度计算全部由 Rust 编译路径承载（``("uniform_accel", ...)`` 力元组，
    ``crates/e2m2e-forces/src/forces/compiled.rs``），Python 侧不保留参考实现。

    Args:
        acceleration_rtn: RTN 三轴常值加速度分量 (aR, aT, aN)，km/s²。
        direction_frame: 方向解释坐标系，当前仅支持 ``"RTN"``。
    """

    def __init__(self, acceleration_rtn: npt.ArrayLike, direction_frame: str = "RTN") -> None:
        arr = np.asarray(acceleration_rtn, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"acceleration_rtn must have shape (3,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"acceleration_rtn must be finite, got {arr.tolist()}")
        if direction_frame != "RTN":
            raise ValueError(f"direction_frame must be 'RTN', got {direction_frame!r}")
        self._acceleration_rtn = arr.copy()
        self._direction_frame = direction_frame

    @property
    def acceleration_rtn(self) -> npt.NDArray[np.floating]:
        """RTN 三轴常值加速度分量 (aR, aT, aN) 副本，km/s²。"""
        return self._acceleration_rtn.copy()

    @property
    def direction_frame(self) -> str:
        """方向解释坐标系：当前仅 ``"RTN"``。"""
        return self._direction_frame

    def to_rust_spec(self, system: Any) -> tuple:
        """序列化为 ``("uniform_accel", [aR, aT, aN], direction_frame)``。

        ``system`` 不使用（本力不查星历/引力参数），签名与
        ``FiniteBurn.to_rust_spec`` 的宽松形式一致。
        """
        return ("uniform_accel", self._acceleration_rtn.tolist(), self._direction_frame)
