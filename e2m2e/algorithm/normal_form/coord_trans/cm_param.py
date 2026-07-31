"""CM ↔ param：中心流形坐标 ↔ 表征参数（复→极坐标，作用量-角变量）。

迁移自 qiao ``Subfunction/coord_trans/param2ActionAngle.py``（仅
方向：CM → 作用量-角变量）+ 本仓库补齐的逆变换。对应变换链的最后一段：
把中心流形坐标 ``(q1, q2, q3, p1, p2, p3)`` 转成表征参数
``(q1, p1, I2, θ2, I3, θ3)``。

数学关系：

- **双曲方向** ``(q1, p1)`` 保持原状——双曲方向不是循环坐标，无作用量-
  角变量表示，表征参数里直接用 ``(q1, p1)`` 携带；
- **平面中心方向** ``(q2, p2)``：``I2 = (q2² + p2²)/2``，
  ``θ2 = atan2(p2, q2)``；
- **垂直中心方向** ``(q3, p3)``：``I3 = (q3² + p3²)/2``，
  ``θ3 = atan2(p3, q3)``。

逆变换用 ``q = √(2I)·cos θ``、``p = √(2I)·sin θ`` 还原中心对。

约定说明：

- qiao ``param2ActionAngle`` 用 ``atan2(q2, p2)``（参数交换），本仓库
  按 PRD 显式约定 ``atan2(p2, q2)``——二者互为 ``π/2 − θ`` 的镜像，但
  只要往返变换内部自洽（``θ2`` 算出来再用同公式反解），往返误差仍为
  机器精度。本模块在 docstring/常量上显式标注约定，避免与 qiao 混淆。
- 角变量 ``θ`` 不做 ``% 2π`` 折回：表征参数保留原始连续角，往返时
  ``atan2`` 的主值分支自动与 ``cos/sin`` 配对，无需手动折回。若调用方
  需要折回，可对返回值自行 ``% (2π)``。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cm_to_param(X_cm: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """中心流形坐标 → 表征参数 (复→极坐标)。

    ``X_cm = [q1, q2, q3, p1, p2, p3]`` →
    ``X_param = [q1, p1, I2, θ2, I3, θ3]``。

    Args:
        X_cm: ``(6,)`` CM 状态 ``[q1, q2, q3, p1, p2, p3]``，无量纲实数。

    Returns:
        ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。
    """
    X = np.asarray(X_cm, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_cm 必须是 (6,)，得到 {X.shape}")
    q1, q2, q3 = X[0], X[1], X[2]
    p1, p2, p3 = X[3], X[4], X[5]
    I2 = 0.5 * (q2 * q2 + p2 * p2)
    theta2 = np.arctan2(p2, q2)
    I3 = 0.5 * (q3 * q3 + p3 * p3)
    theta3 = np.arctan2(p3, q3)
    return np.array([q1, p1, I2, theta2, I3, theta3], dtype=float)


def param_to_cm(X_param: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """表征参数 → 中心流形坐标 (极→复坐标)。

    ``X_param = [q1, p1, I2, θ2, I3, θ3]`` →
    ``X_cm = [q1, q2, q3, p1, p2, p3]``。是 :func:`cm_to_param` 的精确逆。

    中心对还原：``q = √(2I)·cos θ``、``p = √(2I)·sin θ``。

    Args:
        X_param: ``(6,)`` 表征参数 ``[q1, p1, I2, θ2, I3, θ3]``，无量纲。

    Returns:
        ``(6,)`` CM 状态 ``[q1, q2, q3, p1, p2, p3]``，无量纲。
    """
    X = np.asarray(X_param, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_param 必须是 (6,)，得到 {X.shape}")
    q1, p1 = X[0], X[1]
    I2, theta2 = X[2], X[3]
    I3, theta3 = X[4], X[5]
    r2 = np.sqrt(2.0 * I2) if I2 > 0.0 else 0.0
    r3 = np.sqrt(2.0 * I3) if I3 > 0.0 else 0.0
    q2 = r2 * np.cos(theta2)
    p2 = r2 * np.sin(theta2)
    q3 = r3 * np.cos(theta3)
    p3 = r3 * np.sin(theta3)
    return np.array([q1, q2, q3, p1, p2, p3], dtype=float)


__all__ = ["cm_to_param", "param_to_cm"]
