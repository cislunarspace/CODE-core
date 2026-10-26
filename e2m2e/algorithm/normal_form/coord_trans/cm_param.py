"""CM ↔ param：中心流形坐标 ↔ 表征参数（复→极坐标，作用量-角变量）。

迁移自 qiao ``Subfunction/coord_trans/param2ActionAngle.py`` （仅
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

- qiao ``param2ActionAngle`` 用 ``atan2(q2, p2)`` （参数交换），本仓库
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


def canonical_to_action_angle(X_cm: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """将完整正则坐标转为 qiao 约定的六个作用量角度根数。

    输入顺序为 ``(q1, q2, q3, p1, p2, p3)``，输出顺序为
    ``(I1, theta1, I2, theta2, I3, theta3)``。双曲定义为
    ``I1=q1*p1``、``theta1=0.5*ln(|q1|/|p1|)``，中心角定义为
    ``atan2(q, p)``，与 qiao ``param2_action_angle`` 一致。

    双曲坐标任一分量为零时 ``theta1`` 没有有限实值，此处显式报错。
    中心作用量为零的角度未定义由 ``numpy.arctan2`` 给主值零，使用者
    在末端约束中应改用非奇异正则坐标，不应将该主值解释为物理相位。
    """
    X = np.asarray(X_cm, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_cm 必须是 (6,)，得到 {X.shape}")
    q1, q2, q3, p1, p2, p3 = X
    if p1 <= 0.0:
        raise ValueError("该分支逆变换要求 p1>0；p1<0 的状态缺少分支符号，不能双射还原")
    if q1 == 0.0 or p1 == 0.0:
        raise ValueError("双曲坐标 q1 或 p1 为零时 theta1 没有有限实值")
    return np.array(
        [
            q1 * p1,
            0.5 * np.log(abs(q1) / abs(p1)),
            0.5 * (q2 * q2 + p2 * p2),
            np.arctan2(q2, p2) % (2.0 * np.pi),
            0.5 * (q3 * q3 + p3 * p3),
            np.arctan2(q3, p3) % (2.0 * np.pi),
        ]
    )


def action_angle_to_canonical(X_elements: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """将作用量角度根数还原到 ``p1>0`` 的正则坐标分支。

    双曲作用量角度丢失 ``(q1,p1)`` 的共同符号。此逆变换固定选取
    ``p1>0`` 的分支，不声称在其他符号分支上双射。中心作用量必须非负。
    """
    X = np.asarray(X_elements, dtype=float).ravel()
    if X.shape != (6,):
        raise ValueError(f"X_elements 必须是 (6,)，得到 {X.shape}")
    I1, theta1, I2, theta2, I3, theta3 = X
    if I1 == 0.0:
        raise ValueError("I1 为零时双曲作用量角度表示不唯一，不能唯一还原")
    if I2 < 0.0 or I3 < 0.0:
        raise ValueError("中心作用量 I2 与 I3 必须非负")
    # 由 theta1=ln(sqrt(|q1|)/sqrt(|p1|)) 和 I1=q1*p1 反解。
    saddle_sign = 1.0 if I1 > 0.0 else -1.0
    q1 = saddle_sign * np.sqrt(abs(I1)) * np.exp(theta1)
    p1 = np.sqrt(abs(I1)) * np.exp(-theta1)
    q2 = np.sqrt(2.0 * I2) * np.sin(theta2)
    p2 = np.sqrt(2.0 * I2) * np.cos(theta2)
    q3 = np.sqrt(2.0 * I3) * np.sin(theta3)
    p3 = np.sqrt(2.0 * I3) * np.cos(theta3)
    return np.array([q1, q2, q3, p1, p2, p3])


def shortest_angle_error(actual: float, target: float) -> float:
    """返回 ``actual-target`` 在 ``[-pi, pi)`` 内的最短角差。"""
    return (actual - target + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "action_angle_to_canonical",
    "canonical_to_action_angle",
    "cm_to_param",
    "param_to_cm",
    "shortest_angle_error",
]
