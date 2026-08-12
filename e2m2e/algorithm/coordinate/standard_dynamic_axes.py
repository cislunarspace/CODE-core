"""常用动态坐标轴实现。

- VNBAxes: 速度-法向-副法向（VNB）坐标轴
- LVLHAxes: 本地垂直本地水平（LVLH）坐标轴
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .dynamic_axes import DynamicAxes


class VNBAxes(DynamicAxes):
    """速度-法向-副法向（VNB）坐标轴。

    V = 速度方向（归一化），N = 角动量方向（r × v，归一化），
    B = V × N（归一化）。旋转矩阵 R = [v_hat, n_hat, b_hat]^T，
    即 R 的第 0 列为 v_hat，第 1 列为 n_hat，第 2 列为 b_hat。

    Raises:
        ValueError: 零速度（|v|=0）或零角动量（|r×v|=0，即 r ∥ v 或 r=0）
            时轴向奇异（ADR 0007 补白：退化态显式失败，不静默）。
    """

    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        r = state[:3]
        v = state[3:]
        h = np.cross(r, v)
        if np.linalg.norm(v) < 1e-12:
            raise ValueError("VNB 轴向退化：速度为零，无法定义速度方向")
        if np.linalg.norm(h) < 1e-12:
            raise ValueError("VNB 轴向退化：角动量为零（r ∥ v 或 r = 0），无法定义法向")
        v_hat = v / np.linalg.norm(v)
        h_hat = h / np.linalg.norm(h)
        b_hat = np.cross(v_hat, h_hat)
        self._rotation = np.column_stack([v_hat, h_hat, b_hat])
        self._updated = True

    def _compute_rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return self._rotation


class LVLHAxes(DynamicAxes):
    """本地垂直本地水平（LVLH）坐标轴。

    R = 径向（r 归一化），H = 角动量方向（r × v，归一化），
    V = H × R（归一化）。旋转矩阵 R = [r_hat, v_hat, h_hat]^T，
    即 R 的第 0 列为 r_hat，第 1 列为 v_hat，第 2 列为 h_hat。

    Raises:
        ValueError: 零位置（|r|=0）或零角动量（|r×v|=0，即 r ∥ v）时
            轴向奇异（ADR 0007 补白：退化态显式失败，不静默）。
    """

    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        r = state[:3]
        v = state[3:]
        h = np.cross(r, v)
        if np.linalg.norm(r) < 1e-12:
            raise ValueError("LVLH 轴向退化：位置为零，无法定义径向")
        if np.linalg.norm(h) < 1e-12:
            raise ValueError("LVLH 轴向退化：角动量为零（r ∥ v），无法定义法向")
        r_hat = r / np.linalg.norm(r)
        h_hat = h / np.linalg.norm(h)
        v_hat = np.cross(h_hat, r_hat)
        self._rotation = np.column_stack([r_hat, v_hat, h_hat])
        self._updated = True

    def _compute_rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return self._rotation
