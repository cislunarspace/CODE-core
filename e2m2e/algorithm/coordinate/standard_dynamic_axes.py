"""常用动态坐标轴实现。

- VNBAxes: 速度-法向-副法向（VNB）坐标轴
- LVLHAxes: 本地垂直本地水平（LVLH）坐标轴
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .dynamic_axes import DynamicAxes

# 退化态阈值（ADR 0020 决策 5：机器精度正则化保留）。三个阈值均为机器精度
# 量级，仅防除零 NaN 与产出无意义方向；低于阈值抛 ValueError，实测量与阈值
# 写入异常信息（阈值可观测）。状态单位由调用方定（星历传播为 km/km·s⁻¹，
# CR3BP 为无量纲），故阈值取绝对机器量级而非物理量纲值。
VELOCITY_NORM_MIN = 1e-12
ANGULAR_MOMENTUM_NORM_MIN = 1e-12
POSITION_NORM_MIN = 1e-12


class VNBAxes(DynamicAxes):
    """速度-法向-副法向（VNB）坐标轴。

    V = 速度方向（归一化），N = 角动量方向（r × v，归一化），
    B = V × N（归一化）。旋转矩阵 R = [v_hat, n_hat, b_hat]^T，
    即 R 的第 0 列为 v_hat，第 1 列为 n_hat，第 2 列为 b_hat。

    Raises:
        ValueError: 零速度（``|v| < VELOCITY_NORM_MIN``）或零角动量
            （``|r×v| < ANGULAR_MOMENTUM_NORM_MIN``，即 r ∥ v 或 r=0）时
            轴向奇异（ADR 0007 补白 / ADR 0020 决策 5：退化态显式失败，
            不静默）；异常信息含实测范数与阈值。
    """

    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        r = state[:3]
        v = state[3:]
        h = np.cross(r, v)
        v_norm = np.linalg.norm(v)
        h_norm = np.linalg.norm(h)
        if v_norm < VELOCITY_NORM_MIN:
            raise ValueError(
                f"VNB 轴向退化：速度为零（|v|={v_norm:.3e}，阈值 "
                f"{VELOCITY_NORM_MIN:.1e}），无法定义速度方向"
            )
        if h_norm < ANGULAR_MOMENTUM_NORM_MIN:
            raise ValueError(
                f"VNB 轴向退化：角动量为零（|r×v|={h_norm:.3e}，阈值 "
                f"{ANGULAR_MOMENTUM_NORM_MIN:.1e}，r ∥ v 或 r = 0），无法定义法向"
            )
        v_hat = v / v_norm
        h_hat = h / h_norm
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
        ValueError: 零位置（``|r| < POSITION_NORM_MIN``）或零角动量
            （``|r×v| < ANGULAR_MOMENTUM_NORM_MIN``，即 r ∥ v）时
            轴向奇异（ADR 0007 补白 / ADR 0020 决策 5：退化态显式失败，
            不静默）；异常信息含实测范数与阈值。
    """

    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        r = state[:3]
        v = state[3:]
        h = np.cross(r, v)
        r_norm = np.linalg.norm(r)
        h_norm = np.linalg.norm(h)
        if r_norm < POSITION_NORM_MIN:
            raise ValueError(
                f"LVLH 轴向退化：位置为零（|r|={r_norm:.3e}，阈值 "
                f"{POSITION_NORM_MIN:.1e}），无法定义径向"
            )
        if h_norm < ANGULAR_MOMENTUM_NORM_MIN:
            raise ValueError(
                f"LVLH 轴向退化：角动量为零（|r×v|={h_norm:.3e}，阈值 "
                f"{ANGULAR_MOMENTUM_NORM_MIN:.1e}，r ∥ v），无法定义法向"
            )
        r_hat = r / r_norm
        h_hat = h / h_norm
        v_hat = np.cross(h_hat, r_hat)
        self._rotation = np.column_stack([r_hat, v_hat, h_hat])
        self._updated = True

    def _compute_rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return self._rotation
