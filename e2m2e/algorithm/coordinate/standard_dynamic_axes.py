"""常用动态坐标轴实现。

- VNBAxes: 速度-法向-副法向（VNB）坐标轴
- LVLHAxes: 本地垂直本地水平（LVLH）坐标轴
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.templates.enums import LVLHReference
from .dynamic_axes import DynamicAxes

# 退化态阈值（ADR 0020 决策 5：机器精度正则化保留）。三个阈值均为机器精度
# 量级，仅防除零 NaN 与产出无意义方向；低于阈值抛 ValueError，实测量与阈值
# 写入异常信息（阈值可观测）。状态单位由调用方定（星历传播为 km/km·s⁻¹，
# CR3BP 为无量纲），故阈值取绝对机器量级而非物理量纲值。
VELOCITY_NORM_MIN = 1e-12
ANGULAR_MOMENTUM_NORM_MIN = 1e-12
POSITION_NORM_MIN = 1e-12


def reference_body_position(
    reference: LVLHReference = LVLHReference.BARYCENTER,
    mu: float | None = None,
) -> npt.NDArray[np.floating]:
    """返回参考体在 CR3BP 会合系中的位置。

    会合系里主天体固定在 (-μ, 0, 0)，次天体固定在 (1-μ, 0, 0)，双质心
    在原点。参考体位置只用于构造 LVLH 轴向，不改变相对状态的参考原点。

    Args:
        reference: LVLH 参考体，默认双质心。
        mu: CR3BP 质量参数，参考体为主天体或次天体时必填。

    Returns:
        参考体位置，形状 ``(3,)``。

    Raises:
        ValueError: 参考体为主天体或次天体但 ``mu`` 缺失或不在 (0, 1) 内，
            或参考体取值无效。
    """
    if reference is LVLHReference.BARYCENTER:
        return np.zeros(3)
    if mu is None:
        raise ValueError("参考体为 PRIMARY 或 SECONDARY 时必须提供 mu")
    mu = float(mu)
    if not 0.0 < mu < 1.0:
        raise ValueError(f"mu 必须在 (0, 1) 内，实际为 {mu!r}")
    if reference is LVLHReference.PRIMARY:
        return np.array([-mu, 0.0, 0.0])
    if reference is LVLHReference.SECONDARY:
        return np.array([1.0 - mu, 0.0, 0.0])
    raise ValueError(f"未知的参考体：{reference!r}")


def lvlh_basis(
    r_rel: npt.ArrayLike,
    v: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """由参考体相对位置与会合速度构造 LVLH 旋转矩阵。

    旋转矩阵的列为 ``[r̂, V̂, ĥ]``，``r̂`` 为参考体到目标的径向单位向量，
    ``ĥ = r_rel × v`` 归一化，``V̂ = ĥ × r̂``。``v`` 必须是会合系速度。

    参考体为月球时会合角动量符号在 2:1 DRO 上全程恒定。参考体为双质心
    时同一轨道的会合角动量在 t/T 约 0.21 与 0.79 过零并翻转基矢，惯性
    口径的月心角动量在 t/T 约 0.28 与 0.72 过零。公里级编队几何应选
    月球参考体。

    Args:
        r_rel: 目标相对参考体的位置，形状 ``(3,)``。
        v: 会合系速度，形状 ``(3,)``。

    Returns:
        旋转矩阵，形状 ``(3, 3)``，列为径向、切向与法向单位向量。

    Raises:
        ValueError: 相对位置或角动量低于机器精度阈值（ADR 0020 决策 5），
            异常信息含实测范数与阈值。
    """
    r_rel = np.asarray(r_rel, dtype=float)
    v = np.asarray(v, dtype=float)
    h = np.cross(r_rel, v)
    r_norm = float(np.linalg.norm(r_rel))
    h_norm = float(np.linalg.norm(h))
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
    r_hat = r_rel / r_norm
    h_hat = h / h_norm
    v_hat = np.cross(h_hat, r_hat)
    return np.column_stack([r_hat, v_hat, h_hat])


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

    R = 径向（参考体相对位置归一化），H = 角动量方向（r_rel × v 归一化），
    V = H × R（归一化）。旋转矩阵的列为 ``[r_hat, v_hat, h_hat]``。

    参考体默认双质心，此时径向从会合系原点量起，行为与旧版一致。可选
    主天体（地月系为地球）或次天体（地月系为月球），此时 ``mu`` 必填。
    参考体只改变轴向，不改变传入状态自身的参考原点。

    法向用会合速度。参考体为月球时会合角动量符号在 2:1 DRO 上全程恒定，
    参考体为双质心时同一轨道的会合角动量在 t/T 约 0.21 与 0.79 过零并
    翻转基矢，惯性口径的月心角动量在 t/T 约 0.28 与 0.72 过零。

    Args:
        reference: LVLH 参考体，默认双质心。
        mu: CR3BP 质量参数，参考体为主天体或次天体时必填。

    Raises:
        ValueError: 参考体非双质心且 ``mu`` 缺失或无效（构造时），或
            参考体相对位置与角动量退化（update 时，ADR 0007 补白与
            ADR 0020 决策 5）。
    """

    def __init__(
        self,
        reference: LVLHReference = LVLHReference.BARYCENTER,
        mu: float | None = None,
    ) -> None:
        super().__init__()
        self.reference = reference
        self.mu = mu
        self._reference_position = reference_body_position(reference, mu)

    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        r = state[:3]
        v = state[3:]
        self._rotation = lvlh_basis(r - self._reference_position, v)
        self._updated = True

    def _compute_rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return self._rotation
