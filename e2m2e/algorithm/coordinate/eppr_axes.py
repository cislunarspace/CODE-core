"""EPPR 旋转坐标轴。

Axes 契约不含尺度因子：基即瞬时地月会合基（与 ``SynodicAxes`` 同定义——
x̂ 地球→月球、ẑ 瞬时轨道角动量、ŷ 右手补全），速率由 ``EPPRFrameModel`` 解析
供给（避免数值微分污染 Ẇ 推导）。脉动长度尺度由转换器 / 系统侧显式供给（仿
``SynodicJ2000System`` 对 synodic 尺度的处理）。
"""

from __future__ import annotations

import numpy.typing as npt

from .axes import Axes
from .eppr_frame import EPPRFrameModel


class EPPRAxes(Axes):
    """星历脉动旋转坐标轴（帧量由 ``EPPRFrameModel`` 供给）。"""

    def __init__(self, frame_model: EPPRFrameModel) -> None:
        self._frame_model = frame_model

    def rotation_matrix(self, et: float) -> npt.NDArray:
        """返回从 EPPR 轴到 ICRF/J2000 的旋转矩阵（``r_icrf = R @ r_eppr``）。"""
        return self._frame_model.frame_state(et).rotation

    def rotation_and_rate(self, et: float) -> tuple[npt.NDArray, npt.NDArray]:
        """返回 ``(R, Ṙ)``；速率取帧模型的解析值，非基类中心差分。"""
        frame_state = self._frame_model.frame_state(et)
        return frame_state.rotation, frame_state.rotation_rate
