"""核心数据模型

基于 Pydantic 的统一数据结构，替代现有分散的 dict/plain class/dataclass 混用模式。

所有模型支持 JSON 序列化/反序列化和运行时类型验证。
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class NumpyArray(np.ndarray):
    """Pydantic 兼容的 numpy 数组类型标记

    作为字段类型注解使用，实际的形状校验由各字段上的
    @field_validator 完成，本类本身不包含 validator。
    """

    pass


class _NumpyModel(BaseModel):
    """支持 numpy 数组的 Pydantic 基类"""

    # numpy.ndarray 不是标准 Pydantic 类型，必须允许任意类型才能通过校验
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OrbitProperties(_NumpyModel):
    """轨道属性数据模型

    从 Orbit 类中提取的计算属性组。

    Attributes:
        period: 轨道周期（无量纲时间）
        amplitudes: 各方向振幅 {'x': float, 'y': float, 'z': float}
        extrema: 极值点 {'x_max': float, 'x_min': float, ...}
        mean_state: 平均状态向量，形状 (6,)
        center: 轨道中心点（位置分量均值），形状 (3,)
        is_periodic: 是否为周期轨道
        periodicity_error: 周期性误差（首末状态欧氏距离）
    """

    period: float | None = None
    amplitudes: dict[str, float] | None = None
    extrema: dict[str, float] | None = None
    mean_state: np.ndarray | None = None
    center: np.ndarray | None = None
    is_periodic: bool = False
    periodicity_error: float | None = None

    @field_validator("mean_state")
    @classmethod
    def validate_mean_state(cls, value: np.ndarray | None) -> np.ndarray | None:
        """确保平均状态向量保持六维状态契约。"""
        if value is not None and value.shape != (6,):
            raise ValueError("mean_state 必须是形状 (6,) 的数组")
        return value

    @field_validator("center")
    @classmethod
    def validate_center(cls, value: np.ndarray | None) -> np.ndarray | None:
        """确保轨道中心保持三维位置契约。"""
        if value is not None and value.shape != (3,):
            raise ValueError("center 必须是形状 (3,) 的数组")
        return value
