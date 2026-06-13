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


class PropagationResult(_NumpyModel):
    """传播结果数据模型

    替代 Dynamics.propagate() 返回的 Dict[str, Any]。

    关键契约：
    - states 形状始终为 (n_points, 6)
    - stm 形状为 (n_points, 6, 6)（当 with_stm=True 时）
    - time 数组单调递增

    Attributes:
        time: 时间序列，形状 (n,)
        states: 状态序列，形状 (n, 6)
        stm: 状态转移矩阵序列，形状 (n, 6, 6)
        jacobi: Jacobi 常数序列，形状 (n,)
        jacobi_error: Jacobi 常数最大漂移
    """

    time: np.ndarray
    states: np.ndarray
    stm: np.ndarray | None = None
    jacobi: list[float] | None = None
    jacobi_error: float = 0.0

    @field_validator("states")
    @classmethod
    def validate_states_shape(cls, v: np.ndarray) -> np.ndarray:
        """校验状态数组形状为 (n, 6)，保证六维状态向量完整性

        Args:
            v: 待校验的状态数组

        Returns:
            通过校验的状态数组

        Raises:
            ValueError: 形状不符合 (n, 6) 时抛出
        """
        if v.ndim != 2 or v.shape[1] != 6:
            raise ValueError(f"states 形状必须为 (n, 6)，实际为 {v.shape}")
        return v

    @field_validator("stm")
    @classmethod
    def validate_stm_shape(cls, v: np.ndarray | None) -> np.ndarray | None:
        """校验状态转移矩阵形状为 (n, 6, 6)

        Args:
            v: 待校验的 STM 数组，可为 None

        Returns:
            通过校验的 STM 数组或 None

        Raises:
            ValueError: 形状不符合 (n, 6, 6) 时抛出
        """
        if v is not None and (v.ndim != 3 or v.shape[1] != 6 or v.shape[2] != 6):
            raise ValueError(f"stm 形状必须为 (n, 6, 6)，实际为 {v.shape}")
        return v


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


class OrbitStability(_NumpyModel):
    """轨道稳定性数据模型

    从 Orbit 类中提取的稳定性分析结果组。

    Attributes:
        monodromy_matrix: 单值矩阵 (6x6)
        eigenvalues: 单值矩阵特征值（复数数组）
        stability: 稳定性标签
        lyapunov_exponents: Lyapunov 指数数组
    """

    monodromy_matrix: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    stability: str | None = None
    lyapunov_exponents: np.ndarray | None = None

    @field_validator("monodromy_matrix")
    @classmethod
    def validate_monodromy(cls, v: np.ndarray | None) -> np.ndarray | None:
        """校验单值矩阵形状为 (6, 6)

        Args:
            v: 待校验的单值矩阵，可为 None

        Returns:
            通过校验的矩阵或 None

        Raises:
            ValueError: 形状不符合 (6, 6) 时抛出
        """
        if v is not None and v.shape != (6, 6):
            raise ValueError(f"monodromy_matrix 形状必须为 (6, 6)，实际为 {v.shape}")
        return v


class JacobiResult(_NumpyModel):
    """Jacobi 常数计算结果

    Attributes:
        value: Jacobi 常数标量值
        drift: 整条轨迹的 Jacobi 常数最大漂移
    """

    value: float
    drift: float = 0.0


class SystemConfig(BaseModel):
    """系统配置数据模型

    替代 CR3BP_System 中散落的配置参数。

    Attributes:
        primary_body: 主天体名称
        secondary_body: 次天体名称
        mu: 质量参数
        semi_major_axis: 半长轴（km）
        orbital_period: 轨道周期（s）
    """

    primary_body: str = "Earth"
    secondary_body: str = "Moon"
    mu: float = 0.01215
    semi_major_axis: float | None = None
    orbital_period: float | None = None
