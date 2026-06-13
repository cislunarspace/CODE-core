"""转移优化公共类型定义。

把 TransferConfig 与 TransferOptimizationResult 放在独立模块，
避免 transfer.py 与 transfer_optimization.py 之间的循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mbse.data.enums import TransferType


DU = 3.84405000e5


@dataclass
class TransferConfig:
    """转移优化配置

    Attributes:
        alpha_min: 切向速度比下界
        alpha_max: 切向速度比上界
        alpha_range: (alpha_min, alpha_max) 元组，与 alpha_min/alpha_max 保持一致
        earth_radius: 地球碰撞检测半径（无量纲）
        moon_radius: 月球碰撞检测半径（无量纲）
        use_relaxed_velocity: 是否使用松弛速度约束
        velocity_angle_tol: 松弛速度约束角度容差（弧度）
        t_ins_range: RO 上插入时间范围；None 时由调用方推导
        use_copt: 是否优先使用 COPT 优化器
        fallback_to_scipy: COPT 失败时是否回退到 SciPy
        verbose: 优化器是否打印迭代信息
    """

    alpha_min: float = 0.5
    alpha_max: float = 2.5
    earth_radius: float = 200.0 / DU
    moon_radius: float = 100.0 / DU
    use_relaxed_velocity: bool = True
    velocity_angle_tol: float = 0.05
    t_ins_range: tuple[float, float] | None = None
    use_copt: bool = False
    fallback_to_scipy: bool = True
    verbose: bool = False

    @property
    def alpha_range(self) -> tuple[float, float]:
        """由 alpha_min / alpha_max 导出的范围。"""
        return (self.alpha_min, self.alpha_max)


@dataclass
class TransferOptimizationResult:
    """转移优化结果

    Attributes:
        success: 优化是否成功
        message: 求解器消息
        departure_state: 出发点状态 [x, y, z, vx, vy, vz]
        departure_alpha: 出发点切向速度比
        departure_beta: 出发点法向速度比
        insertion_state: RO 上的插入点状态 [x, y, z, vx, vy, vz]
        final_state: 插入后最终状态 [x, y, z, vx, vy, vz]
        delta_v1: 出发脉冲大小
        delta_v2: 插入脉冲大小
        total_delta_v: 总脉冲（delta_v1 + delta_v2）
        transfer_time: 转移时长
        t_ins: RO 上的插入时间
        transfer_trajectory: 完整转移轨迹 [n_steps, 6]
        transfer_trajectory_times: 轨迹时间序列 [n_steps]
        constraints_violation: 最大约束违反量
        transfer_type: 转移类型
    """

    success: bool = False
    message: str = ""
    departure_state: np.ndarray | None = None
    departure_alpha: float = 0.0
    departure_beta: float = 0.0
    insertion_state: np.ndarray | None = None
    final_state: np.ndarray | None = None
    delta_v1: float = 0.0
    delta_v2: float = 0.0
    total_delta_v: float = 0.0
    transfer_time: float = 0.0
    t_ins: float = 0.0
    transfer_trajectory: np.ndarray | None = None
    transfer_trajectory_times: np.ndarray | None = None
    constraints_violation: float = 0.0
    transfer_type: TransferType = TransferType.DIRECT
