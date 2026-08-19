"""转移轨道统一配置类型定义。

把原先分散在 ``config.py`` （优化阶段）与 ``search_config.py`` （搜索阶段）的
两个 dataclass 合并为单一 :class:`TransferConfig`，用 ``search_*`` / ``nlp_*``
前缀区分子域字段，消除"两个 dataclass 描述同一件事"的碎片化。

同时保留 :class:`TransferOptimizationResult` 于本模块，避免 ``transfer.py`` 与
``transfer_optimization.py`` 之间的循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.enums import TransferType
from ..results import ResultStatus

DU = 3.84405000e5


@dataclass
class TransferConfig:
    """转移轨道统一配置。

    同时承载网格搜索阶段（``search_*`` 前缀，供
    :class:`~e2m2e.algorithm.transfer.transfer_search.TransferSearch` 使用）与 NLP 优化阶段
    （``nlp_*`` 前缀，供 :class:`~e2m2e.algorithm.transfer.transfer.Transfer` 与
    :class:`~e2m2e.algorithm.transfer.transfer_optimization.DROTRONLPOptimizer` 使用）的参数。

    搜索阶段字段默认为 ``None``，运行 ``search()`` 前须显式赋值（或通过散装 kwargs /
    ``configure_search()`` 设置）；优化阶段字段提供工程默认值。

    Search-stage fields (default ``None``):

        search_alpha_min / search_alpha_max: α 网格上下界
        search_n_alpha: α 方向网格点数
        search_n_departure: 出发点采样数量
        search_max_transfer_time: 最大转移时间（CR3BP 无量纲时间）
        search_intersection_threshold: 相交判定距离阈值（无量纲距离）
        search_min_distance_threshold: 候选解最小距离阈值（无量纲距离）
        search_collision_earth_radius: 地球碰撞检测半径（无量纲距离）
        search_collision_moon_radius: 月球碰撞检测半径（无量纲距离）
        search_integration_dt: 积分时间步长（无量纲时间）

    NLP-stage fields (with engineering defaults):

        nlp_alpha_min / nlp_alpha_max: 优化阶段 α 范围
        nlp_earth_radius / nlp_moon_radius: 优化阶段碰撞检测半径（无量纲距离）
        nlp_use_relaxed_velocity: 是否使用松弛速度约束
        nlp_velocity_angle_tol: 松弛速度约束角度容差（弧度）
        nlp_t_ins_range: RO 上的插入时间范围；None 时由调用方推导
        nlp_transfer_time_range: 优化阶段转移时间范围
        nlp_use_copt: 是否优先使用 COPT 优化器
        nlp_fallback_to_scipy: COPT 失败时是否回退到 SciPy
        nlp_verbose: 优化器是否打印迭代信息
    """

    # --- 搜索阶段参数（search_*）---
    search_alpha_min: float | None = None
    search_alpha_max: float | None = None
    search_n_alpha: int | None = None
    search_n_departure: int | None = None
    search_max_transfer_time: float | None = None
    search_intersection_threshold: float | None = None
    search_min_distance_threshold: float | None = None
    search_collision_earth_radius: float | None = None
    search_collision_moon_radius: float | None = None
    search_integration_dt: float | None = None

    # --- 优化阶段参数（nlp_*）---
    nlp_alpha_min: float = 0.5
    nlp_alpha_max: float = 2.5
    nlp_earth_radius: float = 200.0 / DU
    nlp_moon_radius: float = 100.0 / DU
    nlp_use_relaxed_velocity: bool = True
    nlp_velocity_angle_tol: float = 0.05
    nlp_t_ins_range: tuple[float, float] | None = None
    nlp_transfer_time_range: tuple[float, float] | None = None
    nlp_use_copt: bool = False
    nlp_fallback_to_scipy: bool = False  # ADR 0020 决策 4：COPT 缺失/失败默认报错，显式 True 才回退
    nlp_verbose: bool = False

    @property
    def nlp_alpha_range(self) -> tuple[float, float]:
        """由 nlp_alpha_min / nlp_alpha_max 导出的优化阶段 α 范围。"""
        return (self.nlp_alpha_min, self.nlp_alpha_max)


@dataclass
class TransferOptimizationResult:
    """转移优化结果

    Attributes:
        status: 优化最终状态。
        cause: 导致该状态的原因码。
        message: 人类可读诊断。
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

    status: ConvergenceState = ConvergenceState.FAILED
    cause: FailureCause = FailureCause.UNKNOWN
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

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass(frozen=True)
class TransferArc:
    """一段无动力飞行弧（脉冲之间的轨迹段）。

    Attributes:
        states: 弧上状态序列 ``[x, y, z, vx, vy, vz]``，形状 ``(n, 6)``，物理单位 (km, km/s)
        times: 对应时刻序列，形状 ``(n,)``，s
        delta_v: 进入该弧所需的脉冲大小，km/s（首段弧即出发脉冲）
    """

    states: np.ndarray
    times: np.ndarray
    delta_v: float = 0.0


@dataclass(frozen=True)
class TransferSolution:
    """三体打靶 / 低能转移的轻量结果类型。

    与 :class:`TransferOptimizationResult` 风格对齐，但按 frozen dataclass 保持最小：
    多段弧各自携带进入脉冲，到达脉冲单列，总脉冲为各脉冲之和。

    Attributes:
        arcs: 转移弧序列（物理单位 km, km/s, s）
        arrival_delta_v: 到达脉冲（末段弧之后的交会/入轨脉冲），km/s
        total_delta_v: 全部脉冲之和，km/s
        transfer_time: 总飞行时间，s
        status: 算法最终状态
        cause: 算法最终原因码
        n_iter: Newton 迭代次数（流水线取末次修正的迭代数）
        message: 附加说明（未收敛原因、流水线备注等）
    """

    arcs: tuple[TransferArc, ...]
    arrival_delta_v: float
    total_delta_v: float
    transfer_time: float
    status: ConvergenceState
    cause: FailureCause
    message: str
    n_iter: int

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)
