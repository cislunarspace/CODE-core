"""转移设计契约：结果 dataclass 与 top-N / state_frame 常量。

``ManeuverEvent``/``TransferCandidate``/``TransferDesignResult`` 与五个
``*TransferDetails`` 是编排器（``orchestrator.py``）与传输层共用的结果
契约；``DEFAULT_TOP_N`` 与 ``STATE_FRAME_*`` 标签按 ADR 0040 增补定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...status import ConvergenceState, FailureCause, ResultStatus
from ..results import StageRecord
from .bplane import AsymptoteParams, BPlaneParams
from .lga import LgaSearchParams
from .lowthrust_shooting import EngineConfig, LowThrustSegment
from .wsb import WsbSearchParams

# top-N 可行解契约（#583，ADR 0040 增补）：opt-in 参数不设隐式开启——
# 契约的推荐候选数为消费方（tod 画布方案对比，tod #430）初值提案 5；
# 调用方显式传 top_n 才启用，默认行为与单解契约逐字段一致。
DEFAULT_TOP_N: int = 5

# ADR 0040 增补：state_frame 数据系标签的转移管线取值。gcrs_km（#584）
# 标注并行惯性段 trajectory_gcrs_km（字段自带数据系，顶层 state_frame
# 仍指 trajectory 主段）；synodic_barycentric_nd 待其他响应
# （propagation/design）批次接入。
STATE_FRAME_SYNODIC_BARYCENTRIC_KM = "synodic_barycentric_km"
STATE_FRAME_FORCE_MODEL_STATE = "force_model_state"
STATE_FRAME_GCRS_KM = "gcrs_km"

_STATE_FRAME_BY_TRANSFER_TYPE: dict[str, str] = {
    "HMN": STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    "LGA": STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    "WSB": STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    "PCN": STATE_FRAME_SYNODIC_BARYCENTRIC_KM,
    "low_thrust": STATE_FRAME_FORCE_MODEL_STATE,
}


@dataclass(frozen=True)
class ManeuverEvent:
    """单次机动事件（#575 契约，与 transfer catalog record 共用 schema）。

    Attributes:
        kind: 事件类别（"departure" / "arrival" / "perilune" / …，开放枚举）。
        t_sec: TLI 起算秒（t=0 为出发脉冲），与 trajectory_times 同基准。
        dv_km_s: 该次机动脉冲大小；非脉冲事件（perilune 旗标）为 0.0。
        note: 可选人类可读注记。
    """

    kind: str
    t_sec: float
    dv_km_s: float
    note: str | None = None


@dataclass(frozen=True)
class TransferCandidate:
    """top-N 可行解契约的单个候选（#583，ADR 0040 增补）。

    Attributes:
        delta_v_km_s: 该候选的总 Δv (km/s)。选中解与顶层
            ``TransferDesignResult.delta_v`` 同口径；未精化候选为搜索
            网格估计（精化前口径）。
        tli_epoch: 出发历元（UTC 字符串或 JD_TDB 浮点，原样透传）；
            low_thrust 的出发态直传路径无历元语义，为 None。
        tof_sec: 飞行时间 (秒)。
        trajectory: 轨迹快照 (n, 6)，数据系由 ``state_frame`` 标注；
            快照组装（传播）失败时为 None，不影响其余候选。
        trajectory_times: 快照时刻 (n,) 秒，TLI 起算，与 trajectory
            逐行对齐；trajectory 为 None 或后端不提供时刻
            （low_thrust）时为 None。
        state_frame: 快照的数据系标签（ADR 0040 ``state_frame`` 词汇）。
        selected: 是否为选中解（与默认路径最优解一致的那一个）；
            恰有一个候选为 True。
        refined: Δv 口径标记——True 为打靶精化后数值（与顶层同口径），
            False 为搜索网格估计（精化前）。LGA/WSB 的未精化候选恒为
            False；HMN/low_thrust 无搜索-精化两级，单候选恒为 True。
    """

    delta_v_km_s: float
    tli_epoch: float | str | None
    tof_sec: float
    trajectory: Any
    trajectory_times: Any
    state_frame: str
    selected: bool
    refined: bool


@dataclass
class TransferDesignResult:
    """转移轨道设计结果。

    Attributes:
        transfer_type: 转移类型（"HMN"/"LGA"/"WSB"/"low_thrust"/"PCN"）。
        delta_v: 总 Δv（km/s）。
        trajectory: 转移轨迹 (n, 6)，地月会合旋转系、质心原点、物理单位
            km / km/s（ADR 0040；HMN 为两体几何的相位对齐显示约定，
            low_thrust 暂为力模型状态系的已知不一致）。
        trajectory_times: 轨迹时刻 (n,) 秒，TLI 起算（t=0 为出发脉冲），
            与 trajectory 逐行对齐。
        trajectory_gcrs_km: 惯性几何段 (n, 6)（#584，ADR 0040 增补）：
            地心原点、不旋转轴（GCRS 约定）物理 km / km/s，与
            trajectory / trajectory_times 逐行对齐，共享时刻数组（时刻
            不双份）。HMN 为两体弧构造系原样；LGA/WSB 为会合系几何旋回
            惯性（速度含 ω×r 牵连项；θ₀=0 约定 TLI 时刻地月连线与惯性
            x 轴重合——理想化方位，无星历语义）；low_thrust 与搜索零
            结果为 None。
        state_frame: trajectory 的数据系标签（ADR 0040 增补）。缺省按
            transfer_type 派生，显式传值可覆盖。
        maneuver_events: 结构化机动事件列表（#575）。HMN 为
            departure/arrival 两条（到达点即近月点，不另发 perilune）；
            LGA/WSB 为 departure/perilune/arrival 三条（perilune 的
            dv_km_s=0，飞越段无脉冲）；low_thrust 连续推进无脉冲语义，
            恒为空；搜索零结果恒为空。
        details: 设计细节（弹道参数汇总）。
        stages: 搜索、精化和打靶等可选阶段的执行记录。
        status: 任务最终状态。
        cause: 导致该状态的原因码。
        message: 人类可读诊断。
        bplane: 达成的月心 B-plane 参数（#635；仅 PCN 路径填充，其余为 None）。
        departure_asymptote: 实际使用的出发双曲渐近线（#635；仅 PCN 填充）。
    """

    transfer_type: str
    delta_v: float
    trajectory: Any
    trajectory_times: Any = None
    trajectory_gcrs_km: Any = None
    state_frame: str = ""
    maneuver_events: tuple[ManeuverEvent, ...] = ()
    candidates: tuple[TransferCandidate, ...] = ()
    details: (
        HmnTransferDetails
        | LgaTransferDetails
        | WsbTransferDetails
        | LowThrustTransferDetails
        | PcnTransferDetails
        | dict[str, Any]
    ) = field(default_factory=dict)
    stages: tuple[StageRecord, ...] = ()
    status: ConvergenceState = ConvergenceState.CONVERGED
    cause: FailureCause = FailureCause.NONE
    message: str = "任务完成"
    bplane: BPlaneParams | None = None
    departure_asymptote: AsymptoteParams | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)
        if not self.state_frame:
            derived = _STATE_FRAME_BY_TRANSFER_TYPE.get(self.transfer_type)
            if derived is None:
                raise ValueError(
                    f"transfer_type {self.transfer_type!r} 无 state_frame 派生规则，须显式指定"
                )
            self.state_frame = derived


@dataclass
class HmnTransferDetails:
    """霍曼转移设计细节。

    Attributes:
        tli_epoch: 出发历元（UTC 字符串或 JD_TDB 浮点数）。
        tof_sec: 飞行时间 (秒)。
        r1_km: 出发轨道半径 (km)。
        r2_km: 目标轨道半径 (km)。
        dv1_km_s: 出发 Δv (km/s)。Deprecated: 改用
            ``TransferDesignResult.maneuver_events`` 的 departure 事件（#575）；
            本字段保留一个版本不变。
        dv2_km_s: 到达 Δv (km/s)。Deprecated: 改用 maneuver_events 的
            arrival 事件（#575）；本字段保留一个版本不变。
        departure_state: ECI 出发状态 (6,)。
        delta_v_theory: 理论 Δv (dv1, dv2)。
    """

    tli_epoch: float | str
    tof_sec: float
    r1_km: float
    r2_km: float
    dv1_km_s: float
    dv2_km_s: float
    departure_state: NDArray[np.float64]
    delta_v_theory: tuple[float, float]


@dataclass
class LgaTransferDetails:
    """LGA 月球引力辅助转移设计细节。

    Deprecated 字段：``dv_departure_km_s`` / ``dv_arrival_km_s`` 改用
    ``TransferDesignResult.maneuver_events`` 的 departure/arrival 事件
    （#575）；本字段保留一个版本不变。
    """

    tli_epoch: float | str
    tof_sec: float
    perilune_alt_km: float
    perilune_vel_km_s: float
    perilune_state: np.ndarray
    dv_departure_km_s: float
    dv_arrival_km_s: float
    jacobi_departure: float
    jacobi_arrival: float
    n_candidates_searched: int
    n_candidates_feasible: int
    status: ConvergenceState
    cause: FailureCause
    message: str
    search_params: LgaSearchParams

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class WsbTransferDetails:
    """WSB 太阳引力辅助转移设计细节。

    Deprecated 字段：``dv_departure_km_s`` / ``dv_arrival_km_s`` 改用
    ``TransferDesignResult.maneuver_events`` 的 departure/arrival 事件
    （#575）；本字段保留一个版本不变。
    """

    tli_epoch: float | str
    tof_sec: float
    perilune_alt_km: float
    perilune_vel_km_s: float
    perilune_state: np.ndarray
    h2_kepler: float
    dv_departure_km_s: float
    dv_arrival_km_s: float
    n_candidates_searched: int
    n_candidates_feasible: int
    status: ConvergenceState
    cause: FailureCause
    message: str
    search_params: WsbSearchParams

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class LowThrustTransferDetails:
    """小推力转移设计细节。

    Attributes:
        engine: 推进配置。
        initial_mass: 初始质量 (kg)。
        final_mass: 末态质量 (kg)。
        fuel_consumed: 燃料消耗 (kg)。
        equivalent_delta_v: 等效 Δv (km/s)，Tsiolkovsky 方程反算。
        n_segments: 求解器段数。
        solver_method: 求解方法 ("shooting" / "collocation")。
        status: 求解最终状态。
        cause: 求解最终原因。
        message: 求解器消息。
        n_iter: 迭代次数。
        terminal_residual_r: 终端位置残差 (km)。
        terminal_residual_v: 终端速度残差 (km/s)。
        time: 采样时间序列 (M,)，SPICE et 秒。
        states_7d: 7D 状态序列 (M, 7) [x,y,z,vx,vy,vz,m]。
        segments: 各段常量控制。
        qlaw_q_history: Q-law Q 值历史（仅 solve_from_qlaw 时非空）。
    """

    engine: EngineConfig
    initial_mass: float
    final_mass: float
    fuel_consumed: float
    equivalent_delta_v: float
    n_segments: int
    solver_method: str  # "shooting" | "collocation"
    status: ConvergenceState
    cause: FailureCause
    message: str
    n_iter: int
    terminal_residual_r: float  # km
    terminal_residual_v: float  # km/s
    time: NDArray[np.float64]
    states_7d: NDArray[np.float64]
    segments: tuple[LowThrustSegment, ...]
    qlaw_q_history: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class PcnTransferDetails:
    """PCN patched-conic 目标参数化转移设计细节（#635）。

    仅标量/字符串字段（catalog ``_details_block`` 逐字段 JSON 化，保持
    JSON 安全；B-plane 基矢量等 ndarray 不在此携带）。

    Attributes:
        tli_epoch: 出发历元（UTC 字符串或 JD_TDB 浮点数）。
        tof_sec: 总飞行时间 (s)，TLI 起算。
        earth_leg_tof_sec: 地球段飞行时间 (s)。
        moon_leg_tof_sec: 月心段（handoff → 近月点）飞行时间 (s)。
        parking_alt_km: 地球停泊轨道高度 (km)。
        dv_tli_km_s: TLI 出发脉冲 (km/s)。
        dv_loi_km_s: 近月点圆化脉冲 (km/s)。
        perilune_alt_km: 近月点高度 (km；近月点半径 − 月平均半径)。交会解为月面以上
            正值；撞月等非交会解为诊断值（可为负）。
        v_inf_moon_km_s: 月心到达剩余速度 v∞ (km/s)。
        c3_departure_km2_s2: 出发 C3 能量 (km²/s²)。
        rha_deg: 出发渐近线赤经 (deg)。
        dha_deg: 出发渐近线赤纬 (deg)。
        bdot_r_km: 达成 B·R (km)。
        bdot_t_km: 达成 B·T (km)。
        b_mag_km: 瞄准距离 |B| (km)。
        mode: ``"arrival"`` 或 ``"departure"``。
        n_grid_evals: 网格初猜评估次数。
        n_newton_iter: Newton 迭代次数。
        status: 求解最终状态。
        cause: 求解最终原因。
        message: 求解器消息。
    """

    tli_epoch: float | str
    tof_sec: float
    earth_leg_tof_sec: float
    moon_leg_tof_sec: float
    parking_alt_km: float
    dv_tli_km_s: float
    dv_loi_km_s: float
    perilune_alt_km: float
    v_inf_moon_km_s: float
    c3_departure_km2_s2: float
    rha_deg: float
    dha_deg: float
    bdot_r_km: float
    bdot_t_km: float
    b_mag_km: float
    mode: str
    n_grid_evals: int
    n_newton_iter: int
    status: ConvergenceState
    cause: FailureCause
    message: str

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)
