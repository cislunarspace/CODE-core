"""低推力转移预设计（low_thrust_preliminary）请求/响应模型（#725，ADR 0054；
ephemeris 星历档上下文字段见 #727）。"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .shared import ResultResponse, _ApiModel

__all__ = [
    "LowThrustPropulsionSpec",
    "LowThrustNodeSpec",
    "LowThrustPreliminaryRequest",
    "LowThrustLegInfo",
    "LowThrustFlybyInfo",
    "LowThrustPreliminaryResponse",
]


class LowThrustPropulsionSpec(_ApiModel):
    """低推力推进配置（常推力 / SEP 二选一，透传 ``SimsFlanaganPropulsion``）。"""

    isp_s: float = Field(gt=0.0, description="比冲（s）")
    t_max_n: float | None = Field(
        default=None,
        ge=0.0,
        description="常推力模式最大推力（N）；与 p0_w 恰一指定",
    )
    p0_w: float | None = Field(
        default=None,
        gt=0.0,
        description="SEP 模式 1 AU 处推进阵列功率（W）；与 t_max_n 恰一指定",
    )
    p_bus_w: float = Field(default=0.0, ge=0.0, description="平台常耗功率（W，仅 SEP 模式）")
    efficiency: float = Field(default=1.0, gt=0.0, le=1.0, description="功率-推力映射效率 η∈(0,1]")
    r_helio_km: float | None = Field(
        default=None,
        gt=0.0,
        description="SEP 功率评估固定日心距（km）；None 用段中点瞬时 |r|",
    )

    @model_validator(mode="after")
    def _validate_thrust_mode_exclusive(self) -> LowThrustPropulsionSpec:
        """常推力（t_max_n）与 SEP（p0_w）二选一（违反 → INVALID_PARAMS）。"""
        if (self.t_max_n is None) == (self.p0_w is None):
            raise ValueError("t_max_n 与 p0_w 必须恰好指定一个（常推力/SEP 二选一）")
        return self


class LowThrustNodeSpec(_ApiModel):
    """多 leg 链节点（rendezvous 位置+速度匹配 / flyby V∞ 旋转）。

    节点级语义（flyby 必填 mu/r_p、rendezvous 须为 None、末节点必须
    rendezvous 等）由算法层 ``SimsFlanaganNode``/``SimsFlanaganMultiLegProblem``
    校验，其 ``ValueError`` 翻译为 INVALID_PARAMS 且消息更精确。
    """

    kind: Literal["rendezvous", "flyby"] = Field(
        description="节点类型；末节点必须 rendezvous（算法层校验）"
    )
    state: list[float] = Field(
        min_length=6,
        max_length=6,
        description="节点状态 [x,y,z,vx,vy,vz]（中心体惯性系，km, km/s）",
    )
    mu_km3_s2: float | None = Field(
        default=None,
        gt=0.0,
        description="flyby 天体引力常数（km³/s²）；flyby 必填、rendezvous 须为 None",
    )
    r_p_min_km: float | None = Field(
        default=None,
        gt=0.0,
        description="flyby 最小近心点半径（km）；flyby 必填、rendezvous 须为 None",
    )


class LowThrustPreliminaryRequest(_ApiModel):
    """低推力转移 Sims-Flanagan 预设计输入（#725，ADR 0054 决策 2/3）。

    链结构：固定出发状态 → nodes[0] → … → 末节点，共 ``len(nodes)`` 个 leg；
    单 leg 转移 = 一个 rendezvous 节点。TOF 边界形状/序关系与节点语义由
    算法层校验（消息更精确），此处只做链维度与成本配置一致性。
    """

    backend: Literal["conic", "ephemeris"] = Field(
        description="保真度档（ADR 0050）：conic 二体封闭解 / ephemeris 星历 N 体数值传播"
    )
    epoch: Any | None = Field(
        default=None,
        description=(
            "出发历元 UTC（ISO 字符串或 [年,月,日,时,分,秒]）；"
            "backend=ephemeris 必填，conic 档须为 None"
        ),
    )
    bodies: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            '星历档天体集（SPICE 天体名，须含 origin，如 ["EARTH","MOON","SUN"]）；'
            "backend=ephemeris 必填，conic 档须为 None"
        ),
    )
    origin: str | None = Field(
        default=None,
        description=(
            "星历档坐标原点天体（SPICE 天体名，中心引力由其提供）；"
            "backend=ephemeris 必填，conic 档须为 None"
        ),
    )
    departure_state: list[float] = Field(
        min_length=6,
        max_length=6,
        description="出发状态 [x,y,z,vx,vy,vz]（中心体惯性系，km, km/s）",
    )
    nodes: list[LowThrustNodeSpec] = Field(
        min_length=1,
        description="节点链（出发 → nodes[0] → … → 末节点）；单 leg = 一个 rendezvous 节点",
    )
    leg_tofs_s: list[float] = Field(
        min_length=1, description="逐 leg 名义飞行时间（s），长度 = 节点数"
    )
    mu_km3_s2: float = Field(
        gt=0.0,
        description="中心天体引力常数（km³/s²）；日心问题取 e2m2e.data.constants 的 SUN.gm",
    )
    initial_mass_kg: float = Field(gt=0.0, description="初始质量（kg）")
    propulsion: LowThrustPropulsionSpec = Field(description="推进配置（常推力/SEP 二选一）")
    n_segments: int | list[int] = Field(
        default=20,
        description="每 leg 段数（≥2；统一整数或逐 leg 列表，长度 = 节点数）",
    )
    cost: Literal["min_fuel", "min_time", "weighted", "gtoc"] = Field(
        default="min_fuel",
        description="成本函数（定义见 sims_flanagan 模块 docstring）",
    )
    cost_weights: list[float] | None = Field(
        default=None,
        description="仅 weighted：[w_dv, w_t]，均 > 0；其余 cost 必须为 None",
    )
    tof_bounds_s: list[float] | list[list[float]] | None = Field(
        default=None,
        description=(
            "逐 leg TOF 决策变量边界（s）；(2,) 共享对或 (节点数, 2)；"
            "min_time/weighted 必填，其余 cost 须为 None"
        ),
    )
    guess: Literal["edelbaum"] | None = Field(
        default=None,
        description="初猜：Edelbaum 闭式螺旋（按 leg）；None 用零/默认初猜",
    )
    ftol: float = Field(default=1e-9, gt=0.0, description="SLSQP 目标容差")
    maxiter: int = Field(default=200, ge=1, description="SLSQP 最大迭代数")
    use_analytic_jac: bool = Field(
        default=True,
        description="True 用解析雅可比（f&g STM 链），False 退回数值差分",
    )

    @model_validator(mode="after")
    def _validate_chain_consistency(self) -> LowThrustPreliminaryRequest:
        """校验链维度与成本配置一致性（违反 → INVALID_PARAMS）。"""
        n_nodes = len(self.nodes)
        if len(self.leg_tofs_s) != n_nodes:
            raise ValueError(
                f"leg_tofs_s 长度须为 {n_nodes}（节点数），得到 {len(self.leg_tofs_s)}"
            )
        for k, tof in enumerate(self.leg_tofs_s):
            if not math.isfinite(tof) or tof <= 0.0:
                raise ValueError(f"leg_tofs_s[{k}] 必须为正的有限数，得到 {tof!r}")
        if isinstance(self.n_segments, int):
            if self.n_segments < 2:
                raise ValueError(f"n_segments 须 ≥ 2，得到 {self.n_segments}")
        elif len(self.n_segments) != n_nodes:
            raise ValueError(
                f"n_segments 列表长度须为 {n_nodes}（节点数），得到 {len(self.n_segments)}"
            )
        else:
            for k, n_seg in enumerate(self.n_segments):
                if n_seg < 2:
                    raise ValueError(f"n_segments[{k}] 须为 ≥ 2 的整数，得到 {n_seg}")
        if self.cost == "weighted":
            if self.cost_weights is None:
                raise ValueError("cost='weighted' 必须提供 cost_weights=[w_dv, w_t]")
            if len(self.cost_weights) != 2 or not all(
                math.isfinite(w) and w > 0.0 for w in self.cost_weights
            ):
                raise ValueError(f"cost_weights 须为两个正的有限数，得到 {self.cost_weights}")
        elif self.cost_weights is not None:
            raise ValueError(
                f"cost_weights 仅在 cost='weighted' 时接受，cost={self.cost!r} 下须为 None"
            )
        if self.cost in ("min_time", "weighted"):
            if self.tof_bounds_s is None:
                raise ValueError(f"cost={self.cost!r} 必须提供 tof_bounds_s（TOF 为决策变量）")
        elif self.tof_bounds_s is not None:
            raise ValueError(
                f"tof_bounds_s 仅在 cost='min_time'/'weighted' 时接受，"
                f"cost={self.cost!r} 下须为 None"
            )
        # 星历档上下文在场性（类型/有限性由算法层校验；模型只管档位一致性）。
        eph_ctx: dict[str, Any | None] = {
            "epoch": self.epoch,
            "bodies": self.bodies,
            "origin": self.origin,
        }
        if self.backend == "ephemeris":
            missing = [name for name, value in eph_ctx.items() if value is None]
            if missing:
                raise ValueError(
                    f"backend='ephemeris' 必须提供 {'、'.join(missing)}（星历档上下文）"
                )
        elif any(value is not None for value in eph_ctx.values()):
            raise ValueError(
                "backend='conic' 须 epoch/bodies/origin 均为 None（星历上下文仅 ephemeris 档使用）"
            )
        return self


class LowThrustLegInfo(_ApiModel):
    """单 leg 解剖面（Sims-Flanagan 双 pass 段中冲量转录）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    impulses_km_s: list[list[float]] = Field(description="段中冲量 (n, 3)（km/s）")
    impulse_times_s: list[float] = Field(description="冲量时刻（相对本 leg 起点，s）")
    node_times_s: list[float] = Field(description="段界时刻（s，含 leg 首尾节点）")
    forward_states: list[list[float]] = Field(description="前向 pass 段界状态（含本 leg 起点）")
    backward_states: list[list[float]] = Field(description="后向 pass 段界状态（含本 leg 终点）")
    matchpoint_residual: list[float] = Field(description="匹配点残差（6 维：位置 km + 速度 km/s）")


class LowThrustFlybyInfo(_ApiModel):
    """flyby 节点的解报告（V∞ 等模约束下的进入/离开渐近线速度）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    v_inf_in_km_s: list[float] = Field(description="进入 V∞ 向量 (3,)（km/s）")
    v_inf_out_km_s: list[float] = Field(description="离开 V∞ 向量 (3,)（km/s）")
    v_inf_km_s: float = Field(description="V∞ 幅值（km/s），取进入模（等模约束下与离开一致）")
    turn_angle_deg: float = Field(description="实现转角 δ（度）")
    pericenter_radius_km: float | None = Field(
        description="由 (δ, v_eff) 反解的近心点半径 (km)；δ=0（零退化）时为 null（#698 口径）"
    )


class LowThrustPreliminaryResponse(ResultResponse):
    """低推力转移 Sims-Flanagan 预设计输出（#725）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    legs: list[LowThrustLegInfo] = Field(description="逐 leg 解剖面，长度 = 节点数")
    flybys: list[LowThrustFlybyInfo] = Field(
        description="逐 flyby 节点的报告（仅 flyby 节点，按节点序）"
    )
    leg_tofs_s: list[float] = Field(description="优化后逐 leg 飞行时间（s）")
    delta_v_total_km_s: float = Field(description="全部段 Σ‖ΔVᵢ‖（km/s）")
    final_mass_kg: float = Field(description="末态质量（kg），m₀·exp(−Σ‖ΔVᵢ‖/c)")
    fuel_kg: float = Field(description="燃料消耗（kg），m₀ − final_mass_kg")
    objective_value: float = Field(description="求解所用成本函数在解处的值")
    cost: str = Field(description="求解所用成本函数名")
    n_iter: int = Field(description="SLSQP 迭代次数")
