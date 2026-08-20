"""任务轨道设计入口。

链路（对齐 ``docs/plans/dfh-parity-prd.md`` FR1）：

1. CR3BP 初猜：按形状参数生成周期轨道（``cr3bp_orbits``）；
2. 星历修正：周期轨道采样 patch points → synodic→J2000 转换 →
   星历 N 体模型下多重打靶收敛。稳定轨道（DRO 等）走 two_level
   （Rust 打靶 + vel_weight），不稳定轨道（Halo/NRHO）走 segmented
   （分段打靶拼接，全程分段约束、不依赖自由外推）；
3. 标称星历：以修正后状态为初值，在 ``perturbation_to_force_config``
   映射出的高精度力模型下生成——two_level 自由外推整段 duration，
   segmented 逐段积分填满 et_grid。输出文本格式星历（``EphemerisTable``）。

参数语义对齐 MATLAB ``design_orbit.m`` 与 inputs-dac.txt 设计块：
DRO 振幅+初始相位；Halo 共线点编号+带符号面外振幅+初始相位；
NRHO 共线点编号+北/南+近月点高度+初始相位。初始相位为周期份额
（0~1），历元时刻的状态 = 周期轨道参考状态沿轨道推进 ``phase × T``；
相位零点按历史标定——Halo/NRHO 在 y=0 穿越点，DRO 在远侧
x 轴穿越点（e2m2e 的 DRO 参考状态为近侧穿越点，内部偏移半周期）。
DRO 振幅取一个周期内距月距离最小/最大值的均值（同按历史标定）。

已知系统差（历史标定值）：

- 参考输出 GCRS，e2m2e 在 ICRF（J2000）下传播，frame bias（~23 mas）
  在月距量级约 0.04 km，计入对比容差；
- 维持时间按 1 年 = 365.25 天折算；
- NRHO 近月点高度起算面取月球平均半径 1737.4 km。
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from e2m2e.data.types.trajectory import EphemerisTable

from ...data.constants import SECONDS_PER_DAY
from ...data.kernels.manager import SPICEManager
from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.perturbations import DEFAULT_PERTURBATION
from ...data.types.orbit import Orbit
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from ..coordinate.synodic_j2000 import SynodicJ2000System
from ..dynamics import CR3BP_Dynamics, EphemerisSystem
from ..family.cr3bp_orbits import (
    design_axial,
    design_dpo,
    design_dro,
    design_halo,
    design_horseshoe,
    design_lissajous,
    design_lpo,
    design_nrho,
    design_spo,
    design_triangular,
    earth_moon_system,
)
from ..results import EphemerisCorrectionResult, ResultStatus, StageRecord
from ..solver.multiple_shooting import (
    sample_patch_points_drop_near_perilune,
    sample_patch_points_perilune_clustered,
)

if TYPE_CHECKING:
    from ...api.models import DesignOrbitRequest

__all__ = [
    "DesignNotConvergedError",
    "OrbitDesignResult",
    "default_kernel_dir",
    "design_orbit",
    "load_design_kernels",
]

#: 设计链路的默认摄动开关：光压用炮弹模型（``solar_radiation=1``）、关耦合项。
#: ECOM 光压与地球非球形×大天体耦合项未实现（#253），显式开启时抛
#: ``NotImplementedError``；约定同 ``control_orbit._DEFAULT_CTRL_PERTURBATION``。
DEFAULT_DESIGN_PERTURBATION: dict[str, int] = {
    **DEFAULT_PERTURBATION,
    "solar_radiation": 1,
    "coupling": 0,
}

#: 星历修正的默认收敛容差（km，6 维状态 max 范数：位置 km + 速度 km/s）。
#:
#: 取 2e-2（20 m）。地月尺度（特征长度 3.84e5 km）下 10 m 已属高精度、
#: 1 km 都不错，0.1 m（原 1e-4）对轨道保形无实质增益，却超出星历模型与
#: CR3BP 初猜的物理可收敛底线（紧凑近月轨道 ~5.7e-5 km，见下），导致
#: 求解器停在 ~1.2e-2 km 永远报 not converged。容差 2e-2 给 solver 留裕度
#: （实测单圈收敛残差 1.7e-2、3 圈段 1.5e-2，均 < 2e-2 明确收敛），迭代数
#: 大减（10→4），保形不受影响（实测 30 天 |r| 首 480967→末 476884 km，
#: 会合系 x∈[1.08,1.19] 紧邻 L2）。
#:
#: 历史依据：原 1e-4 的注释——紧凑近月 DRO（amplitude=10000，月距~1 万 km）
#: 因月球引力梯度强，CR3BP 几何与真实星历 N 体动力学存在 ~5.7e-5 km 的
#: 物理收敛底线；1e-4 相对底线留约 1.7 倍裕度。该底线说明 0.1 m 级收敛对
#: 部分轨道本就不可达，进一步佐证 2e-2 是更贴合物理的默认。
CORRECTION_TOL_KM = 2e-2

#: 星历修正的速度连续目标（km/s）。0.01 m/s——patch point 速度跳变小于此即视为
#: 光滑轨道（非需脉冲拼接的断弧）。依据：DRO 等中性稳定轨道，速度连续的修正解
#: 才落在准周期轨道上，自由外推才有界（实测 amp=60000 DRO 坏历元，速度残差
#: 从 ~25 m/s 压到 <0.01 m/s 后，30 天传播从发散 200000 km 收敛到有界 ~80000 km）。
VELOCITY_TOL_KMS = 1e-5

#: 多重打靶速度残差加权 = 位置容差 / 速度容差。Rust 打靶残差向量把位置（km）与
#: 速度（km/s）混在一起取 ‖F‖²，cislunar 下位置项（几百 km）单边主导，速度项被
#: 忽略，求解器停在"位置连续 / 速度跳变数十 m/s"的局部极小。乘以 vel_weight 后
#: 两者在容差尺度可比，LM 真正压速度连续（见 Rust ``build_residual`` 注释）。
CORRECTION_VEL_WEIGHT = CORRECTION_TOL_KM / VELOCITY_TOL_KMS

#: 每圈 patch 节点数（均匀采样基线；NRHO/Halo 按族覆盖）
_POINTS_PER_REV = 8

#: 拼接点采样策略（按轨道族覆盖，不暴露到请求模型）：
#: - uniform：等时间（NRHO 生产默认，#473；其余族默认）
#: - perilune_clustered：近月点加密（Halo 长弧实测更稳）
#: - drop_near_perilune：删近月点附近节点（工具函数/对照；
#:   #463 曾作 NRHO 默认，#473 起不再作生产默认——phase=0.5
#:   约 1 个月弧 + 3 圈/段合并层易卡，且未钉历元会出前缀空洞）
_PATCH_SAMPLING_UNIFORM = "uniform"
_PATCH_SAMPLING_PERILUNE_CLUSTERED = "perilune_clustered"
_PATCH_SAMPLING_DROP_NEAR_PERILUNE = "drop_near_perilune"

#: 星历修正用固定时间打靶（var_time=False）的轨道族：Halo/NRHO（不稳定，
#: 分段打靶全程固定时刻，对齐杨洪伟 2015）、拟周期/无周期闭合族
#: （Lissajous / 三角平动点 L4/L5）与 Axial。
#:
#: 拟周期族用固定时间的机理（#366）：CR3BP 初猜无周期闭合（Lissajous
#: 面内/面外频率不可约；L4/L5 短/长周期模态耦合），自由时间模式下时间
#: 自由度与沿流状态自由度近似线性相关（时间平移 δt ≈ 沿轨道移动 δt·f），
#: 雅可比列病态，LM 陷入线性收敛卡在 0.5–174 km（实测 L2/L4/L5 迭代到
#: 80 次上限不收敛）。固定时间下节点时刻保持 CR3BP 名义周期均匀采样，
#: 位置/速度修正直接吸收星历偏差，Gauss-Newton 二次收敛（实测 4–6 迭代，
#: 秒级到几十秒）。
#:
#: Axial 同属此病态：它从 Lyapunov 族 1:1 共振分岔（Gómez Type B）产生，
#: 分岔邻域面内周期 = 面外周期，时间平移与面外相位平移近似简并，自由
#: 时间打靶雅可比列病态——实测 L2/L1 默认参数 LM 停滞（
#: STAGNATION_DETECTED，15/17 次迭代后位置残差停在 1.5e-01 / 1.1e+01 km）；
#: 固定时间后两种修正方法均在约 10 s 内收敛到容差内。
_FIXED_TIME_ORBIT_TYPES = frozenset({"HALO", "NRHO", "LISSAJOUS", "L4", "L5", "AXIAL"})

#: body-fixed 帧（ITRF93 / MOON_PA）所需内核文件名，与 tests/kernel_helpers.py 一致
_BODY_FIXED_KERNELS = [
    "earth_latest_high_prec.bpc",
    "pck00010.tpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
]


class DesignNotConvergedError(RuntimeError):
    """任务轨道设计未生成可用标称轨道。"""

    def __init__(
        self,
        message: str,
        *,
        status: ConvergenceState = ConvergenceState.FAILED,
        cause: FailureCause = FailureCause.UNKNOWN,
    ) -> None:
        super().__init__(message)
        ResultStatus(status, cause, message)
        self.status = status
        self.cause = cause
        self.message = message


@dataclass
class OrbitDesignResult:
    """任务轨道设计结果。

    Attributes:
        orbit_type: 轨道类型（``"DRO"`` / ``"HALO"`` / ``"NRHO"`` / ``"ELFO"``）。
        epoch_utc: 起始历元 UTC（ISO 字符串）。
        duration_day: 维持时间（天）。
        output_step_sec: 星历输出间隔（秒）。
        initial_state: 历元时刻惯性系状态（km, km/s），星历修正后首节点。
        ephemeris: 标称星历（文本格式容器：UTC + GCRS 位置 km /
            速度 m/s + 地月会合系无量纲位置）。
        cr3bp_orbit: CR3BP 周期轨道（参考相位，无量纲）；ELFO 场景为 None。
        cr3bp_jacobi: CR3BP 周期轨道的 Jacobi 常数；ELFO 场景为 nan。
        correction: 星历修正结果（收敛标志、迭代次数、残差历史、修正后
            patch points）；ELFO 场景为 None。
        force_config: 标称预报使用的力模型配置字典。
        drift_e: 传播弧段 Δe 首末差（仅 ELFO）。
        drift_aop_deg: 传播弧段 Δω 首末差（度，仅 ELFO）。
        drift_rp_km: 传播弧段 Δrp 首末差（km，仅 ELFO）。
        secular_aop_rate_deg_per_year: ω 线性拟合年漂移率（仅 ELFO）。
        moon_centric_elements: 月心惯性系根数序列（仅 ELFO）。
    """

    orbit_type: str
    epoch_utc: str
    duration_day: float
    output_step_sec: float
    initial_state: np.ndarray
    ephemeris: EphemerisTable
    cr3bp_orbit: Orbit | None
    cr3bp_jacobi: float
    correction: EphemerisCorrectionResult | None
    force_config: dict[str, Any]
    status: ConvergenceState = ConvergenceState.CONVERGED
    cause: FailureCause = FailureCause.NONE
    message: str = "任务完成"
    stages: tuple[StageRecord, ...] = ()
    drift_e: float | None = None

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    drift_aop_deg: float | None = None
    drift_rp_km: float | None = None
    secular_aop_rate_deg_per_year: float | None = None
    moon_centric_elements: dict[str, np.ndarray] | None = None

    def write_ephemeris(self, path: str | Path) -> None:
        """按文本格式写出标称星历。"""
        from ...data.types.trajectory import write_ephemeris

        write_ephemeris(self.ephemeris, path)


def _design_stages() -> tuple[StageRecord, ...]:
    """返回成功设计链路的阶段记录。"""
    return (
        StageRecord(
            "initial_guess",
            applicable=True,
            executed=True,
            result_status=ConvergenceState.CONVERGED,
        ),
        StageRecord(
            "ephemeris_correction",
            applicable=True,
            executed=True,
            result_status=ConvergenceState.CONVERGED,
        ),
        StageRecord(
            "propagation", applicable=True, executed=True, result_status=ConvergenceState.CONVERGED
        ),
    )


def default_kernel_dir() -> str:
    """仓库自带 SPICE 内核目录（``kernels/``）。"""
    return str(Path(__file__).resolve().parent.parent.parent.parent / "kernels")


def load_design_kernels(spice: SPICEManager, kernel_dir: str | None = None) -> list[str]:
    """加载设计链路所需内核：行星历 + body-fixed 帧内核。

    行星名→质心/本体 NAIF ID 别名由 :meth:`SPICEManager.load_kernel` 首次
    调用时统一注册（双侧同步，见 ``_BODY_ID_ALIASES``）。

    返回实际加载的内核路径列表（调用方管理卸载）。
    """
    kernel_dir = kernel_dir or default_kernel_dir()
    loaded: list[str] = []
    for name in ["de440s.bsp", "de430.bsp"]:
        path = os.path.join(kernel_dir, name)
        if os.path.exists(path):
            spice.load_kernel(path)
            loaded.append(path)
            break
    else:
        raise FileNotFoundError(f"行星历内核不存在（de440s/de430）: {kernel_dir}")
    for name in _BODY_FIXED_KERNELS:
        path = os.path.join(kernel_dir, name)
        if os.path.exists(path):
            spice.load_kernel(path)
            loaded.append(path)
    return loaded


def _epoch_to_iso(epoch: Sequence[float] | str) -> str:
    """起始历元统一为 ISO UTC 字符串。接受 ``[年, 月, 日, 时, 分, 秒]`` 或字符串。"""
    if isinstance(epoch, str):
        return epoch
    parts = list(epoch)
    if len(parts) != 6:
        raise ValueError(f"epoch 必须为 6 分量 [年, 月, 日, 时, 分, 秒]，当前 {len(parts)} 个")
    y, mo, d, h, mi, s = parts
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(h):02d}:{int(mi):02d}:{float(s):06.3f}"


def _validate_params(
    sel: str,
    *,
    amplitude: float | None,
    phase: float | None,
    collinear_point: int | None,
    north_south: int | None,
    perilune_height: float | None,
    amplitude_in: float | None,
    amplitude_out: float | None,
    phase_in: float | None,
    phase_out: float | None,
) -> dict[str, float | int]:
    """按类型校验形状参数并填默认值（对齐 MATLAB ``design_orbit.m`` 的
    类型依赖默认值与取值范围），返回规范化参数。"""
    if sel == "DRO":
        amplitude = 10000.0 if amplitude is None else float(amplitude)
        phase = 0.5001 if phase is None else float(phase)
        if not 1737.0 <= amplitude <= 110000.0:
            raise ValueError(f"DRO amplitude 应在 1737~110000 km 之间，实际为 {amplitude:.0f} km")
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"DRO phase 应在 0~1 之间，实际为 {phase}")
        return {"amplitude": amplitude, "phase": phase}

    if sel == "DPO":
        amplitude = 20000.0 if amplitude is None else float(amplitude)
        phase = 0.5001 if phase is None else float(phase)
        if not 1737.0 <= amplitude <= 110000.0:
            raise ValueError(f"DPO amplitude 应在 1737~110000 km 之间，实际为 {amplitude:.0f} km")
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"DPO phase 应在 0~1 之间，实际为 {phase}")
        return {"amplitude": amplitude, "phase": phase}

    if sel == "HALO":
        collinear_point = 2 if collinear_point is None else int(collinear_point)
        amplitude = 30000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if collinear_point not in (1, 2):
            raise ValueError(f"Halo collinear_point 必须为 1 或 2，当前 {collinear_point}")
        if not abs(amplitude) <= 73000.0:
            raise ValueError(f"Halo amplitude 应在 -73000~73000 km 之间，实际为 {amplitude:.0f} km")
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"Halo phase 应在 0~1 之间，实际为 {phase}")
        return {
            "collinear_point": collinear_point,
            "amplitude": amplitude,
            "phase": phase,
        }

    if sel == "NRHO":
        collinear_point = 2 if collinear_point is None else int(collinear_point)
        north_south = 2 if north_south is None else int(north_south)
        perilune_height = 5000.0 if perilune_height is None else float(perilune_height)
        phase = 0.5 if phase is None else float(phase)
        if collinear_point not in (1, 2):
            raise ValueError(f"NRHO collinear_point 必须为 1 或 2，当前 {collinear_point}")
        if north_south not in (1, 2):
            raise ValueError(f"NRHO north_south 必须为 1（北）或 2（南），当前 {north_south}")
        if not 100.0 <= perilune_height <= 10000.0:
            raise ValueError(
                f"NRHO perilune_height 应在 100~10000 km 之间，实际为 {perilune_height:.0f} km"
            )
        # NRHO 的历元相位允许从 0 开始；API 模型同样定义为 [0, 1]。
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"NRHO phase 应在 0~1 之间，实际为 {phase}")
        return {
            "collinear_point": collinear_point,
            "north_south": north_south,
            "perilune_height": perilune_height,
            "phase": phase,
        }

    if sel == "LISSAJOUS":
        collinear_point = 2 if collinear_point is None else int(collinear_point)
        amplitude_in = 2500.0 if amplitude_in is None else float(amplitude_in)
        amplitude_out = 7500.0 if amplitude_out is None else float(amplitude_out)
        phase_in = 0.01 if phase_in is None else float(phase_in)
        phase_out = 0.55 if phase_out is None else float(phase_out)
        if collinear_point not in (1, 2, 3):
            raise ValueError(f"Lissajous collinear_point 必须为 1/2/3，当前 {collinear_point}")
        limit = 100000.0 if collinear_point == 3 else 7600.0
        if not 0.0 < amplitude_in <= limit:
            raise ValueError(
                f"Lissajous L{collinear_point} amplitude_in 应在 (0, {limit:.0f}] km，"
                f"实际为 {amplitude_in:.0f} km"
            )
        if not 0.0 < amplitude_out <= limit:
            raise ValueError(
                f"Lissajous L{collinear_point} amplitude_out 应在 (0, {limit:.0f}] km，"
                f"实际为 {amplitude_out:.0f} km"
            )
        if not 0.0 <= phase_in <= 1.0:
            raise ValueError(f"Lissajous phase_in 应在 0~1 之间，实际为 {phase_in}")
        if not 0.0 <= phase_out <= 1.0:
            raise ValueError(f"Lissajous phase_out 应在 0~1 之间，实际为 {phase_out}")
        return {
            "collinear_point": collinear_point,
            "amplitude_in": amplitude_in,
            "amplitude_out": amplitude_out,
            "phase_in": phase_in,
            "phase_out": phase_out,
        }

    if sel in ("L4", "L5"):
        amplitude_in = 8000.0 if amplitude_in is None else float(amplitude_in)
        amplitude_out = 6000.0 if amplitude_out is None else float(amplitude_out)
        phase_in = 0.0 if phase_in is None else float(phase_in)
        phase_out = 0.0 if phase_out is None else float(phase_out)
        if not 0.0 < amplitude_in <= 10000.0:
            raise ValueError(
                f"{sel} amplitude_in 应在 (0, 10000] km 之间，实际为 {amplitude_in:.0f} km"
            )
        if not 0.0 < amplitude_out <= 76000.0:
            raise ValueError(
                f"{sel} amplitude_out 应在 (0, 76000] km 之间，实际为 {amplitude_out:.0f} km"
            )
        if not 0.0 <= phase_in <= 1.0:
            raise ValueError(f"{sel} phase_in 应在 0~1 之间，实际为 {phase_in}")
        if not 0.0 <= phase_out <= 1.0:
            raise ValueError(f"{sel} phase_out 应在 0~1 之间，实际为 {phase_out}")
        return {
            "amplitude_in": amplitude_in,
            "amplitude_out": amplitude_out,
            "phase_in": phase_in,
            "phase_out": phase_out,
        }

    if sel == "AXIAL":
        collinear_point = 2 if collinear_point is None else int(collinear_point)
        amplitude = 5000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if collinear_point not in (1, 2, 3):
            raise ValueError(f"Axial collinear_point 必须为 1/2/3，当前 {collinear_point}")
        if abs(amplitude) > 60000.0:
            raise ValueError(
                f"Axial amplitude 应在 -60000~60000 km 之间，实际为 {amplitude:.0f} km"
            )
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"Axial phase 应在 0~1 之间，实际为 {phase}")
        return {
            "collinear_point": collinear_point,
            "amplitude": amplitude,
            "phase": phase,
        }

    if sel in ("L4_SPO", "L5_SPO"):
        amplitude = 10000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if not 1737.0 <= amplitude <= 200000.0:
            raise ValueError(f"{sel} amplitude 应在 1737~200000 km 之间，实际为 {amplitude:.0f} km")
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"{sel} phase 应在 0~1 之间，实际为 {phase}")
        return {"amplitude": amplitude, "phase": phase}

    if sel in ("L4_LPO", "L5_LPO"):
        amplitude = 50000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if not 1000.0 <= amplitude <= 110000.0:
            raise ValueError(f"{sel} amplitude 应在 1000~110000 km 之间，实际为 {amplitude:.0f} km")
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"{sel} phase 应在 0~1 之间，实际为 {phase}")
        return {"amplitude": amplitude, "phase": phase}

    if sel in ("L4_HORSESHOE", "L5_HORSESHOE"):
        amplitude = 100000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if not 50000.0 <= amplitude <= 110000.0:
            raise ValueError(
                f"{sel} amplitude 应在 50000~110000 km 之间，实际为 {amplitude:.0f} km"
            )
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"{sel} phase 应在 0~1 之间，实际为 {phase}")
        return {"amplitude": amplitude, "phase": phase}

    raise ValueError(
        f"orbit_type 必须为 DRO/DPO/NRHO/Halo/Lissajous/L4/L5/Axial/L4_SPO/L5_SPO"
        f"/L4_LPO/L5_LPO/L4_HORSESHOE/L5_HORSESHOE，当前 {sel!r}"
    )


def _cr3bp_orbit_for(sel: str, params: dict[str, float | int], dynamics: CR3BP_Dynamics) -> Orbit:
    """按规范化形状参数生成 CR3BP 周期轨道。"""
    if sel == "DRO":
        return design_dro(params["amplitude"], dynamics=dynamics)
    if sel == "DPO":
        return design_dpo(params["amplitude"], dynamics=dynamics)
    if sel == "HALO":
        return design_halo(int(params["collinear_point"]), params["amplitude"], dynamics=dynamics)
    if sel == "NRHO":
        return design_nrho(
            int(params["collinear_point"]),
            int(params["north_south"]),
            params["perilune_height"],
            dynamics=dynamics,
        )
    if sel == "LISSAJOUS":
        return design_lissajous(
            int(params["collinear_point"]),
            params["amplitude_in"],
            params["amplitude_out"],
            params["phase_in"],
            params["phase_out"],
            dynamics=dynamics,
        )
    if sel == "AXIAL":
        return design_axial(
            int(params["collinear_point"]),
            params["amplitude"],
            dynamics=dynamics,
        )
    if sel in ("L4_SPO", "L5_SPO"):
        return design_spo(
            4 if sel == "L4_SPO" else 5,
            params["amplitude"],
            dynamics=dynamics,
        )
    if sel in ("L4_LPO", "L5_LPO"):
        return design_lpo(
            4 if sel == "L4_LPO" else 5,
            params["amplitude"],
            dynamics=dynamics,
        )
    if sel in ("L4_HORSESHOE", "L5_HORSESHOE"):
        return design_horseshoe(
            4 if sel == "L4_HORSESHOE" else 5,
            params["amplitude"],
            dynamics=dynamics,
        )
    return design_triangular(
        4 if sel == "L4" else 5,
        params["amplitude_in"],
        params["amplitude_out"],
        params["phase_in"],
        params["phase_out"],
        dynamics=dynamics,
    )


def _dense_orbit(
    dynamics: CR3BP_Dynamics, state0: np.ndarray, period: float, n_points: int = 720
) -> Orbit:
    """从 ``state0`` 传播一个周期的稠密轨道（供 patch points 采样）。"""
    t_eval = np.linspace(0.0, period, n_points + 1)
    result = dynamics.propagate(state0, (0.0, period), t_eval=t_eval)
    orbit = Orbit(states=result["states"], times=result["time"], system=dynamics.system)
    orbit.period = period
    return orbit


def _patch_sampling_for(orbit_type: str) -> str:
    """按轨道族选择拼接点采样策略（内部策略，不进请求契约）。

    NRHO 与 Halo 解耦（#473）：NRHO 默认等时间；Halo 近月点加密。
    删近月点采样保留在 ``_sample_patch_points`` 分派中供对照，不作生产默认。
    """
    if orbit_type == "HALO":
        return _PATCH_SAMPLING_PERILUNE_CLUSTERED
    return _PATCH_SAMPLING_UNIFORM


def _sample_patch_points(
    dynamics: CR3BP_Dynamics,
    state0: np.ndarray,
    period: float,
    n_revolutions: int,
    *,
    sampling: str = _PATCH_SAMPLING_UNIFORM,
) -> tuple[np.ndarray, np.ndarray]:
    """从历元状态出发，在 ``n_revolutions`` 圈上采样 patch points（synodic）。

    默认每圈 ``_POINTS_PER_REV`` 等时间点。``sampling`` 覆盖族相关策略：

    - ``perilune_clustered``：近月点加密（Halo）
    - ``drop_near_perilune``：删近月点附近节点（对照/研究；#473 起非生产默认）
    - ``uniform``：等时间（NRHO 与其余族默认）
    """
    dense = _dense_orbit(dynamics, state0, period)
    if sampling == _PATCH_SAMPLING_PERILUNE_CLUSTERED:
        t_rel, states = sample_patch_points_perilune_clustered(dense, dynamics)
    elif sampling == _PATCH_SAMPLING_DROP_NEAR_PERILUNE:
        t_rel, states = sample_patch_points_drop_near_perilune(
            dense, dynamics, n_points=_POINTS_PER_REV
        )
    else:
        t_rel = np.linspace(0.0, period, _POINTS_PER_REV, endpoint=False)
        states = np.empty((len(t_rel), 6))
        for i in range(6):
            states[:, i] = np.interp(t_rel, dense.times, dense.states[:, i])

    t_patch = np.concatenate([t_rel + k * period for k in range(n_revolutions)])
    state_patch = np.tile(states, (n_revolutions, 1))
    return t_patch, state_patch


def _sample_patch_points_from_trajectory(
    orbit: Orbit, period: float, n_revolutions: int
) -> tuple[np.ndarray, np.ndarray]:
    """从轨道自带稠密轨迹插值 patch points（准周期 Lissajous 用）。

    准周期 Lissajous 不能用 :func:`_sample_patch_points`——它原生 CR3BP
    重传播 ``states[0]``，会重新激发不稳定方向而发散（issue #323 根因）。
    改从 ``orbit`` 的中心流形有界轨迹跨 ``n_revolutions`` 圈均匀采样
    ``_POINTS_PER_REV`` 点/圈、逐分量线性插值。调用方须保证
    ``orbit.times`` 覆盖 ``[0, n_revolutions·period]``。
    """
    n_points = _POINTS_PER_REV * n_revolutions
    t_patch = np.linspace(0.0, n_revolutions * period, n_points, endpoint=False)
    states = np.empty((n_points, 6))
    for i in range(6):
        states[:, i] = np.interp(t_patch, orbit.times, orbit.states[:, i])
    return t_patch, states


def _build_ephemeris_table(
    spice: SPICEManager,
    syn_j2000: SynodicJ2000System,
    et0: float,
    et_grid: np.ndarray,
    states: np.ndarray,
) -> EphemerisTable:
    """把传播结果组装成文本格式星历表（UTC + GCRS + 地月会合系）。

    地月会合系为地心归一（月球在 +x 单位距离处；依据：历史标定样本中
    DRO 轨道在该约定下关于 x=1 近似对称）；内部转换器输出质心归一（月球在
    1-mu），x 分量加 mu 平移对齐。
    """
    t_c = syn_j2000.cr3bp_system.characteristic_time
    assert t_c is not None
    # 位置/速度与时间网格必须逐点对齐：segmented 逐段积分曾把 propagate
    # 自动追加的段终点 tf 拼进 states（位置数组比时间网格多出段数个点），
    # batch_j2000_to_synodic 按索引配对位置与旋转时刻，错位逐段累积导致
    # 会合系曲线一圈一圈偏离周期轨道。此断言防未来回归。
    if len(states) != len(et_grid):
        raise ValueError(
            f"星历状态点数 {len(states)} 与时间网格点数 {len(et_grid)} 不一致，位置与时间将错位"
        )
    t_syn = (et_grid - et0) / t_c
    synodic = syn_j2000.batch_j2000_to_synodic(states, t_syn, et0)[:, :3]
    synodic[:, 0] += syn_j2000.cr3bp_system.mu

    # ET→UTC 日历分量批量下沉 Rust（frame_convert.batch_et_to_utc_py）：
    # 免去逐点 spiceypy.et2utc FFI + datetime.fromisoformat 字符串解析
    # （一年 8766 点的逐点 Python 循环）。
    from e2m2e.integrators import batch_et_to_utc_py, require_rust_extension

    require_rust_extension("batch_et_to_utc_py")
    y, mo, d, h, mi, s = batch_et_to_utc_py([float(x) for x in et_grid])
    year = np.asarray(y, dtype=int)
    month = np.asarray(mo, dtype=int)
    day = np.asarray(d, dtype=int)
    hour = np.asarray(h, dtype=int)
    minute = np.asarray(mi, dtype=int)
    second = np.asarray(s, dtype=float)

    return EphemerisTable(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        position_km=states[:, :3].copy(),
        velocity_mps=states[:, 3:] * 1000.0,
        synodic_position=synodic,
    )


def _design_apolune_segmented(
    forces_py: list[Any],
    observer: str,
    t_patch_j2000: np.ndarray,
    state_patch_j2000: np.ndarray,
    revs_per_group: int,
    points_per_rev: int,
    *,
    max_iter: int = 50,
    tolerance: float = CORRECTION_TOL_KM,
    vel_weight: float = CORRECTION_VEL_WEIGHT,
    var_time: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """分段打靶星历转换（朱彦伟 2026 多重打靶拼接，Rust 实现）。

    将 CR3BP 周期解转换到星历模型：整条 CR3BP tile 按 ``revs_per_group``
    圈切段，每段独立多重打靶转星历；段数 >1 时分层两两合并（合并段全节点
    自由、最小范数更新，对齐文献）直至整条连续。

    计算全部下沉 Rust ``segmented_shooting_correct``：切段、第 1 步各段
    打靶（段间独立 rayon 并行）、分层合并（同层配对合并段 rayon 并行）。
    本函数仅做参数装配与结果归一（此前分段/合并循环在 Python，逐段串行调
    打靶——#400 性能修复下沉）。

    关键配置（实测 Halo/NRHO，对齐文献）：

    - **第 1 步段长**：Halo/NRHO 用多圈长段（调用方传 ``min(n_rev, 3)``）。
      长段节点密、段内约束强，单弧打靶即可收敛且各段不漂离真实动力学；
      1 圈短段各段独立修正后会漂走（seam 跳 ~1e5 km，合并层无法消除——
      STM 条件数分析见 Liu & Liu 2025 §3）。
    - **合并层**：合并段全节点自由（``fixed_node_mask=None``），LM 最小范数
      更新对齐文献。固定首末锚定曾致合并层不收敛（#400）：各段独立打靶后
      seam 不连续，锚定把修正全压给内部节点，60 天合并层停在 7.5e-01 km；
      去锚定后 60 天合并层收敛到 1.4e-03 km，180 天三层合并全程收敛
      （5.8e-03 / 5.7e-04 / 1.4e-02 km）。

      对照论文的合并层节点稀疏化（每圈仅 1 个远月点节点，约束更疏、矩阵
      更良态，issue #400 需求②）：当前全节点合并 180 天 5 段 3 层已收敛到
      1.4e-02 km，说明当前圈数下全节点矩阵病态未显现，不跟进；年量级
      （50+ 圈）时矩阵规模与病态会放大，稀疏化留作该场景的前置评估。
    - **var_time**：Halo/NRHO 固定节点时刻（False，对齐杨洪伟 2015、
      刘刚 2017）；稳定轨道（DRO 等）保留自由时刻（True）吸收 CR3BP→星历
      的时间偏差。

    星历下 Halo/NRHO 的圈间漂移是标称轨道的固有准周期特征（星历非严格周期），
    由轨道保持（``algorithm.station_keeping``）处理，不在本转换范围内。

    Args:
        forces_py: 全摄动 Rust forces 序列（与长期预报同模型）。
        observer: 坐标系原点（"EARTH"）。
        t_patch_j2000 / state_patch_j2000: 整条 CR3BP tile（J2000, km/km/s）。
        revs_per_group: 第 1 步每组圈数。短期（≤ 该圈数）单段即收敛；
            长期分段后由分层合并拼接。
        points_per_rev: 每圈节点数（由采样 tile 推断；族策略下可能非整除 8）。
        var_time: 节点时刻是否作为自由变量。Halo/NRHO 传 False（见上）。

    Returns:
        ``(t_patch, state_patch, max_residual)`` （J2000），整条连续星历轨迹与
        全程各段/合并段的最大打靶残差（km）。
    """
    from e2m2e.integrators import segmented_shooting_correct_py

    try:
        result = segmented_shooting_correct_py(
            forces_py,
            observer,
            list(t_patch_j2000),
            [list(map(float, x)) for x in state_patch_j2000],
            revs_per_group=revs_per_group,
            per_rev=points_per_rev,
            var_time=var_time,
            max_iter_per_segment=max_iter,
            tolerance=tolerance,
            rtol=1e-10,
            vel_weight=vel_weight,
            verbose=verbose,
        )
    except RuntimeError as e:
        raise DesignNotConvergedError(
            f"分段打靶拼接积分失败: {e}",
            cause=FailureCause.INTEGRATION_FAILED,
        ) from e
    if result.status is not ConvergenceState.CONVERGED:
        raise DesignNotConvergedError(
            f"{result.message}（容差 {tolerance:.3e} km）",
            status=result.status,
            cause=result.cause,
        )
    return (
        np.asarray(result.t_patch, dtype=float),
        np.asarray(result.state_patch, dtype=float),
        float(result.max_residual),
    )


def _design_elfo(
    request: DesignOrbitRequest,
    spice: SPICEManager,
    kernel_dir: str,
    verbose: bool,  # noqa: ARG001
) -> OrbitDesignResult:
    """ELFO 冻结轨道管线：经典根数 → 地心初值 → 全摄动传播 → 月心漂移分析。"""

    from ..forces import ForceModel
    from ..forces.force_mapping import perturbation_to_force_config  # noqa: I001
    from .frozen_orbit import (
        MU_MOON,
        R_MOON,
        _compute_drift,
        _extract_moon_centric_elements,
        _oe2cart,
    )

    # model_validator 已确保 ELFO 必填字段非 None
    assert request.semi_major_axis is not None
    assert request.inclination is not None
    assert request.arg_of_pericenter is not None
    assert request.perilune_height is not None
    assert request.duration is not None

    a = float(request.semi_major_axis)
    rp = R_MOON + float(request.perilune_height)
    e = 1.0 - rp / a
    i_deg = float(request.inclination)
    aop_deg = float(request.arg_of_pericenter)
    duration_sec = float(request.duration)
    output_step = float(request.output_step)
    perturbation = (
        request.perturbation if request.perturbation is not None else DEFAULT_DESIGN_PERTURBATION
    )

    epoch_iso = _epoch_to_iso(request.epoch)
    et0 = spice.utc_to_et(epoch_iso)

    # 力模型（与 CR3BP 管线同路径）
    sw = dict(perturbation)
    bodies = ["EARTH", "MOON"]
    if sw["sun_body"]:
        bodies.append("SUN")
    if sw["planets"]:
        bodies += ["MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]
    full_system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
    full_system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    force_config = perturbation_to_force_config(
        perturbation,
        earth_degree=request.earth_degree,
        moon_degree=request.moon_degree,
        dyb=request.dyb,
    )
    fm = ForceModel.from_config(force_config, full_system)
    fm.rtol = 1e-12
    fm.atol = 1e-12
    fm.max_step = 600.0

    # 初值：月心根数 → 月心笛卡尔 → 叠加月球地心状态
    moon_state = spice.get_body_state("MOON", et0, "J2000", "EARTH")
    seleno = _oe2cart(a, e, i_deg, 0.0, aop_deg, 0.0, MU_MOON)
    state0 = np.concatenate(
        [
            seleno[:3] + moon_state[:3],
            seleno[3:6] + moon_state[3:6],
        ]
    )

    # 传播
    et_end = et0 + duration_sec
    et_grid = et0 + np.arange(0.0, duration_sec + 0.5 * output_step, output_step)
    out = fm.propagate(state0, (et0, et_end), t_eval=et_grid, max_steps=5_000_000)
    states = np.asarray(out["states"], dtype=float)
    # ForceModel._prepare_t_eval 会在 t_eval 末尾自动追加 t_span 终点
    # （duration 非整小时时 et_end 不在 et_grid 上，输出多 1 点）；截断到
    # et_grid 长度，保证状态与时间网格逐点对齐（否则位置数组比时间字段
    # 多 1 点，_build_ephemeris_table 的长度一致性断言会拒绝）。输出短于
    # et_grid 说明积分提前终止（max_steps 撞上），显式报错而非静默出短星历。
    if len(states) < len(et_grid):
        raise DesignNotConvergedError(
            "ELFO 传播输出点数少于时间网格（积分提前终止）",
            cause=FailureCause.INTEGRATION_FAILED,
        )
    states = states[: len(et_grid)]
    times = et_grid[: len(states)]

    # 月心根数提取与漂移统计
    moon_oe = _extract_moon_centric_elements(times, states, spice)
    drift = _compute_drift(moon_oe, times=times, output_step_sec=output_step)

    # 星历表（ELFO 仍输出地心惯性系 + 会合系，格式与 CR3BP 一致）
    system = earth_moon_system()
    syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
    ephemeris = _build_ephemeris_table(spice, syn_j2000, et0, et_grid, states)

    return OrbitDesignResult(
        orbit_type="ELFO",
        epoch_utc=epoch_iso,
        duration_day=duration_sec / SECONDS_PER_DAY,
        output_step_sec=output_step,
        initial_state=state0,
        ephemeris=ephemeris,
        cr3bp_orbit=None,
        cr3bp_jacobi=float("nan"),
        correction=None,
        force_config=force_config,
        stages=(
            StageRecord("initial_guess", applicable=False, executed=False, result_status=None),
            StageRecord(
                "ephemeris_correction", applicable=False, executed=False, result_status=None
            ),
            StageRecord(
                "propagation",
                applicable=True,
                executed=True,
                result_status=ConvergenceState.CONVERGED,
            ),
        ),
        drift_e=drift["drift_e"],
        drift_aop_deg=drift["drift_aop_deg"],
        drift_rp_km=drift["drift_rp_km"],
        secular_aop_rate_deg_per_year=drift["secular_aop_rate_deg_per_year"],
        moon_centric_elements=moon_oe,
    )


def design_orbit(
    request: DesignOrbitRequest,
    *,
    spice: SPICEManager | None = None,
    kernel_dir: str | None = None,
    verbose: bool = False,
) -> OrbitDesignResult:
    """端到端设计标称轨道（DRO/DPO/NRHO/Halo/Lissajous/L4/L5/Axial/.../ELFO）。

    通过 ``request.orbit_type`` 在内部分派管线：

    - **CR3BP 类型** （DRO/NRHO/Halo/Lissajous/…）：CR3BP 初猜 → 星历修正
      （多重打靶）→ 高精度长期预报。
    - **ELFO**：经典开普勒根数构造初值 → 全摄动传播 → 月心根数漂移分析。

    Args:
        request: ``DesignOrbitRequest``，经 model_validator 校验并填充默认值。
        spice: 已加载内核的 ``SPICEManager``；缺省自动创建并加载。
        kernel_dir: SPICE 内核目录。
        verbose: 修正过程显示进度条。

    Returns:
        ``OrbitDesignResult`` （标称星历 + 收敛/漂移信息）。

    Raises:
        ValueError: 形状参数/任务参数超界。
        NotImplementedError: 摄动开关含 ECOM 光压或耦合项（#253）。
        DesignNotConvergedError: 星历修正未收敛。
    """
    sel = request.orbit_type.upper()

    if spice is None:
        spice = SPICEManager()
        load_design_kernels(spice, kernel_dir)
    kernel_dir = kernel_dir or default_kernel_dir()

    # --- ELFO 管线 ---
    if sel == "ELFO":
        return _design_elfo(request, spice, kernel_dir, verbose)

    # --- CR3BP 管线 ---
    # model_validator 已按 orbit_type 填充默认值，此处构建 params dict
    _params_attrs = (
        "amplitude",
        "phase",
        "collinear_point",
        "north_south",
        "perilune_height",
        "amplitude_in",
        "amplitude_out",
        "phase_in",
        "phase_out",
    )
    params: dict[str, float | int] = {}
    for attr in _params_attrs:
        v = getattr(request, attr)
        if v is not None:
            params[attr] = v

    output_step = float(request.output_step)
    perturbation = request.perturbation
    earth_degree = request.earth_degree
    moon_degree = request.moon_degree
    dyb = request.dyb
    correction_method = request.correction_method
    correction_revolutions = request.correction_revolutions
    # Halo/NRHO 不稳定：two_level/standard 的"修正 1 圈 + 自由外推"必发散。
    # 统一走 segmented（全程分段打靶），产出
    # 不发散的标称参考轨道。圈间漂移是固有准周期特征，由 station_keeping
    # 处理（架构分工见 docs/architecture/architecture.md §总体定位）。
    if sel in ("HALO", "NRHO") and correction_method != "segmented":
        correction_method = "segmented"

    system = earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    cr3bp_orbit = _cr3bp_orbit_for(sel, params, dynamics)
    phase = float(params.get("phase", 0.0))
    jacobi = float(system.get_jacobi_constant(cr3bp_orbit.states[0]))

    # --- 相位 → 历元状态，采样 patch points，转 J2000 ---
    assert cr3bp_orbit.period is not None
    period = float(cr3bp_orbit.period)
    if sel in ("LISSAJOUS", "L4", "L5"):
        t0_syn = 0.0
    else:
        phase_offset = 0.5 if sel in ("DRO", "DPO") else 0.0
        t0_syn = ((phase + phase_offset) % 1.0) * period
    if t0_syn > 0.0:
        state0_syn = np.asarray(
            dynamics.propagate_orbit_state_at_time(cr3bp_orbit, t0_syn), dtype=float
        )
    else:
        state0_syn = np.asarray(cr3bp_orbit.states[0], dtype=float)

    if sel == "LISSAJOUS" and cr3bp_orbit.states.shape[0] > 1:
        if float(cr3bp_orbit.times[-1]) < (correction_revolutions - 1e-9) * period:
            cr3bp_orbit = design_lissajous(
                int(params["collinear_point"]),
                params["amplitude_in"],
                params["amplitude_out"],
                params["phase_in"],
                params["phase_out"],
                dynamics=dynamics,
                n_periods=correction_revolutions,
            )
        t_patch_syn, state_patch_syn = _sample_patch_points_from_trajectory(
            cr3bp_orbit, period, correction_revolutions
        )
    else:
        t_patch_syn, state_patch_syn = _sample_patch_points(
            dynamics,
            state0_syn,
            period,
            correction_revolutions,
            sampling=_patch_sampling_for(sel),
        )

    epoch_iso = _epoch_to_iso(request.epoch)
    et0 = spice.utc_to_et(epoch_iso)
    t_c = system.characteristic_time
    assert t_c is not None
    syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
    state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
        states_syn=state_patch_syn, t_syn_arr=t_patch_syn, et0=et0
    )
    t_patch_j2000 = et0 + t_patch_syn * t_c

    # --- 力模型构建（修正 + 长期预报共用） ---
    from ..forces import ForceModel
    from ..forces.force_mapping import perturbation_to_force_config

    if perturbation is None:
        perturbation = DEFAULT_DESIGN_PERTURBATION
    sw = dict(perturbation)
    bodies = ["EARTH", "MOON"]
    if sw["sun_body"]:
        bodies.append("SUN")
    if sw["planets"]:
        bodies += ["MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]

    full_system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
    full_system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    force_config = perturbation_to_force_config(
        perturbation, earth_degree=earth_degree, moon_degree=moon_degree, dyb=dyb
    )
    fm = ForceModel.from_config(force_config, full_system)
    fm.rtol = 1e-12
    fm.atol = 1e-12
    fm.max_step = 600.0

    assert request.duration is not None  # model_validator 已填默认值
    duration_sec = float(request.duration)
    duration_day = duration_sec / SECONDS_PER_DAY
    et_grid = et0 + np.arange(0.0, duration_sec + 0.5 * output_step, output_step)

    if correction_method == "segmented":
        # --- segmented：论文式分段打靶拼接（默认方法）---
        # 整条 CR3BP 多圈 tile → 逐段独立转星历 → 远月点分层合并 → 逐段
        # 积分填满 et_grid。不再单点自由外推整个 duration（旧方法发散根因）。
        from ..results import EphemerisCorrectionResult

        # 收集 Rust force 序列。跳过 RelativisticCorrection（与
        # ForceModel._STM_UNSUPPORTED_TYPES 对齐）：相对论修正的 STM
        # 雅可比尚未实现（compiled.rs `_ => Err`），放进打靶并行区无法
        # 积分 STM。注：sxform/spkezr 已走星历缓存（#268），若未来补
        # 雅可比，需同步注册 body/sxform 缓存键方可并入打靶。
        forces_py = []
        for entry in fm.list_forces():
            if not entry.enabled:
                continue
            if type(entry.force).__name__ == "RelativisticCorrection":
                continue
            spec = entry.force.to_rust_spec(full_system)
            if spec is not None:
                forces_py.append(spec)
        # 覆盖整条 duration 所需圈数（ceil 保证 et_grid 尾部有数据）
        n_rev = max(1, math.ceil(duration_sec / (period * t_c)))
        if sel == "LISSAJOUS" and cr3bp_orbit.states.shape[0] > 1:
            # 准周期 Lissajous：重建覆盖 n_rev 圈的有界轨迹再插值采样
            if float(cr3bp_orbit.times[-1]) < (n_rev - 1e-9) * period:
                cr3bp_orbit = design_lissajous(
                    int(params["collinear_point"]),
                    params["amplitude_in"],
                    params["amplitude_out"],
                    params["phase_in"],
                    params["phase_out"],
                    dynamics=dynamics,
                    n_periods=n_rev,
                )
            t_patch_syn_n, state_patch_syn_n = _sample_patch_points_from_trajectory(
                cr3bp_orbit, period, n_rev
            )
        else:
            # 按 n_rev 圈重采样整条 tile（初猜）。采样策略按族覆盖
            # （见 _patch_sampling_for）：Halo 近月点加密，NRHO 与其余等时间（#473）。
            t_patch_syn_n, state_patch_syn_n = _sample_patch_points(
                dynamics,
                state0_syn,
                period,
                n_rev,
                sampling=_patch_sampling_for(sel),
            )
        t_patch_j2000_n = et0 + t_patch_syn_n * t_c
        state_patch_j2000_n = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn_n, t_syn_arr=t_patch_syn_n, et0=et0
        )
        per_rev = len(t_patch_syn_n) // n_rev  # 每圈节点数（族策略下可能非 8）
        # 预采样星历缓存：打靶 + 逐段积分全程查三次样条表，不逐次调 cspice。
        # 这是 2 年 50 圈 × 每圈 8 节点 × 50 迭代打靶能跑完的关键（否则
        # 每步每力跨界查 SPICE，量级不可接受，且并发下触发 DAFFRNOTFOUND）。
        # 帧对覆盖 GravityField 的 body-fixed→J2000（地月非球形需要）。
        # 潮汐扰动体对：effective_coefficients 查 (perturber, central_body)
        # 在 J2000 的位置 + (input_frame, "J2000") 帧旋转，合成 body-fixed
        # 位置。注册扰动体对确保 tide=1 下全程走缓存、零 cspice FFI。
        # try/finally：缓存是进程级单例，用完必须清除避免污染后续调用。
        # 扰动体→中心天体对（与 Rust perturbers_for_body 一致）：
        #   EARTH → [SUN, MOON]，MOON → [EARTH]。
        _perturber_map = {"EARTH": ["SUN", "MOON"], "MOON": ["EARTH"]}
        _perturber_names = {p for b in bodies for p in _perturber_map.get(b.upper(), [])}
        # 合并：原 bodies + 扰动体中未包含的天体（如 SUN），
        # enable_ephem_cache 自动注册 (name, observer) + (name, SSB) + NAIF-ID 变体。
        _cache_bodies = list(dict.fromkeys([*bodies, *_perturber_names]))
        spice.enable_ephem_cache(
            _cache_bodies,
            float(et0) - float(period) * float(t_c),
            float(max(et_grid[-1], t_patch_j2000_n[-1])) + float(period) * float(t_c),
            dt=3600.0,
            observer="EARTH",
            frame_pairs=[("ITRF93", "J2000"), ("MOON_PA", "J2000")],
        )
        try:
            # 第 1 步段长（每组圈数）。Halo 与稳定轨道用多圈/段（上限 3）：
            # 长段节点密、段内约束强，各段修到正确星历弧（对齐朱彦伟 2026）。
            # NRHO 单独 1 圈/段（#473）：默认相位 0.5、约 1 个月弧上
            # revs_per_group=3 合并层残差可卡在约 10² km；1 圈/段与等时间
            # 采样组合下 GUI 默认量级收敛。配合下方 var_time 固定时刻族
            # （_FIXED_TIME_ORBIT_TYPES，含 Halo/NRHO、拟周期族与 Axial）。
            #
            # 对照论文的三项差异评估（issue #400 需求②）：论文每段 9 圈（对应
            # 972 圈/15 年量级的 12→3→3 层级拼接），合并层节点稀疏化为每圈 1 个
            # 远月点。Halo 3 圈/段 + 全节点合并已覆盖 180 天站保基准；年量级
            # 若段数过多、全节点合并矩阵病态放大，再评估 9 圈/段与远月点稀疏化。
            if sel == "NRHO":
                revs_per_group = 1
            elif sel == "HALO":
                revs_per_group = min(n_rev, 3)
            else:
                revs_per_group = max(1, min(3, n_rev))
            t_patch_long, s_patch_long, max_residual = _design_apolune_segmented(
                forces_py,
                "EARTH",
                t_patch_j2000_n,
                state_patch_j2000_n,
                revs_per_group,
                per_rev,
                max_iter=50,
                tolerance=CORRECTION_TOL_KM,
                var_time=sel not in _FIXED_TIME_ORBIT_TYPES,
                verbose=verbose,
            )

            # 逐段积分填满 et_grid（下沉 Rust propagate_segments_py 并发积分，
            # #400 性能修复）：每段从修正后节点初值积分到下一节点。中间段
            # mask 用右开区间（seam 整点只归后段），相邻段 t_eval 无重叠；
            # 最后一段闭区间、t_span 终点延伸到 et_grid 尾部——打靶节点采样
            # 不含圈终点（endpoint=False），et_grid 尾部可能超出最后节点，
            # 右开会把尾部点排除在段外，星历缺尾段数据（长度断言报错）。
            # Rust 侧输出逐点对应 t_eval（不追加段终点），无需截断——此前
            # ForceModel._prepare_t_eval 追加 tf 使输出多 1 点（段终点状态，
            # 不在 et_grid 上），截断丢掉才能保证 states_dense 与 et_grid
            # 严格对齐（位置-时间错位回归 #398）。
            seg_t0_list: list[float] = []
            seg_t1_list: list[float] = []
            seg_states_list: list[np.ndarray] = []
            t_eval_list: list[np.ndarray] = []
            for i in range(len(t_patch_long) - 1):
                seg_t0, seg_t1 = t_patch_long[i], t_patch_long[i + 1]
                is_last = i == len(t_patch_long) - 2
                if is_last:
                    mask = et_grid >= seg_t0 - 1e-6
                    seg_end = et_grid[-1]
                else:
                    mask = (et_grid >= seg_t0 - 1e-6) & (et_grid < seg_t1 - 1e-6)
                    seg_end = seg_t1
                t_eval_seg = et_grid[mask]
                if len(t_eval_seg) == 0:
                    continue
                seg_t0_list.append(float(seg_t0))
                seg_t1_list.append(float(seg_end))
                seg_states_list.append(s_patch_long[i])
                t_eval_list.append(t_eval_seg)
            if not seg_t0_list:
                raise DesignNotConvergedError(
                    "segmented 拼接未生成任何星历点",
                    cause=FailureCause.INTEGRATION_FAILED,
                )
            from e2m2e.integrators import propagate_segments_py

            states_dense = np.concatenate(
                [
                    np.asarray(seg_states, dtype=float)
                    for seg_states in propagate_segments_py(
                        "EARTH",
                        forces_py,
                        seg_t0_list,
                        seg_t1_list,
                        [list(map(float, s)) for s in seg_states_list],
                        [list(map(float, t)) for t in t_eval_list],
                        rtol=fm.rtol,
                    )
                ],
                axis=0,
            )
            # 右开区间下段间无重叠，无需去重；去重反而会掩蔽 mask 回归（若改回
            # 闭区间，seam 重复点被删、长度恢复一致，长度断言失效）。
        finally:
            spice.disable_ephem_cache()

        correction = EphemerisCorrectionResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="收敛",
            iterations=0,
            max_residual=max_residual,
            residual_history=[],
            t_patch=t_patch_long,
            state_patch=s_patch_long,
        )
        ephemeris = _build_ephemeris_table(spice, syn_j2000, et0, et_grid, states_dense)

        return OrbitDesignResult(
            orbit_type=sel,
            epoch_utc=epoch_iso,
            duration_day=duration_day,
            output_step_sec=float(output_step),
            initial_state=np.asarray(s_patch_long[0], dtype=float),
            ephemeris=ephemeris,
            cr3bp_orbit=cr3bp_orbit,
            cr3bp_jacobi=jacobi,
            correction=correction,
            force_config=force_config,
            stages=_design_stages(),
        )

    # --- 稳定轨道路径（DRO 等）：Rust 多重打靶（速度加权）+ 长期预报 ---
    # Halo/NRHO 已在上方重定向到 segmented（自由外推对不稳定轨道必发散）。
    # 旧 two_level（Python）残差向量把位置 km 与速度 km/s 混在一起，cislunar 下
    # 位置项单边主导，Level 2 速度连续化不跑，修正产出是位置连续但速度跳变
    # ~50 m/s 的断弧——自由外推大幅 DRO 一两个月发散到 20 万 km（#324）。
    # 改走 Rust 打靶 + vel_weight（=pos_tol/vel_tol）：速度项加权后在容差尺度
    # 与位置可比，LM 真正压速度连续到 ≤0.01 m/s，修正解落在准周期轨道上，
    # 稳定轨道自由外推有界。同时全程预制星历表（cspice 缓存），不再逐步 FFI。
    # --- 稳定轨道路径（DRO 等）：Rust 多重打靶（速度加权）+ 长期预报 ---
    if correction_method not in ("two_level", "standard", "rust"):
        raise ValueError(
            "correction_method 需为 segmented / two_level / standard / rust，"
            f"当前 {correction_method!r}"
        )
    from e2m2e.integrators import multiple_shooting_correct_py as _msc

    from ..results import EphemerisCorrectionResult

    forces_py = []
    for entry in fm.list_forces():
        if not entry.enabled:
            continue
        if type(entry.force).__name__ == "RelativisticCorrection":
            continue
        spec = entry.force.to_rust_spec(full_system)
        if spec is not None:
            forces_py.append(spec)
    _perturber_map = {"EARTH": ["SUN", "MOON"], "MOON": ["EARTH"]}
    _perturber_names = {p for b in bodies for p in _perturber_map.get(b.upper(), [])}
    _cache_bodies = list(dict.fromkeys([*bodies, *_perturber_names]))
    # var_time 打靶会把节点时间当自由变量，迭代中可把节点移到 patch 区间之外
    # （实测可前移 ~20h）。cache 上下界各留 1 周期余量，避免越界报错。
    _cache_margin = float(period) * float(t_c)
    spice.enable_ephem_cache(
        _cache_bodies,
        float(min(et0, t_patch_j2000[0])) - _cache_margin,
        float(max(et_grid[-1], t_patch_j2000[-1])) + _cache_margin,
        dt=3600.0,
        observer="EARTH",
        frame_pairs=[("ITRF93", "J2000"), ("MOON_PA", "J2000")],
    )
    try:
        # 不稳定轨道（Halo/NRHO，STM 谱半径 ~1e7/圈）单弧打靶每轮线搜索代价高、
        # 收敛慢；正路是延拓计算 + 缓存初值复用（独立工作）。在此之前给低 max_iter
        # 使其快速判定不收敛（抛 DesignNotConvergedError 供上层 skip），避免长时挂起。
        _ms_max_iter = 25 if sel in ("HALO", "NRHO") else 80
        result = _msc(
            forces_py,
            "EARTH",
            list(t_patch_j2000),
            [list(map(float, x)) for x in state_patch_j2000],
            var_time=sel not in _FIXED_TIME_ORBIT_TYPES,
            fix_first_node=False,
            fixed_node_mask=None,
            max_iter=_ms_max_iter,
            tolerance=CORRECTION_TOL_KM,
            rtol=1e-10,
            vel_weight=CORRECTION_VEL_WEIGHT,
        )
        if result.status is not ConvergenceState.CONVERGED:
            raise DesignNotConvergedError(
                f"{sel} 星历修正（Rust 多重打靶）未收敛：迭代 {result.iterations} 次，"
                f"位置残差 {result.position_residual:.3e} km，"
                f"速度残差 {result.velocity_residual:.3e} km/s",
                status=result.status,
                cause=result.cause,
            )
        t0_corr = float(result.t_patch[0])
        out = fm.propagate(
            np.asarray(result.state_patch[0], dtype=float),
            (min(t0_corr, et0), float(et_grid[-1])),
            t_eval=et_grid,
            max_steps=2_000_000,
        )
    finally:
        spice.disable_ephem_cache()

    correction = EphemerisCorrectionResult(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="收敛",
        iterations=int(result.iterations),
        max_residual=float(result.max_residual),
        residual_history=[float(x) for x in result.residual_history],
        t_patch=np.asarray(result.t_patch, dtype=float),
        state_patch=np.asarray(result.state_patch, dtype=float),
        velocity_residual=float(result.velocity_residual),
        velocity_residual_history=[float(result.velocity_residual)],
    )
    ephemeris = _build_ephemeris_table(
        spice, syn_j2000, et0, et_grid, np.asarray(out["states"], dtype=float)
    )

    return OrbitDesignResult(
        orbit_type=sel,
        epoch_utc=epoch_iso,
        duration_day=duration_day,
        output_step_sec=float(output_step),
        initial_state=np.asarray(result.state_patch[0], dtype=float),
        ephemeris=ephemeris,
        cr3bp_orbit=cr3bp_orbit,
        cr3bp_jacobi=jacobi,
        correction=correction,
        force_config=force_config,
    )
