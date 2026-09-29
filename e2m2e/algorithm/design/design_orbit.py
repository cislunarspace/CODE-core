"""任务轨道设计入口。

链路（对齐 ``docs/plans/dfh-parity-prd.md`` FR1）：

1. CR3BP 初猜：按形状参数生成周期轨道（``family/orbits/``）；
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
相位零点约定：Halo/NRHO 在 y=0 穿越点，DRO 在远侧
x 轴穿越点（e2m2e 的 DRO 参考状态为近侧穿越点，内部偏移半周期）。
DRO 振幅取一个周期内距月距离最小/最大值的均值。

已知系统差：

- 参考输出 GCRS，e2m2e 在 ICRF（J2000）下传播，frame bias（~23 mas）
  在月距量级约 0.04 km，计入对比容差；
- 维持时间按 1 年 = 365.25 天折算；
- NRHO 近月点高度起算面取月球平均半径 1737.4 km。

星历修正的实验调参常量在 ``tuning.py``；拼接点采样与分段打靶封装在
``segmented.py``。
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from e2m2e.data.types.trajectory import EphemerisTable

from ...data.constants import SECONDS_PER_DAY
from ...data.kernels.manager import SPICEManager
from ...data.templates.design import SEGMENTED_CORRECTION_ORBIT_TYPES
from ...data.templates.perturbations import DEFAULT_PERTURBATION
from ...data.types.orbit import Orbit
from ...status import ConvergenceState, FailureCause, ResultStatus
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from ..coordinate.synodic_j2000 import SynodicJ2000System
from ..dynamics import CR3BP_Dynamics, EphemerisSystem
from ..family.orbits import (
    design_axial,
    design_dpo,
    design_dro,
    design_halo,
    design_horseshoe,
    design_lissajous,
    design_lpo,
    design_lyapunov,
    design_nrho,
    design_ro,
    design_spo,
    design_triangular,
    earth_moon_system,
)
from ..results import EphemerisCorrectionResult, StageRecord
from .segmented import (
    _design_apolune_segmented,
    _patch_sampling_for,
    _points_per_rev_for,
    _revs_per_group_for,
    _sample_patch_points,
    _sample_patch_points_from_trajectory,
)
from .tuning import _FIXED_TIME_ORBIT_TYPES, CORRECTION_TOL_KM, CORRECTION_VEL_WEIGHT

if TYPE_CHECKING:

    class _DesignRequest(Protocol):
        """design_orbit 请求的最小结构契约（镜像 api.DesignOrbitRequest）。

        算法层不 import api/（ADR 0012 硬边界①）；此处按实际消费字段
        声明协议，api.DesignOrbitRequest 结构性满足。新增消费字段须
        同步本协议。
        """

        orbit_type: str
        amplitude: float | None
        phase: float | None
        collinear_point: int | None
        north_south: int | None
        amplitude_in: float | None
        amplitude_out: float | None
        phase_in: float | None
        phase_out: float | None
        perilune_height: float | None
        resonance_p: int | None
        resonance_q: int | None
        inclination: float | None
        arg_of_pericenter: float | None
        semi_major_axis: float | None
        epoch: Any
        duration: float | None
        output_step: float
        perturbation: dict[str, int] | None
        dyb: list[float] | None
        earth_degree: int
        moon_degree: int
        correction_method: str
        correction_revolutions: int


__all__ = [
    "DesignNotConvergedError",
    "OrbitDesignResult",
    "default_kernel_dir",
    "design_orbit",
    "load_design_kernels",
]

#: 设计链路的默认摄动开关：光压用炮弹模型（``solar_radiation=1``）、关耦合项。
#: ECOM 光压与地球非球形×大天体耦合项未实现，显式开启时抛
#: ``NotImplementedError``；约定同 ``control_orbit._DEFAULT_CTRL_PERTURBATION``。
DEFAULT_DESIGN_PERTURBATION: dict[str, int] = {
    **DEFAULT_PERTURBATION,
    "solar_radiation": 1,
    "coupling": 0,
}

#: body-fixed 帧（ITRF93 / MOON_PA）所需内核文件名，与 tests/kernel_helpers.py 一致。
#: 预测 PCK 必须先于历史 PCK 加载：SPICE 对重叠覆盖段取后加载者，历史
#: 重构数据（高精度）因此在过去时段优先，未来时段由预测数据补齐
#: （历史文件覆盖有终点，超出即 FRAMEDATANOTFOUND）。
_BODY_FIXED_KERNELS = [
    "SPICEEarthPredictedKernel.bpc",
    "earth_latest_high_prec.bpc",
    "pck00010.tpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
]

#: 缺省行星历内核候选（按优先级）。None 口径下行为与历史一致。
_DEFAULT_EPHEMERIS_KERNELS = ["de440s.bsp", "de430.bsp"]


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
        correction_method: 实际执行的星历修正方法（``"segmented"`` /
            ``"two_level"`` 等）；ELFO 场景（无星历修正）为 None。
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
    correction_method: str | None = None
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


def load_design_kernels(
    spice: SPICEManager, kernel_dir: str | None = None, *, datum: str | None = None
) -> list[str]:
    """加载设计链路所需内核：行星历 + body-fixed 帧内核。

    行星名→质心/本体 NAIF ID 别名由 :meth:`SPICEManager.load_kernel` 首次
    调用时统一注册（双侧同步，见 ``data/kernels/registry.py`` 的
    ``_BODY_ID_ALIASES``）。

    Args:
        spice: 目标 SPICE 管理器。
        kernel_dir: 内核目录；缺省用仓库自带 ``kernels/``。
        datum: 行星历口径（``"DE421"``/``"DE440"``）。指定 ``"DE421"`` 时优先
            加载 de421.bsp，使第三体位置与 GM 同为 DE421 口径（ADR 0048）；
            **该内核缺失即报错，不静默降级**。None（默认）保持原有
            de440s > de430 选择顺序，行为不变。

    Returns:
        实际加载的内核路径列表（调用方管理卸载）。

    Raises:
        FileNotFoundError: 目录内无可用行星历内核；或显式请求的口径内核缺失。
    """
    kernel_dir = kernel_dir or default_kernel_dir()
    if datum is None:
        ephemeris_candidates = _DEFAULT_EPHEMERIS_KERNELS
    else:
        names = SPICEManager.datum_kernel_names(datum)
        if not names:
            raise ValueError(f"未知的星历基准（无偏好内核）: {datum}")
        # 显式口径：只用该口径内核（同一 datum 的多个内核等价，如 de440/de440s）；
        # 不再追加其它 DE 系列，避免「请求 DE421 却因文件缺失静默用 de440s」的谎报。
        ephemeris_candidates = list(names)
    loaded: list[str] = []
    for name in ephemeris_candidates:
        path = os.path.join(kernel_dir, name)
        if os.path.exists(path):
            spice.load_kernel(path)
            loaded.append(path)
            break
    else:
        wanted = " 或 ".join(ephemeris_candidates)
        raise FileNotFoundError(
            f"行星历内核不存在（{wanted}）: {kernel_dir}"
            "（内核随仓库 git-lfs 入库，缺失时跑 make kernels 从 kernels-v1 获取）"
        )
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
        # L1 设计域止于族折叠常量 26 908 km（#643 探测确认）；L2 放宽到 77 000 km
        limit = 26_908.0 if collinear_point == 1 else 77_000.0
        if not abs(amplitude) <= limit:
            raise ValueError(
                f"Halo L{collinear_point} amplitude 应在 -{limit:.0f}~{limit:.0f} km 之间，"
                f"实际为 {amplitude:.0f} km"
            )
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"Halo phase 应在 0~1 之间，实际为 {phase}")
        return {
            "collinear_point": collinear_point,
            "amplitude": amplitude,
            "phase": phase,
        }

    if sel == "LYAPUNOV":
        collinear_point = 2 if collinear_point is None else int(collinear_point)
        amplitude = 12000.0 if amplitude is None else float(amplitude)
        phase = 0.0 if phase is None else float(phase)
        if collinear_point not in (1, 2):
            raise ValueError(f"Lyapunov collinear_point 必须为 1 或 2，当前 {collinear_point}")
        if not 5000.0 <= amplitude <= 60000.0:
            raise ValueError(
                f"Lyapunov L{collinear_point} amplitude 应在 5000~60000 km 之间，"
                f"实际为 {amplitude:.0f} km"
            )
        if not 0.0 <= phase <= 1.0:
            raise ValueError(f"Lyapunov phase 应在 0~1 之间，实际为 {phase}")
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
        if not 100.0 <= perilune_height <= 40000.0:
            raise ValueError(
                f"NRHO perilune_height 应在 100~40000 km 之间，实际为 {perilune_height:.0f} km"
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
        f"orbit_type 必须为 DRO/DPO/NRHO/Halo/Lyapunov/Lissajous/L4/L5/Axial/L4_SPO/L5_SPO"
        f"/L4_LPO/L5_LPO/L4_HORSESHOE/L5_HORSESHOE，当前 {sel!r}"
    )


def _cr3bp_orbit_for(sel: str, params: dict[str, float | int], dynamics: CR3BP_Dynamics) -> Orbit:
    """按规范化形状参数生成 CR3BP 周期轨道。"""
    if sel == "DRO":
        return design_dro(params["amplitude"], dynamics=dynamics)
    if sel == "DPO":
        return design_dpo(params["amplitude"], dynamics=dynamics)
    if sel == "RO":
        return design_ro(
            int(params["resonance_p"]),
            int(params["resonance_q"]),
            params.get("amplitude"),
            dynamics=dynamics,
        )
    if sel == "HALO":
        return design_halo(int(params["collinear_point"]), params["amplitude"], dynamics=dynamics)
    if sel == "LYAPUNOV":
        return design_lyapunov(
            int(params["collinear_point"]), params["amplitude"], dynamics=dynamics
        )
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


def _build_ephemeris_table(
    spice: SPICEManager,
    syn_j2000: SynodicJ2000System,
    et0: float,
    et_grid: np.ndarray,
    states: np.ndarray,
) -> EphemerisTable:
    """把传播结果组装成文本格式星历表（UTC + GCRS + 地月会合系）。

    地月会合系为地心归一（月球在 +x 单位距离处；DRO 轨道在该约定下
    关于 x=1 近似对称）；内部转换器输出质心归一（月球在
    1-mu），x 分量加 mu 平移对齐。
    """
    t_c = syn_j2000.cr3bp_system.characteristic_time
    assert t_c is not None
    # 位置/速度与时间网格必须逐点对齐：若把 propagate 自动追加的段终点
    # tf 拼进 states（位置数组比时间网格多出段数个点），
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


def _design_elfo(
    request: _DesignRequest,
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
        correction_method=None,
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
    request: _DesignRequest,
    *,
    spice: SPICEManager | None = None,
    kernel_dir: str | None = None,
    verbose: bool = False,
) -> OrbitDesignResult:
    """端到端设计标称轨道（DRO/DPO/NRHO/Halo/Lyapunov/Lissajous/L4/L5/Axial/RO/.../ELFO）。

    通过 ``request.orbit_type`` 在内部分派管线：

    - **CR3BP 类型** （DRO/NRHO/Halo/Lissajous/…）：CR3BP 初猜 → 星历修正
      （多重打靶）→ 高精度长期预报。
    - **ELFO**：经典开普勒根数构造初值 → 全摄动传播 → 月心根数漂移分析。

    Args:
        request: 请求对象——``api.DesignOrbitRequest``（经 model_validator
            校验并填充默认值）或其他结构性满足 ``_DesignRequest`` 协议的对象。
        spice: 已加载内核的 ``SPICEManager``；缺省自动创建并加载。
        kernel_dir: SPICE 内核目录。
        verbose: 修正过程显示进度条。

    Returns:
        ``OrbitDesignResult`` （标称星历 + 收敛/漂移信息）。

    Raises:
        ValueError: 形状参数/任务参数超界。
        NotImplementedError: 摄动开关含 ECOM 光压或耦合项。
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
        "resonance_p",
        "resonance_q",
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
    # 修正方法已由请求校验层按族规范化（DesignOrbitRequest）；
    # 此处只拦截绕过校验的请求并抛 ValueError，不静默改写。
    if sel in SEGMENTED_CORRECTION_ORBIT_TYPES and correction_method != "segmented":
        raise ValueError(
            f"{sel} 属不稳定轨道族，correction_method 必须为 'segmented'，"
            f"当前 {correction_method!r}（请求未经校验层规范化）"
        )

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
            points_per_rev=_points_per_rev_for(sel),
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
        # 积分填满 et_grid。单点自由外推整个 duration 会发散，不可采用。
        from ..results import EphemerisCorrectionResult

        # 收集 Rust force 序列。跳过 RelativisticCorrection（与
        # ForceModel._STM_UNSUPPORTED_TYPES 对齐）：相对论修正的 STM
        # 雅可比尚未实现（compiled.rs `_ => Err`），放进打靶并行区无法
        # 积分 STM。注：sxform/spkezr 已走星历缓存，若未来补
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
            # （见 _patch_sampling_for）：Halo 近月点加密；NRHO 与其余族等时间。
            # DPO 周期长且不稳定，使用 64 点/圈缩短单弧。
            t_patch_syn_n, state_patch_syn_n = _sample_patch_points(
                dynamics,
                state0_syn,
                period,
                n_rev,
                sampling=_patch_sampling_for(sel),
                points_per_rev=_points_per_rev_for(sel),
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
            # 第 1 步段长（每组圈数）按族选择：见 segmented._revs_per_group_for。
            revs_per_group = _revs_per_group_for(sel, n_rev)
            try:
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
            except DesignNotConvergedError:
                # NRHO n_rev=2 回退：9:2 贴月共振成员（近月高
                # ~1500 km）在 phase=0.5 下两段独立打靶后 seam 失配
                # ~10² km，合并层不收敛；改为单段 2 圈（无合并层）可收敛。
                # 反之 phase=0 端点单段不收敛、原路径可收敛，两种策略
                # 互补，按失败自动切换。仅限 n_rev=2（多圈单段长弧打靶
                # 另有收敛风险）。
                if sel != "NRHO" or n_rev != 2 or revs_per_group >= n_rev:
                    raise
                t_patch_long, s_patch_long, max_residual = _design_apolune_segmented(
                    forces_py,
                    "EARTH",
                    t_patch_j2000_n,
                    state_patch_j2000_n,
                    n_rev,
                    per_rev,
                    max_iter=50,
                    tolerance=CORRECTION_TOL_KM,
                    var_time=sel not in _FIXED_TIME_ORBIT_TYPES,
                    verbose=verbose,
                )

            # 逐段积分填满 et_grid（下沉 Rust propagate_segments_py 并发积分）：
            # 每段从修正后节点初值积分到下一节点。中间段
            # mask 用右开区间（seam 整点只归后段），相邻段 t_eval 无重叠；
            # 最后一段闭区间、t_span 终点延伸到 et_grid 尾部——打靶节点采样
            # 不含圈终点（endpoint=False），et_grid 尾部可能超出最后节点，
            # 右开会把尾部点排除在段外，星历缺尾段数据（长度断言报错）。
            # Rust 侧输出逐点对应 t_eval（不追加段终点），无需截断即可保证
            # states_dense 与 et_grid 严格对齐。
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
            correction_method=correction_method,
            force_config=force_config,
            stages=_design_stages(),
        )

    # --- 稳定轨道路径（DRO 等）：Rust 多重打靶（速度加权）+ 长期预报 ---
    # 不稳定族（HALO/NRHO/DPO）已由请求校验层规范化为 segmented，不会到达此路径。
    # Rust 打靶 + vel_weight（=pos_tol/vel_tol）：速度项加权后在容差尺度
    # 与位置可比，LM 真正压速度连续到 ≤0.01 m/s，修正解落在准周期轨道上，
    # 稳定轨道自由外推有界（不加权则位置项单边主导，产出速度跳变数十 m/s
    # 的断弧）。同时全程预制星历表（cspice 缓存）代替逐次 FFI。
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
        correction_method=correction_method,
        force_config=force_config,
    )
