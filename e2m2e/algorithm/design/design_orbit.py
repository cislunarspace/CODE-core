"""任务轨道设计入口（DFH 功能码 1 对齐层）。

链路（对齐 ``docs/plans/dfh-parity-prd.md`` FR1）：

1. CR3BP 初猜：按 DFH 形状参数生成周期轨道（``cr3bp_orbits``）；
2. 星历修正：周期轨道采样 patch points → synodic→J2000 转换 →
   星历 N 体模型下多重打靶收敛（``algorithms.ephemeris_correction``
   注册表）；
3. 标称星历：以修正后首节点状态为初值，在
   ``io.dfh_perturbation_to_force_config`` 映射出的高精度力模型下
   长期预报，输出 DFH 同格式星历（``io.EphemerisTable``）。

参数语义对齐 MATLAB ``design_orbit.m`` 与 inputs-dac.txt 设计块：
DRO 振幅+初始相位；Halo 共线点编号+带符号面外振幅+初始相位；
NRHO 共线点编号+北/南+近月点高度+初始相位。初始相位为周期份额
（0~1），历元时刻的状态 = 周期轨道参考状态沿轨道推进 ``phase × T``；
相位零点按 DFH 黄金样本标定——Halo/NRHO 在 y=0 穿越点，DRO 在远侧
x 轴穿越点（e2m2e 的 DRO 参考状态为近侧穿越点，内部偏移半周期）。
DRO 振幅取一个周期内距月距离最小/最大值的均值（同按黄金样本标定）。

已知系统差（与 tests/dfh 回归测试共用同一套假设）：

- DFH 输出 GCRS，e2m2e 在 ICRF（J2000）下传播，frame bias（~23 mas）
  在月距量级约 0.04 km，计入对比容差；
- 维持时间按 1 年 = 365.25 天折算；
- NRHO 近月点高度起算面取月球平均半径 1737.4 km。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from e2m2e.io import (
    DEFAULT_PERTURBATION,
    EphemerisTable,
    dfh_perturbation_to_force_config,
)

from ...data.kernels._spice_loader import get_spiceypy
from ...data.kernels.manager import SPICEManager
from ...data.types.orbit import Orbit
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from ..coordinate.synodic_j2000 import SynodicJ2000System
from ..dynamics import CR3BP_Dynamics, EphemerisDynamics, EphemerisSystem
from ..ephemeris_correction import (
    EphemerisCorrectionResult,
    correct_ephemeris_patch_points,
)
from ..family.cr3bp_orbits import (
    design_dro,
    design_halo,
    design_lissajous,
    design_nrho,
    design_triangular,
    earth_moon_system,
)
from ..forces import ForceModel
from ..solver.multiple_shooting import sample_patch_points_perilune_clustered

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

#: 维持时间年→天折算
DAYS_PER_YEAR = 365.25

#: 星历修正的默认收敛容差（km，6 维状态 max 范数：位置 km + 速度 km/s）。
#:
#: 取 1e-4 而非 1e-6 的依据：紧凑近月 DRO（如 amplitude=10000，周期~1 天、
#: 月距~1 万 km）因月球引力梯度强，CR3BP 几何与真实星历 N 体动力学存在
#: ~5.7e-5 km 的物理收敛底线，多重打靶无论收紧积分器或增加节点都突破不了
#: （实测见 design_orbit 调研）；远月 3:1 DRO 无此底线可达 1e-6（见
#: tests/algorithms/test_dro_ephemeris_correction.py 的多点用例亦用 1e-4）。
#: 1e-4 相对底线留约 1.7 倍裕度，兼顾紧凑与宽松两类轨道。
CORRECTION_TOL_KM = 1e-4

#: 每圈 patch 节点数（均匀采样）；NRHO 用近月点加密采样替代
_POINTS_PER_REV = 8

#: body-fixed 帧（ITRF93 / MOON_PA）所需内核文件名，与 tests/kernel_helpers.py 一致
_BODY_FIXED_KERNELS = [
    "earth_latest_high_prec.bpc",
    "pck00010.tpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
]

#: de440s 只有行星质心段，照 tests/dfh 的做法把行星名注册到质心 ID
_BODY_ID_ALIASES = [
    ("MERCURY", 1),
    ("VENUS", 2),
    ("EARTH", 399),
    ("MARS", 4),
    ("JUPITER", 5),
    ("SATURN", 6),
    ("URANUS", 7),
    ("NEPTUNE", 8),
    ("MOON", 301),
    ("SUN", 10),
]


class DesignNotConvergedError(RuntimeError):
    """星历修正未收敛。"""


@dataclass
class OrbitDesignResult:
    """任务轨道设计结果。

    Attributes:
        orbit_type: 轨道类型（``"DRO"`` / ``"HALO"`` / ``"NRHO"``）。
        epoch_utc: 起始历元 UTC（ISO 字符串）。
        duration_day: 维持时间（天）。
        output_step_sec: 星历输出间隔（秒）。
        initial_state: 历元时刻惯性系状态（km, km/s），星历修正后首节点。
        ephemeris: 标称星历（DFH 同格式容器：UTC + GCRS 位置 km /
            速度 m/s + 地月会合系无量纲位置）。
        cr3bp_orbit: CR3BP 周期轨道（参考相位，无量纲）。
        cr3bp_jacobi: CR3BP 周期轨道的 Jacobi 常数。
        correction: 星历修正结果（收敛标志、迭代次数、残差历史、修正后
            patch points）。
        force_config: 标称预报使用的力模型配置字典。
    """

    orbit_type: str
    epoch_utc: str
    duration_day: float
    output_step_sec: float
    initial_state: np.ndarray
    ephemeris: EphemerisTable
    cr3bp_orbit: Orbit
    cr3bp_jacobi: float
    correction: EphemerisCorrectionResult
    force_config: dict[str, Any]

    def write_ephemeris(self, path: str | Path) -> None:
        """按 DFH EPHEMERIDES 格式写出标称星历。"""
        from e2m2e.io import write_ephemeris

        write_ephemeris(self.ephemeris, path)


def default_kernel_dir() -> str:
    """仓库自带 SPICE 内核目录（``kernels/``）。"""
    return str(Path(__file__).resolve().parent.parent.parent / "kernels")


def load_design_kernels(spice: SPICEManager, kernel_dir: str | None = None) -> list[str]:
    """加载设计链路所需内核：行星历 + body-fixed 帧内核，注册行星名。

    返回实际加载的内核路径列表（调用方管理卸载）。
    """
    kernel_dir = kernel_dir or default_kernel_dir()
    for name, naif_id in _BODY_ID_ALIASES:
        get_spiceypy().boddef(name, naif_id)
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
        # 说明文档 inputs-dac.txt 第 33 行：初始相位取值 0.01~0.99，
        # 近月点高度较低时初始相位需明显大于 0
        if not 0.01 <= phase <= 0.99:
            raise ValueError(f"NRHO phase 应在 0.01~0.99 之间，实际为 {phase}")
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

    raise ValueError(
        f"orbit_type 必须为 DRO/NRHO/Halo/Lissajous/L4/L5，当前 {sel!r}"
    )


def _cr3bp_orbit_for(sel: str, params: dict[str, float | int], dynamics: CR3BP_Dynamics) -> Orbit:
    """按规范化形状参数生成 CR3BP 周期轨道。"""
    if sel == "DRO":
        return design_dro(params["amplitude"], dynamics=dynamics)
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


def _sample_patch_points(
    dynamics: CR3BP_Dynamics,
    state0: np.ndarray,
    period: float,
    n_revolutions: int,
    *,
    perilune_clustered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """从历元状态出发，在 ``n_revolutions`` 圈上采样 patch points（synodic）。

    均匀采样每圈 ``_POINTS_PER_REV`` 点；NRHO 改用近月点加密采样
    （近月点速度大、STM 条件数高，等间隔采样欠约束，见
    ``algorithms.multiple_shooting.sample_patch_points_perilune_clustered``）。
    """
    dense = _dense_orbit(dynamics, state0, period)
    if perilune_clustered:
        t_rel, states = sample_patch_points_perilune_clustered(dense, dynamics)
    else:
        t_rel = np.linspace(0.0, period, _POINTS_PER_REV, endpoint=False)
        states = np.empty((len(t_rel), 6))
        for i in range(6):
            states[:, i] = np.interp(t_rel, dense.times, dense.states[:, i])

    t_patch = np.concatenate([t_rel + k * period for k in range(n_revolutions)])
    state_patch = np.tile(states, (n_revolutions, 1))
    return t_patch, state_patch


def _build_ephemeris_table(
    spice: SPICEManager,
    syn_j2000: SynodicJ2000System,
    et0: float,
    et_grid: np.ndarray,
    states: np.ndarray,
) -> EphemerisTable:
    """把传播结果组装成 DFH 同格式星历表（UTC + GCRS + 地月会合系）。

    DFH 的地月会合系为地心归一（月球在 +x 单位距离处；依据：DFH 设计
    黄金样本中 DRO 轨道在该约定下关于 x=1 近似对称）；内部转换器输出
    质心归一（月球在 1-mu），x 分量加 mu 平移对齐。
    """
    t_c = syn_j2000.cr3bp_system.characteristic_time
    t_syn = (et_grid - et0) / t_c
    synodic = syn_j2000.batch_j2000_to_synodic(states, t_syn, et0)[:, :3]
    synodic[:, 0] += syn_j2000.cr3bp_system.mu

    n = len(et_grid)
    year = np.empty(n, dtype=int)
    month = np.empty(n, dtype=int)
    day = np.empty(n, dtype=int)
    hour = np.empty(n, dtype=int)
    minute = np.empty(n, dtype=int)
    second = np.empty(n, dtype=float)
    for k in range(n):
        dt = datetime.fromisoformat(spice.et_to_utc(float(et_grid[k])))
        year[k], month[k], day[k] = dt.year, dt.month, dt.day
        hour[k], minute[k] = dt.hour, dt.minute
        second[k] = dt.second + dt.microsecond / 1e6

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


def design_orbit(
    orbit_type: str,
    *,
    amplitude: float | None = None,
    phase: float | None = None,
    collinear_point: int | None = None,
    north_south: int | None = None,
    perilune_height: float | None = None,
    amplitude_in: float | None = None,
    amplitude_out: float | None = None,
    phase_in: float | None = None,
    phase_out: float | None = None,
    epoch: Sequence[float] | str = (2024, 1, 1, 0, 0, 0.0),
    duration: float = 1.0,
    output_step: float = 3600.0,
    perturbation: dict[str, int] | None = None,
    dyb: Sequence[float] | None = None,
    earth_degree: int = 10,
    moon_degree: int = 10,
    spice: SPICEManager | None = None,
    kernel_dir: str | None = None,
    correction_method: str = "two_level",
    correction_revolutions: int = 1,
    correction_velocity_tolerance: float = 0.1,
    verbose: bool = False,
) -> OrbitDesignResult:
    """端到端设计六类标称轨道（DRO/NRHO/Halo/Lissajous/L4/L5）。

    Args:
        orbit_type: ``"DRO"`` / ``"NRHO"`` / ``"Halo"`` / ``"Lissajous"`` /
            ``"L4"`` / ``"L5"``。
        amplitude: 振幅（km）。DRO 1737~110000，默认 10000；Halo 面外振幅
            ±73000（正北负南），默认 30000；NRHO 不用。
        phase: 初始相位（周期份额）。DRO/Halo 取值 0~1，默认 0.5001/0；
            NRHO 取值 0.01~0.99，默认 0.5。
        collinear_point: 共线平动点编号 1/2（Halo/NRHO，默认 2）。
            Lissajous 取 1/2/3（默认 2）。
        north_south: 1=北 / 2=南（NRHO，默认 2）。
        perilune_height: 近月点高度（km，100~10000，NRHO，默认 5000）。
        amplitude_in / amplitude_out: 面内/面外振幅（km）。Lissajous
            L1/L2 各 ≤ 7600、L3 各 ≤ 100000（默认 2500/7500）；L4/L5 面内
            ≤ 10000、面外 ≤ 76000（默认 8000/6000）。
        phase_in / phase_out: 面内/面外初始相位（0~1）。Lissajous 默认
            0.01/0.55；L4/L5 默认 0/0。
        epoch: 起始历元 UTC，``[年, 月, 日, 时, 分, 秒]`` 或 ISO 字符串。
        duration: 维持时间（年，0 < d ≤ 20）。
        output_step: 星历输出间隔（秒）。
        perturbation: DFH 摄动开关字典（键与取值同
            ``io.inputs_dac.DEFAULT_PERTURBATION``），经
            ``io.dfh_perturbation_to_force_config`` 映射为力模型。缺省时用
            ``DEFAULT_DESIGN_PERTURBATION``：光压炮弹模型、关耦合项
            （ECOM 光压与耦合项属 #253，需要时显式开启会抛
            ``NotImplementedError``）。
        dyb: DYB 系数 9 分量（``dyb[0]`` = 等效面质比 m²/kg）。
        earth_degree / moon_degree: 引力场阶次。
        spice: 已加载内核的 ``SPICEManager``；缺省自动创建并加载
            ``kernel_dir``（默认仓库 ``kernels/``）下的内核。
        kernel_dir: SPICE 内核目录。
        correction_method: 星历修正方法（``"standard"`` / ``"two_level"`` /
            ``"homotopy"``，注册表分发）。默认 ``"two_level"``——端点固定 +
            回溯线搜索（Marchand-Howell-Wilson 两层法），对 Halo/NRHO 等不稳
            定轨道能收敛；``"standard"`` 全自由变量无步长控制，仅对稳定 DRO 可靠。
        correction_revolutions: 星历修正弧长（圈数）。
        correction_velocity_tolerance: 速度残差容差（km/s，仅 ``"two_level"``）。
            默认 0.1（100 mm/s）。two_level 的 Level 2（速度连续）对紧凑近月轨道
            （NRHO 近月点 STM 病态）存在 ~70-90 mm/s 的收敛底线，多轮迭代/加密
            取点/月球非球形均无法突破；位置连续性（亚毫米）才是 patch point 质量
            的主导因素，0.1 容留裕度使位置已亚毫米连续的解判通过。
        verbose: 修正过程显示进度条。

    Returns:
        ``OrbitDesignResult``（标称星历 + 收敛信息）。

    Raises:
        ValueError: 形状参数/任务参数超界。
        NotImplementedError: 摄动开关含 ECOM 光压或耦合项（#253）。
        DesignNotConvergedError: 星历修正未收敛。
    """
    sel = orbit_type.upper()
    if not 0.0 < float(duration) <= 20.0:
        raise ValueError(f"duration 应大于 0 且不超过 20 年，实际为 {duration}")
    if float(output_step) <= 0.0:
        raise ValueError(f"output_step 必须为正数，当前 {output_step}")
    if correction_revolutions < 1:
        raise ValueError(f"correction_revolutions 必须 ≥ 1，当前 {correction_revolutions}")

    # --- 1. CR3BP 初猜 ---
    params = _validate_params(
        sel,
        amplitude=amplitude,
        phase=phase,
        collinear_point=collinear_point,
        north_south=north_south,
        perilune_height=perilune_height,
        amplitude_in=amplitude_in,
        amplitude_out=amplitude_out,
        phase_in=phase_in,
        phase_out=phase_out,
    )

    if spice is None:
        spice = SPICEManager()
        load_design_kernels(spice, kernel_dir)
    kernel_dir = kernel_dir or default_kernel_dir()

    system = earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    cr3bp_orbit = _cr3bp_orbit_for(sel, params, dynamics)
    phase = float(params.get("phase", 0.0))
    jacobi = float(system.get_jacobi_constant(cr3bp_orbit.states[0]))

    # --- 2. 相位 → 历元状态，采样 patch points，转 J2000 ---
    period = float(cr3bp_orbit.period)
    # 相位零点约定：Halo/NRHO 的参考状态（y=0 穿越点）与 DFH 黄金样本的
    # phase=0 起点一致；DRO 的参考状态是近侧穿越点，而 DFH 相位零点在远侧
    # 穿越点（黄金样本标定：DFH phase=0.5001 的 DRO 首行恰在近侧穿越点，
    # 距月取最小值），差半个周期
    if sel in ("LISSAJOUS", "L4", "L5"):
        # 面内/面外相位已体现在初猜状态（t=0 即历元状态）
        t0_syn = 0.0
    else:
        phase_offset = 0.5 if sel == "DRO" else 0.0
        t0_syn = ((phase + phase_offset) % 1.0) * period
    if t0_syn > 0.0:
        state0_syn = np.asarray(
            dynamics.propagate_orbit_state_at_time(cr3bp_orbit, t0_syn), dtype=float
        )
    else:
        state0_syn = np.asarray(cr3bp_orbit.states[0], dtype=float)

    t_patch_syn, state_patch_syn = _sample_patch_points(
        dynamics,
        state0_syn,
        period,
        correction_revolutions,
        perilune_clustered=(sel == "NRHO"),
    )

    epoch_iso = _epoch_to_iso(epoch)
    et0 = spice.utc_to_et(epoch_iso)
    t_c = system.characteristic_time
    syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
    state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
        states_syn=state_patch_syn, t_syn_arr=t_patch_syn, et0=et0
    )
    t_patch_j2000 = et0 + t_patch_syn * t_c

    # --- 3. 星历修正（N 体模型多重打靶） ---
    # 默认光压用炮弹模型、关耦合项（ECOM/耦合属 #253，显式开启会报错）
    if perturbation is None:
        perturbation = DEFAULT_DESIGN_PERTURBATION
    sw = dict(perturbation)
    bodies = ["EARTH", "MOON"]
    if sw["sun_body"]:
        bodies.append("SUN")
    if sw["planets"]:
        bodies += ["MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]

    eph_system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
    eph_dynamics = EphemerisDynamics(system=eph_system)
    eph_dynamics.rtol = 1e-12
    eph_dynamics.atol = 1e-12
    eph_dynamics.max_step = 600.0

    correction = correct_ephemeris_patch_points(
        method=correction_method,
        dynamics=eph_dynamics,
        t_patch=t_patch_j2000,
        state_patch=state_patch_j2000,
        tolerance=CORRECTION_TOL_KM,
        max_iter=50,
        verbose=verbose,
        n_workers=1,
        kernel_dir=kernel_dir,
        velocity_tolerance=correction_velocity_tolerance,
        base_bodies=["EARTH", "MOON"] if correction_method == "homotopy" else None,
    )
    if not correction.converged:
        raise DesignNotConvergedError(
            f"{sel} 星历修正（{correction_method}）未收敛：迭代 {correction.iterations} 次，"
            f"最大残差 {correction.max_residual:.3e} km"
        )

    # --- 4. 高精度力模型长期预报 ---
    duration_day = float(duration) * DAYS_PER_YEAR
    duration_sec = duration_day * 86400.0
    et_grid = et0 + np.arange(0.0, duration_sec + 0.5 * float(output_step), float(output_step))

    full_system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
    full_system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )
    force_config = dfh_perturbation_to_force_config(
        perturbation, earth_degree=earth_degree, moon_degree=moon_degree, dyb=dyb
    )
    fm = ForceModel.from_config(force_config, full_system)
    fm.max_step = 600.0
    t0_corr = float(correction.t_patch[0])
    out = fm.propagate(
        correction.state_patch[0],
        (min(t0_corr, et0), float(et_grid[-1])),
        t_eval=et_grid,
        max_steps=2_000_000,
    )

    ephemeris = _build_ephemeris_table(
        spice, syn_j2000, et0, et_grid, np.asarray(out["states"], dtype=float)
    )

    return OrbitDesignResult(
        orbit_type=sel,
        epoch_utc=epoch_iso,
        duration_day=duration_day,
        output_step_sec=float(output_step),
        initial_state=np.asarray(correction.state_patch[0], dtype=float),
        ephemeris=ephemeris,
        cr3bp_orbit=cr3bp_orbit,
        cr3bp_jacobi=jacobi,
        correction=correction,
        force_config=force_config,
    )

