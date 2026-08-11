"""功能码 2（任务轨道控制/轨道保持）对齐入口。

端到端轨道保持功能：输入标称轨道星历（FR1 ``design_orbit``
产物），按控制模式（特征点/目标点严格/目标点宽松，可选角动量管理）施加
脉冲控制，以三轨道结构（目标/真实/测量）仿真测定轨与控制误差，蒙特卡洛
批量评估，输出 SK_STATISTIC / MANEUVERS / 受控星历。算法以《控制方案.md》
（hybrid_auto 版）为准，规格见 ``docs/plans/dfh-parity-prd.md`` FR2。

参数与 MATLAB ``control_orbit.m`` 对齐（关键字参数 + dataclass 结果）。
角动量管理模式（control_mode 4-6）通过 ``engine_layout`` 参数激活，
对应 MATLAB ControlMode 4-6。两处能力边界差异（e2m2e 尚未实现，见
#253）：光压默认用炮弹模型（``solar_radiation=1``，MATLAB 默认 ECOM=2）；
耦合项默认关闭（``coupling=0``，MATLAB 默认开）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...data.constants import SECONDS_PER_DAY
from ...data.kernels.manager import SPICEManager
from ...data.templates import ConvergenceState, FailureCause
from ...data.templates.perturbations import DEFAULT_DYB, DEFAULT_PERTURBATION
from ...data.types import EphemerisTable, ManeuverTable, SKStatistic
from ...data.types.trajectory import read_ephemeris
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from ..design.design_orbit import default_kernel_dir, load_design_kernels
from ..dynamics import EphemerisSystem
from ..forces.force_mapping import perturbation_to_force_config
from ..results import ResultStatus, StageRecord
from .monte_carlo import MonteCarloResult, run_monte_carlo

if TYPE_CHECKING:
    from .momentum_management import EngineLayout

__all__ = ["ControlOrbitResult", "control_orbit"]

#: 受控星历输出文件名（按控制模式分文件，mode 4-6 沿用基础模式文件名）
_EPHEMERIS_NAMES = {1: "EPHEMERIDES_LOOSE", 2: "EPHEMERIDES_TIGHT", 3: "EPHEMERIDES_SPECIAL"}

#: e2m2e 能力边界内的默认摄动开关：球模型光压、关耦合项（MATLAB 默认
#: 分别为 ECOM 与开，e2m2e 未实现，#253）
_DEFAULT_CTRL_PERTURBATION: dict[str, int] = {
    **DEFAULT_PERTURBATION,
    "solar_radiation": 1,
    "coupling": 0,
}


@dataclass
class ControlOrbitResult:
    """轨道保持仿真结果（对齐 MATLAB ``control_orbit`` 输出结构）。

    Attributes:
        sk_statistic: SK_STATISTIC 表（3 列或 5 列，m/s；角动量管理时含姿态列）
        num_failed: 蒙特卡洛失败样本数
        maneuvers: MANEUVERS 机动序列（MJD(TDB) + Δv，m/s）
        controlled_ephemeris: 最后一次样本的受控真实轨道星历
            （None 表示所有样本均失败提前退出）
    """

    sk_statistic: SKStatistic
    num_failed: int
    maneuvers: ManeuverTable
    controlled_ephemeris: EphemerisTable | None
    raw: MonteCarloResult = field(repr=False)
    status: ConvergenceState = ConvergenceState.CONVERGED
    cause: FailureCause = FailureCause.NONE
    message: str = "任务完成"
    stages: tuple[StageRecord, ...] = ()

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)

    def write_outputs(self, out_dir: str | Path, mode: int = 1) -> list[Path]:
        """把三个输出文件写入目录（SK_STATISTIC/MANEUVERS/受控星历）。"""
        from ...data.types.maneuver import write_maneuvers
        from ...data.types.sk_statistic import write_sk_statistic
        from ...data.types.trajectory import write_ephemeris

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            write_sk_statistic(self.sk_statistic, out_dir / "SK_STATISTIC.TXT"),
            write_maneuvers(self.maneuvers, out_dir / "MANEUVERS.TXT"),
        ]
        if self.controlled_ephemeris is not None:
            # mode 4-6 沿用基础模式的文件名
            base = mode if mode <= 3 else mode - 3
            name = _EPHEMERIS_NAMES.get(base, f"EPHEMERIDES_MODE{base}")
            paths.append(write_ephemeris(self.controlled_ephemeris, out_dir / f"{name}.TXT"))
        return paths


def control_orbit(
    input_ephemeris: str | Path | EphemerisTable,
    *,
    control_mode: int = 1,
    is_nrho: int = 0,
    special_mode: int = 1,
    control_interval: float = 30.0,
    feedback_arc: float = 28.0,
    special_crossings: int = 3,
    num_controls: int = 120,
    num_monte_carlo: int = 5,
    output_step: float = SECONDS_PER_DAY,
    position_accuracy: float = 1500.0,
    velocity_accuracy: float = 0.002,
    thrust_angle_err: float = 0.333,
    thrust_mean: float = 10.0,
    thrust_rel_err: float = 0.003,
    thrust_abs_err: float = 0.033,
    thrust_min: float = 0.1,
    thrust_max: float = 100.0,
    thrust_total: float = 1000.0,
    srp_error_level: float = 0.10,
    perturbation: dict[str, int] | None = None,
    dyb: Sequence[float] | None = None,
    earth_degree: int = 2,
    moon_degree: int = 2,
    real_perturbation: dict[str, int] | None = None,
    real_dyb: Sequence[float] | None = None,
    real_earth_degree: int = 10,
    real_moon_degree: int = 10,
    spice: SPICEManager | None = None,
    kernel_dir: str | None = None,
    n_workers: int = 1,
    seed: int | None = None,
    engine_layout: EngineLayout | None = None,
    momentum_interval: float = 5.0,
    srp_offset_m: Sequence[float] | None = None,
    spacecraft_mass: float = 1000.0,
    srp_torque: Sequence[float] | None = None,
    tight_tolerance_km: float = 0.1,
    tight_max_iter: int = 6,
    special_damping_factor: float = 1.0,
) -> ControlOrbitResult:
    """端到端轨道保持仿真。

    Args:
        input_ephemeris: 标称轨道星历文件路径或 ``EphemerisTable``
        control_mode: 1=目标点宽松、2=目标点严格、3=特征点、
            4=目标点宽松+角动量管理、5=目标点严格+角动量管理、6=特征点+角动量管理
        is_nrho: 目标轨道是否 NRHO（1=是；特征点模式的约束随之取
            ẋ=0 且 ż=0 由 ``special_mode`` 决定，本参数保留作输入对齐）
        special_mode: 特征点模式 1=Lissajous（ẋ=0）、2=Halo/NRHO（ẋ=0 且 ż=0）
        control_interval: 控制时间间隔（天）
        feedback_arc: 目标点模式反馈弧段（天）
        special_crossings: 特征点目标穿越 x-z 平面次数
        num_controls: 控制次数（总时间 = (N-1)·间隔）
        num_monte_carlo: 蒙特卡洛样本数（惯例 100）
        output_step: 受控星历输出间隔（秒）
        position_accuracy / velocity_accuracy: 测定轨 1-sigma（m / m/s）
        thrust_angle_err / thrust_mean / thrust_rel_err / thrust_abs_err /
        thrust_min / thrust_max / thrust_total: 分段控制误差参数
            （角度 1-sigma deg、中点值 m/s、相对/绝对 1-sigma、最小/最大
            开机 m/s、累计上限 m/s，见《控制方案.md》式 5.40）
        srp_error_level: 光压弧段随机误差量级（百分比/100）
        perturbation / earth_degree / moon_degree / dyb: 控制（理论）力模型
            （表 5-3 左列；默认 2×2、关相对论/大气，见 ``_DEFAULT_CTRL_PERTURBATION``）
        real_perturbation / real_earth_degree / real_moon_degree / real_dyb:
            真实（实际）力模型（表 5-3 右列；默认 10×10、全开）
        spice: 已加载内核的 ``SPICEManager``；缺省自动创建
        kernel_dir: SPICE 内核目录（缺省仓库 ``kernels/``；``n_workers>1``
            时 worker 用它在子进程重建上下文）
        n_workers: 进程池大小（>1 时样本并行）
        seed: 随机种子（同种子同结果）
        engine_layout: ``EngineLayout`` 实例（角动量管理模式 4-6 必填）
        momentum_interval: 角动量卸载间隔（天），0 表示与轨道控制同步
        srp_offset_m: SRP 压心相对质心偏移 ``[x,y,z]``（m），常值
        spacecraft_mass: 航天器质量（kg）
        srp_torque: 常值 SRP 力矩 ``[τx,τy,τz]``（N·m）
        tight_tolerance_km: TIGHT 模式位置重合容差（km，默认 0.1）
        tight_max_iter: TIGHT 模式微分修正迭代上限（默认 6）
        special_damping_factor: SPECIAL 模式牛顿迭代阻尼因子（<1 时启用回溯，默认 1.0 不阻尼）

    Returns:
        :class:`ControlOrbitResult`（SK_STATISTIC/MANEUVERS/受控星历）

    Raises:
        ValueError: 参数超界（控制模式、误差参数等）
        NotImplementedError: 摄动开关含 ECOM 光压或耦合项（#253）
    """
    if control_mode not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"control_mode 必须为 1-6，当前 {control_mode}")
    if control_mode >= 4 and engine_layout is None:
        raise ValueError(f"control_mode {control_mode}（角动量管理）需提供 engine_layout")
    if special_mode not in (1, 2):
        raise ValueError(
            f"special_mode 必须为 1（Lissajous）或 2（Halo/NRHO），当前 {special_mode}"
        )
    if num_monte_carlo < 1:
        raise ValueError(f"num_monte_carlo 必须 >= 1，当前 {num_monte_carlo}")
    if num_controls < 1:
        raise ValueError(f"num_controls 必须 >= 1，当前 {num_controls}")
    for name, v in [
        ("position_accuracy", position_accuracy),
        ("velocity_accuracy", velocity_accuracy),
        ("thrust_min", thrust_min),
        ("thrust_max", thrust_max),
    ]:
        if v <= 0:
            raise ValueError(f"{name} 必须为正数，当前 {v}")

    # 角动量管理时校验发动机布局
    if engine_layout is not None:
        from .momentum_management import validate_engine_layout

        validate_engine_layout(engine_layout)

    if isinstance(input_ephemeris, EphemerisTable):
        eph = input_ephemeris
    else:
        eph = read_ephemeris(input_ephemeris)

    if spice is None:
        spice = SPICEManager()
        load_design_kernels(spice, kernel_dir)
    kernel_dir = kernel_dir or default_kernel_dir()

    system = EphemerisSystem(
        bodies=[
            "SUN",
            "EARTH",
            "MOON",
            "MARS",
            "JUPITER",
            "SATURN",
            "VENUS",
            "MERCURY",
            "URANUS",
            "NEPTUNE",
        ],
        spice=spice,
        origin="EARTH",
    )
    system.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=spice),
    )

    dyb_vals = list(DEFAULT_DYB if dyb is None else dyb)
    real_dyb_vals = list(DEFAULT_DYB if real_dyb is None else real_dyb)
    theory_pert = {**_DEFAULT_CTRL_PERTURBATION, **(perturbation or {})}
    real_pert = {**theory_pert, **(real_perturbation or {})}

    cfg_ctrl = perturbation_to_force_config(
        theory_pert,
        earth_degree=earth_degree,
        moon_degree=moon_degree,
        dyb=dyb_vals,
    )
    cfg_true = perturbation_to_force_config(
        real_pert,
        earth_degree=real_earth_degree,
        moon_degree=real_moon_degree,
        dyb=real_dyb_vals,
    )

    # mode 4-6：基础控制律用 mode-3（4→1, 5→2, 6→3）
    base_mode = control_mode if control_mode <= 3 else control_mode - 3

    result = run_monte_carlo(
        eph,
        spice=spice,
        system=system,
        force_config_ctrl=cfg_ctrl,
        force_config_true=cfg_true,
        control_mode=base_mode,
        special_mode=special_mode,
        special_crossings=special_crossings,
        control_interval_days=control_interval,
        feedback_arc_days=feedback_arc,
        num_controls=num_controls,
        num_monte_carlo=num_monte_carlo,
        output_step_sec=output_step,
        position_accuracy_m=position_accuracy,
        velocity_accuracy_mps=velocity_accuracy,
        thrust_angle_err_deg=thrust_angle_err,
        thrust_mean_mps=thrust_mean,
        thrust_rel_err=thrust_rel_err,
        thrust_abs_err_mps=thrust_abs_err,
        thrust_min_mps=thrust_min,
        thrust_max_mps=thrust_max,
        thrust_total_mps=thrust_total,
        srp_error_level=srp_error_level,
        seed=seed,
        n_workers=n_workers,
        kernel_dir=kernel_dir,
        engine_layout=engine_layout,
        momentum_interval_days=momentum_interval,
        srp_offset_m=np.asarray(srp_offset_m, dtype=float) if srp_offset_m is not None else None,
        spacecraft_mass_kg=spacecraft_mass,
        srp_torque_nm=np.asarray(srp_torque, dtype=float) if srp_torque is not None else None,
        tight_tolerance_km=tight_tolerance_km,
        tight_max_iter=tight_max_iter,
        special_damping_factor=special_damping_factor,
    )
    status = (
        ConvergenceState.CONVERGED
        if result.num_failed < num_monte_carlo
        else ConvergenceState.FAILED
    )
    cause = FailureCause.NONE if status is ConvergenceState.CONVERGED else FailureCause.UNKNOWN
    message = "任务完成" if status is ConvergenceState.CONVERGED else "全部蒙特卡洛样本失败"
    return ControlOrbitResult(
        sk_statistic=result.sk_statistic(),
        num_failed=result.num_failed,
        maneuvers=result.maneuver_table(),
        controlled_ephemeris=result.controlled_ephemeris,
        raw=result,
        status=status,
        cause=cause,
        message=message,
        stages=(StageRecord("monte_carlo", applicable=True, executed=True, result_status=status),),
    )
