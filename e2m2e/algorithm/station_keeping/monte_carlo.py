"""三轨道结构蒙特卡洛仿真驱动（《控制方案.md》§1.5/§1.7）。

误差仿真的三轨道结构（§1.5.1）：

- **目标轨道**：标称轨道（FR1 设计产物），只作参照
- **真实轨道**：入轨时刻相对目标加测定轨扰动（式 5.38），以"实际"力
  模型外推，模拟探测器的真实状态
- **测量轨道**：每个控制时刻相对真实轨道加测定轨扰动（式 5.39），以
  "理论"力模型外推，控制量基于它计算

控制量施加于真实轨道；执行误差按式 5.40 分段（< Δv_min 不开机、小量
绝对误差、大量相对误差、> Δv_max 判失败）；真实轨道与控制轨道的力模型
差异为表 5-3 双配置，其中光压误差每控制弧段重新抽样、弧段内固定（表
5-3 脚注）。蒙特卡洛对每个样本独立抽样运行（§1.7，DFH 惯例 100 次），
统计总 Δv、最大 Δv 与失败次数。

传播全部走 Rust 编译力模型（``propagate_compiled*``），控制律的 STM 由
42 维增广传播给出。蒙特卡洛的样本并行在 Python 进程池（CSPICE 全局状态非线程安全，rayon
不可用，见 ``spice_ffi.rs`` 模块注释）。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ...data.types import EphemerisTable, ManeuverTable, SKStatistic
from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ICRSAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from ..dynamics import EphemerisSystem
from ..forces import ForceModel
from .error_models import (
    BoxMullerSampler,
    NavigationErrorModel,
    SrpErrorModel,
    ThrustExecutionError,
)
from .special_point import SpecialPointLaw
from .target_point import LooseTargetPointLaw, StrictTargetPointLaw

__all__ = [
    "MonteCarloResult",
    "NominalOrbit",
    "RustPropagator",
    "SampleResult",
    "SingleSampleSimulation",
    "run_monte_carlo",
]

_SECONDS_PER_DAY = 86400.0

#: 地月系统质量参数与特征尺度（与 e2m2e/dfh/cr3bp_orbits.py 的
#: earth_moon_system 一致，保证会合系转换与 design 链路同约定）
EARTH_MOON_MU = 0.0121506683
_CHAR_LENGTH_KM = 384400.0
_CHAR_PERIOD_SEC = 27.32 * 86400.0


def _earth_moon_system():
    """构造标准地月 CR3BP 系统（仅特征尺度，供会合系转换用）。"""
    from ..dynamics.cr3bp_system import CR3BP_System

    sys_ = CR3BP_System(mu=EARTH_MOON_MU, primary="Earth", secondary="Moon")
    sys_.set_characteristic_scales(_CHAR_LENGTH_KM, _CHAR_PERIOD_SEC)
    return sys_


def _utc_iso(eph: EphemerisTable, k: int) -> str:
    return (
        f"{eph.year[k]:04d}-{eph.month[k]:02d}-{eph.day[k]:02d}"
        f"T{eph.hour[k]:02d}:{eph.minute[k]:02d}:{eph.second[k]:06.3f}"
    )


class NominalOrbit:
    """标称轨道视图：``EphemerisTable`` + SPICE 时间轴，``state_at(t)`` 线性插值。

    Attributes:
        t_start/t_end: 星历时间轴覆盖范围（et 秒）
    """

    def __init__(self, ephemeris: EphemerisTable, spice: Any) -> None:
        et_all = np.array([spice.utc_to_et(_utc_iso(ephemeris, k)) for k in range(len(ephemeris))])
        # DFH 会把端点历元写两遍（数值相同），按时间戳去重
        keep = np.unique(et_all, return_index=True)[1]
        keep.sort()
        self._et = et_all[keep]
        self._states = np.column_stack(
            [
                ephemeris.position_km[keep],
                ephemeris.velocity_mps[keep] / 1000.0,  # m/s → km/s
            ]
        )

    @property
    def t_start(self) -> float:
        return float(self._et[0])

    @property
    def t_end(self) -> float:
        return float(self._et[-1])

    def state_at(self, t: float) -> npt.NDArray[np.floating]:
        """t（et 秒）时刻的标称状态（GCRS，km, km/s），逐分量线性插值。"""
        return np.array([np.interp(t, self._et, self._states[:, i]) for i in range(6)])


class RustPropagator:
    """Rust 编译力模型传播器（``StmPropagator`` 实现，单样本）。

    直接调 ``propagate_compiled*_py``（不经 ``ForceModel``），绕开其 STM
    路径对 SRP 的排除（站保控制模型含光压，STM 必须含光压贡献；Rust 侧
    已补 SRP 雅可比，见 ``compiled.rs::acceleration_and_jacobian``）。
    """

    def __init__(
        self,
        observer: str,
        forces_py: list[tuple[Any, ...]],
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_step: float = 3600.0,
        max_steps: int = 500_000,
    ) -> None:
        from e2m2e._integrators import propagate_compiled_stm_py

        self._observer = observer
        self._forces_py = forces_py
        self._rtol = rtol
        self._atol = atol
        self._max_step = max_step
        self._max_steps = max_steps
        self._propagate_compiled_stm_py = propagate_compiled_stm_py

    @staticmethod
    def _align_t_eval(t0: float, t_eval: npt.ArrayLike) -> tuple[npt.NDArray[np.floating], bool]:
        """Rust STM 传播要求 t_eval 首点 == t0（否则只输出起点状态与恒等
        STM，实测单点 [t_j] 返回的 B 块为 0）。首点不是 t0 时前置补 t0，
        返回 (对齐后的 t_eval, 是否补过)。
        """
        t_eval = np.asarray(t_eval, dtype=float)
        if len(t_eval) == 0 or t_eval[0] == t0:
            return t_eval, False
        return np.concatenate([[t0], t_eval]), True

    def propagate_with_stm(
        self, state0: npt.ArrayLike, t0: float, t_eval: npt.ArrayLike
    ) -> dict[str, npt.NDArray[np.floating]]:
        t_eval, prepended = self._align_t_eval(t0, t_eval)
        state0 = np.asarray(state0, dtype=float)
        try:
            result = self._propagate_compiled_stm_py(
                self._observer,
                self._forces_py,
                (float(t0), float(t_eval[-1])),
                [float(x) for x in t_eval],
                [float(x) for x in state0],
                self._rtol,
                self._atol,
                self._max_step,
                self._max_steps,
            )
        except RuntimeError:
            # 崩溃上下文（站保调试）：状态位置/速度量级与弧段
            r = float(np.linalg.norm(state0[:3]))
            v = float(np.linalg.norm(state0[3:]))
            raise RuntimeError(
                f"STM propagation failed: state0 r={r:.3f} km, v={v:.6f} km/s, "
                f"t0={t0:.3f}, dt={float(t_eval[-1]) - t0:.3f} s"
            ) from None
        states = np.asarray(result["states"])
        stm = np.asarray(result["stm"]).reshape(-1, 6, 6)
        times = np.asarray(result["time"])
        if prepended:
            states, stm, times = states[1:], stm[1:], times[1:]
        return {"time": times, "states": states, "stm": stm}

    def propagate(
        self, state0: npt.ArrayLike, t0: float, t_eval: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """无 STM 传播，返回 ``(n, 6)`` 状态序列。

        走 STM 传播路径（``propagate_compiled_stm_py``）：``propagate_compiled``
        的步长无上限，稀疏 t_eval 下自适应步长失控（实测 2 点 vs 31 点
        网格 30 天结果差 22 万 km）；STM 路径受 ``max_step`` 限制，网格
        无关。STM 计算为附带成本。
        """
        t_eval, prepended = self._align_t_eval(t0, t_eval)
        result = self._propagate_compiled_stm_py(
            self._observer,
            self._forces_py,
            (float(t0), float(t_eval[-1])),
            [float(x) for x in t_eval],
            [float(x) for x in np.asarray(state0)],
            self._rtol,
            self._atol,
            self._max_step,
            self._max_steps,
        )
        states = np.asarray(result["states"])
        return states[1:] if prepended else states


class PropagatorFactory:
    """力模型配置 → 传播器工厂（真实力模型的光压乘子每弧段可换）。

    容差/步长取站保统计量级够用的默认（1e-10、3600 s）：DFH 黄金对比的
    对象是统计特征（±30% 量级），1e-12 容差只增加耗时（实测 4 控制 × 2
    样本 156 s → 约 3 倍提速空间）。
    """

    def __init__(
        self,
        system: Any,
        observer: str = "EARTH",
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_step: float = 3600.0,
        max_steps: int = 500_000,
    ) -> None:
        self._system = system
        self._observer = observer
        self._rtol = rtol
        self._atol = atol
        self._max_step = max_step
        self._max_steps = max_steps

    def _forces_py(
        self, force_config: dict[str, Any], srp_cr_scale: float = 1.0
    ) -> list[tuple[Any, ...]]:
        """力配置 → Rust 元组列表；srp 的 cr 乘上光压弧段误差乘子。"""
        fm = ForceModel.from_config(force_config, self._system)
        out: list[tuple[Any, ...]] = []
        for entry in fm._entries:  # noqa: SLF001  # 与 force_model 内部一致
            if not entry.enabled:
                continue
            spec = entry.force.to_rust_spec(self._system)
            if spec is None:
                continue
            if spec[0] == "srp":
                area, mass, cr, shadows = spec[1], spec[2], spec[3], spec[4]
                spec = ("srp", area, mass, cr * srp_cr_scale, shadows)
            out.append(spec)
        return out

    def make(self, force_config: dict[str, Any], srp_cr_scale: float = 1.0) -> RustPropagator:
        return RustPropagator(
            self._observer,
            self._forces_py(force_config, srp_cr_scale),
            rtol=self._rtol,
            atol=self._atol,
            max_step=self._max_step,
            max_steps=self._max_steps,
        )


@dataclass
class SampleResult:
    """单样本蒙特卡洛结果。

    Attributes:
        total_delta_v_mps: 总 Δv（m/s），角动量管理模式含轨道+姿态联合 Δv
        max_delta_v_mps: 最大单次 Δv（m/s）
        failed: 是否控制失败（单次 > Δv_max 或累计 > 总上限）
        maneuver_times: 控制时刻（et 秒），形状 ``(n,)``
        maneuver_dv_mps: 各控制时刻施加的 Δv（m/s），形状 ``(n,)``
        controlled_times: 受控星历时间（et 秒），形状 ``(m,)``
        controlled_states: 受控真实轨道状态（GCRS，km, km/s），形状 ``(m, 6)``
        attitude_delta_v_mps: 姿态总 Δv（m/s），仅角动量管理模式非零
        attitude_delta_v_independent_mps: 姿态不影响轨道时的 Δv（m/s），即
            纯角动量卸载（不含轨道控制贡献）的 Δv 累计量
    """

    total_delta_v_mps: float
    max_delta_v_mps: float
    failed: bool
    maneuver_times: npt.NDArray[np.floating]
    maneuver_dv_mps: npt.NDArray[np.floating]
    controlled_times: npt.NDArray[np.floating]
    controlled_states: npt.NDArray[np.floating]
    attitude_delta_v_mps: float = 0.0
    attitude_delta_v_independent_mps: float = 0.0


@dataclass
class MonteCarloResult:
    """蒙特卡洛批量结果（对应 DFH SK_STATISTIC/MANEUVERS/受控星历）。

    Attributes:
        total_delta_v: 每样本总 Δv（m/s），形状 ``(n_samples,)``
        max_delta_v: 每样本最大 Δv（m/s），形状 ``(n_samples,)``
        failed_mask: 每样本失败标记，形状 ``(n_samples,)``
        num_failed: 失败样本数
        maneuvers: 第一个样本的机动序列（et 秒, m/s）
        controlled_ephemeris: 第一个样本的受控星历（可为 None）
        attitude_delta_v: 每样本姿态总 Δv（m/s），形状 ``(n_samples,)``（角动量管理）
        attitude_delta_v_independent: 每样本姿态独立 Δv（m/s），形状 ``(n_samples,)``
    """

    total_delta_v: npt.NDArray[np.floating]
    max_delta_v: npt.NDArray[np.floating]
    failed_mask: npt.NDArray[np.bool_]
    num_failed: int
    maneuvers: npt.NDArray[np.floating] = field(default_factory=lambda: np.empty((0, 2)))
    controlled_ephemeris: EphemerisTable | None = None
    attitude_delta_v: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0)
    )
    attitude_delta_v_independent: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0)
    )

    def sk_statistic(self) -> SKStatistic:
        """组装 SK_STATISTIC 表（3 列或 5 列版）。

        无角动量管理时 3 列（仿真序号/总 Δv/最大 Δv，序号在写出时追加）；
        含角动量管理时 5 列（追加姿态总 Δv、姿态独立 Δv）。

        对齐 DFH 口径：失败样本不写统计行（黄金样本中失败行缺失）。
        """
        keep = ~self.failed_mask
        if self.attitude_delta_v.size > 0:
            rows = np.column_stack([
                self.total_delta_v[keep],
                self.max_delta_v[keep],
                self.attitude_delta_v[keep],
                self.attitude_delta_v_independent[keep],
            ])
        else:
            rows = np.column_stack([self.total_delta_v[keep], self.max_delta_v[keep]])
        return SKStatistic(rows=rows, num_failed=self.num_failed)

    def maneuver_table(self) -> ManeuverTable:
        """组装 MANEUVERS 表（MJD(TDB) = 51544.5 + et/86400，SPICE et 原点为
        J2000 历元 JD 2451545.0，即 MJD 51544.5）。"""
        mjd = 51544.5 + self.maneuvers[:, 0] / _SECONDS_PER_DAY
        return ManeuverTable(mjd_tdb=mjd, delta_v_mps=self.maneuvers[:, 1])


@dataclass
class SingleSampleSimulation:
    """单样本闭环仿真（三轨道结构 + 分段控制误差 + 光压弧段误差）。

    Attributes:
        nominal: 标称轨道视图
        law: 控制律（special/strict/loose 之一）
        control_interval_sec: 控制时间间隔（秒）
        num_controls: 控制次数（总时间 = (N-1)·间隔）
        output_step_sec: 受控星历输出间隔（秒）
        nav_error: 测定轨误差模型
        thrust_error: 分段控制误差模型
        srp_error: 光压弧段误差模型
        thrust_total_mps: 累计 Δv 上限（超过判失败）
        factory: 传播器工厂（控制/真实力模型由 ``force_config_ctrl/true`` 构造）
        force_config_ctrl: 控制（理论）力模型配置
        force_config_true: 真实（实际）力模型配置
        sampler: 标准正态采样器
        engine_layout: 姿态发动机布局（None 表示无角动量管理）
        momentum_interval_sec: 角动量卸载间隔（秒），0 表示与轨道控制同步
        srp_offset_m: SRP 压心相对质心偏移（m），None 表示无 SRP 力矩
        spacecraft_mass_kg: 航天器质量（kg）
        srp_torque_nm: 常值 SRP 力矩（N·m），None 时由 srp_offset_m 和 SRP 力模型计算
    """

    nominal: NominalOrbit
    law: Any
    control_interval_sec: float
    num_controls: int
    output_step_sec: float
    nav_error: NavigationErrorModel
    thrust_error: ThrustExecutionError
    srp_error: SrpErrorModel
    thrust_total_mps: float
    factory: PropagatorFactory
    force_config_ctrl: dict[str, Any]
    force_config_true: dict[str, Any]
    sampler: BoxMullerSampler
    engine_layout: Any = None  # EngineLayout | None
    momentum_interval_sec: float = 0.0
    srp_offset_m: npt.NDArray[np.floating] | None = None
    spacecraft_mass_kg: float = 1000.0
    srp_torque_nm: npt.NDArray[np.floating] | None = None

    def run(self) -> SampleResult:
        from .momentum_management import (
            compute_delta_m as _compute_dm,
        )
        from .momentum_management import (
            solve_joint_control as _solve_joint,
        )
        from .momentum_management import (
            solve_momentum_unload as _solve_unload,
        )

        nominal = self.nominal
        t0 = nominal.t_start

        # 式 5.37：测量初值相对目标扰动；式 5.38：真实初值相对测量扰动
        x_meas = self.nav_error.perturb(nominal.state_at(t0), self.sampler)
        x_true = self.nav_error.perturb(x_meas, self.sampler)

        prop_ctrl = self.factory.make(self.force_config_ctrl)

        # DFH 惯例（MANEUVERS 结构，golden 实测）：t0 为参考行（Δv=0），
        # 控制动作发生在 t0 + k·间隔（k = 1..NumControls-2，共
        # NumControls-2 次）；总时长 (NumControls-1)·间隔，最后一个控制
        # 时刻（跨度终点）不施加控制。
        maneuver_times = [t0]
        maneuver_dv = [0.0]
        controlled_times = [t0]
        controlled_states = [x_true.copy()]

        total_dv = 0.0
        max_dv = 0.0
        attitude_total_dv = 0.0
        attitude_independent_dv = 0.0
        failed = False
        t_prev = t0

        # 角动量管理：SRP 力矩常值（用户输入 srp_torque_nm）
        has_mm = self.engine_layout is not None
        srp_torque = (
            np.asarray(self.srp_torque_nm, dtype=float)
            if self.srp_torque_nm is not None
            else None
        )
        layout = self.engine_layout
        mass = self.spacecraft_mass_kg
        delta_m_cum = np.zeros(3)  # 累积角动量（上次卸载以来）

        # 构造合并事件时间表（轨道控制 + 角动量卸载）
        # event_type[t] = "orbital" | "momentum" | "both"
        orbital_times: set[float] = set()
        momentum_times: set[float] = set()
        if has_mm and self.momentum_interval_sec > 0:
            t_end = t0 + (self.num_controls - 1) * self.control_interval_sec
            orbital_times = {t0 + k * self.control_interval_sec
                            for k in range(1, self.num_controls - 1)}
            t_m = t0 + self.momentum_interval_sec
            while t_m < t_end:
                # 与最近轨道控制时刻对齐（相差 < 1 秒视为同时）
                if any(abs(t_m - t_o) < 1.0 for t_o in orbital_times):
                    # 合并到轨道事件（标记为 both）
                    pass
                else:
                    momentum_times.add(t_m)
                t_m += self.momentum_interval_sec
            events = sorted(orbital_times | momentum_times)
        elif has_mm:
            # 卸载间隔=0：与轨道控制同步
            events = [t0 + k * self.control_interval_sec
                      for k in range(1, self.num_controls - 1)]
        else:
            # 无角动量管理
            events = [t0 + k * self.control_interval_sec
                      for k in range(1, self.num_controls - 1)]

        for t_k in events:
            # 弧段 [t_prev, t_k] 真实轨道传播（实际力模型 + 本弧段光压误差，
            # 表 5-3 脚注：每控制弧段重新抽样、弧段内固定）
            cr_scale = self.srp_error.sample_cr_scale(self.sampler)
            prop_true = self.factory.make(self.force_config_true, srp_cr_scale=cr_scale)
            n_out = max(1, int(round((t_k - t_prev) / self.output_step_sec)))
            t_eval = np.linspace(t_prev, t_k, n_out + 1)
            states = prop_true.propagate(x_true, t_prev, t_eval)
            controlled_times.extend(t_eval[1:])
            controlled_states.extend(states[1:])
            x_true = states[-1]
            # 式 5.39：控制时刻的测量状态 = 真实 + 测定轨扰动
            x_meas = self.nav_error.perturb(x_true, self.sampler)

            # 角动量累积（SRP 力矩 × 弧段时间）
            if has_mm and srp_torque is not None:
                delta_m_cum += _compute_dm(srp_torque, t_k - t_prev)

            # 判断事件类型
            is_orbital = t_k in orbital_times or (
                has_mm and self.momentum_interval_sec <= 0
            )
            is_momentum = has_mm and (
                t_k in momentum_times
                or self.momentum_interval_sec <= 0
            )

            # 控制量计算
            if has_mm and is_orbital and is_momentum:
                # 联合控制：轨道 + 角动量一次开机
                dv_c = self.law.compute_maneuver(x_meas, t_k, propagator=prop_ctrl, nominal=nominal)
                if dv_c is not None:
                    V = _solve_joint(layout.E, layout.E_r, dv_c * 1000.0,
                                     delta_m_cum, mass)
                    dv_orbital = layout.E @ V
                    # 姿态独立 Δv：纯卸载需要的 Δv（与联合控制对比用）
                    dv_att_ind = _solve_unload(layout.E_r, delta_m_cum, mass)
                    attitude_independent_dv += float(np.linalg.norm(layout.E @ dv_att_ind))
                    dv_r_orbital, failed_k = self.thrust_error.apply(dv_orbital, self.sampler)
                    # 联合控制姿态贡献：纯卸载所需的发动机输出投影到轨道面
                    if failed_k:
                        pass
                    elif np.linalg.norm(delta_m_cum) > 0:
                        att_contribution = float(np.linalg.norm(layout.E @ dv_att_ind))
                        attitude_total_dv += att_contribution
                else:
                    dv_r_orbital = None
                    failed_k = False
                    if np.linalg.norm(delta_m_cum) > 0:
                        V = _solve_unload(layout.E_r, delta_m_cum, mass)
                        dv_att_ind = V
                        attitude_independent_dv += float(np.linalg.norm(layout.E @ dv_att_ind))
                    else:
                        V = None
                if failed_k:
                    failed = True
                    break
            elif has_mm and not is_orbital:
                # 纯角动量卸载
                dv_r_orbital = None
                failed_k = False
                if np.linalg.norm(delta_m_cum) > 0:
                    V = _solve_unload(layout.E_r, delta_m_cum, mass)
                    dv_att_ind = V
                    attitude_independent_dv += float(np.linalg.norm(layout.E @ dv_att_ind))
                    dv_att, failed_k = self.thrust_error.apply(layout.E @ V, self.sampler)
                    if failed_k:
                        failed = True
                        break
                    if dv_att is not None:
                        att_mag = float(np.linalg.norm(dv_att))
                        attitude_total_dv += att_mag
                        total_dv += att_mag
                        max_dv = max(max_dv, att_mag)
                        x_true = x_true.copy()
                        x_true[3:] += dv_att / 1000.0
                V = None
            else:
                # 纯轨道控制（无角动量管理 或 角动量卸载间隔=0 与轨道同步）
                dv_c = self.law.compute_maneuver(x_meas, t_k, propagator=prop_ctrl, nominal=nominal)
                if dv_c is None:
                    dv_r_orbital = None
                    failed_k = False
                else:
                    dv_r_orbital, failed_k = self.thrust_error.apply(dv_c * 1000.0, self.sampler)
                if failed_k:
                    failed = True
                    break
                # 同步模式下角动量卸载与轨道控制合并
                V = None
                if has_mm and np.linalg.norm(delta_m_cum) > 0 and self.momentum_interval_sec <= 0:
                    if dv_c is not None:
                        V = _solve_joint(layout.E, layout.E_r, dv_c * 1000.0,
                                         delta_m_cum, mass)
                        dv_att_ind = _solve_unload(layout.E_r, delta_m_cum, mass)
                        attitude_independent_dv += float(np.linalg.norm(layout.E @ dv_att_ind))
                    else:
                        V = _solve_unload(layout.E_r, delta_m_cum, mass)
                        dv_att_ind = V
                        attitude_independent_dv += float(np.linalg.norm(layout.E @ dv_att_ind))
                        att_applied, att_failed = self.thrust_error.apply(
                            layout.E @ V, self.sampler)
                        if att_failed:
                            failed = True
                            break
                        if att_applied is not None:
                            att_mag = float(np.linalg.norm(att_applied))
                            attitude_total_dv += att_mag
                            total_dv += att_mag
                            max_dv = max(max_dv, att_mag)
                            x_true = x_true.copy()
                            x_true[3:] += att_applied / 1000.0

            # 累计轨道控制 Δv
            mag = 0.0 if dv_r_orbital is None else float(np.linalg.norm(dv_r_orbital))
            if is_orbital:
                maneuver_times.append(t_k)
                maneuver_dv.append(mag)
                total_dv += mag
                max_dv = max(max_dv, mag)
                if total_dv > self.thrust_total_mps:
                    failed = True
                    break

                if dv_r_orbital is not None:
                    x_true = x_true.copy()
                    x_true[3:] += dv_r_orbital / 1000.0  # 施加于真实轨道（m/s → km/s）

            # 清零累积角动量（卸载完成）
            if has_mm:
                delta_m_cum = np.zeros(3)

            t_prev = t_k

        return SampleResult(
            total_delta_v_mps=total_dv,
            max_delta_v_mps=max_dv,
            failed=failed,
            maneuver_times=np.array(maneuver_times),
            maneuver_dv_mps=np.array(maneuver_dv),
            controlled_times=np.array(controlled_times),
            controlled_states=np.array(controlled_states),
            attitude_delta_v_mps=attitude_total_dv,
            attitude_delta_v_independent_mps=attitude_independent_dv,
        )


# ── 进程池：CSPICE 全局状态非线程安全，样本并行用进程隔离 ──

_CTX: dict[str, Any] = {}


def _kernel_paths(kernel_dir: str | None = None) -> list[str]:
    """Rust cspice 域的内核清单（对齐 dfh/design_orbit 的 _BODY_FIXED_KERNELS）。"""
    import os

    kernel_dir = kernel_dir or os.environ.get(
        "SPICE_KERNEL_DIR", str(Path(__file__).resolve().parents[3] / "kernels")
    )
    paths = []
    for name in [
        "de440s.bsp",
        "de430.bsp",
        "earth_latest_high_prec.bpc",
        "pck00010.tpc",
        "SPICELunaCurrentKernel.bpc",
        "SPICELunaFrameKernel.tf",
        "naif0012.tls",
        "naif0011.tls",
    ]:
        p = os.path.join(kernel_dir, name)
        if os.path.exists(p):
            paths.append(p)
    return paths


def _init_worker(params: dict[str, Any]) -> None:
    """worker ?????? SPICE ????"""
    from ...data.kernels.manager import SPICEManager
    from ..design.design_orbit import load_design_kernels

    mgr = SPICEManager()
    # 完整内核（行星历 + body-fixed 帧 + 行星名注册），与 design 链路一致
    load_design_kernels(mgr, params["kernel_dir"])
    sys = EphemerisSystem(bodies=params["bodies"], spice=mgr, origin=params["observer"])
    sys.coordinate_system = CoordinateSystem(
        axes=ICRSAxes(),
        origin=CelestialBodyOrigin(body="EARTH", spice=mgr),
    )
    _CTX["spice"] = mgr
    _CTX["system"] = sys
    _CTX["nominal"] = NominalOrbit(params["nominal_ephemeris"], mgr)


def _run_sample(spec: dict[str, Any]) -> SampleResult:
    """单样本运行（进程池任务）。``spec`` 为除种子外的公共参数。"""
    sim = _build_simulation(
        spec, BoxMullerSampler(spec["seed"]), _CTX["nominal"], _CTX["system"], _CTX["spice"]
    )
    return sim.run()


class SynodicJ2000Adapter:
    """``SynodicView`` 协议适配：包装 ``SynodicJ2000System``（et 时间轴）。

    ``SynodicJ2000System`` 的接口是 ``j2000_to_synodic(state, t_syn, et0)``
    （无量纲时间轴），特征点控制律需要 ``to_synodic(states, ets)`` 批量
    转换（绝对 et 时间轴）+ 旋转矩阵。
    """

    def __init__(self, synodic_system: Any, et0: float) -> None:
        self._sys = synodic_system
        self._et0 = float(et0)
        self._t_c = synodic_system._get_time_unit()  # noqa: SLF001

    def to_synodic(self, states: npt.ArrayLike, ets: npt.ArrayLike) -> npt.NDArray[np.floating]:
        t_syn = (np.asarray(ets, dtype=float) - self._et0) / self._t_c
        return self._sys.batch_j2000_to_synodic(states, t_syn, self._et0)

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return self._sys.synodic_axes.rotation_matrix(et)


def _make_law(
    control_mode: int,
    special_mode: int,
    special_crossings: int,
    feedback_arc_days: float,
    horizon_sec: float,
    spice: Any,
    et0: float,
) -> Any:
    """按 DFH 控制模式构造控制律（1=宽松、2=严格、3=特征点）。"""
    if control_mode == 1:
        return LooseTargetPointLaw(feedback_arc_days=feedback_arc_days)
    if control_mode == 2:
        return StrictTargetPointLaw(feedback_arc_days=feedback_arc_days)
    if control_mode == 3:
        from ..coordinate.synodic_j2000 import SynodicJ2000System

        cr3 = _earth_moon_system()
        synodic = SynodicJ2000Adapter(SynodicJ2000System(cr3bp_system=cr3, spice=spice), et0)
        return SpecialPointLaw(
            special_mode=special_mode,
            crossings=special_crossings,
            horizon_sec=horizon_sec,
            synodic=synodic,
        )
    raise ValueError(f"control_mode 必须为 1/2/3（角动量管理 4-6 归属 #261），当前 {control_mode}")


def _build_simulation(
    spec: dict[str, Any], sampler: BoxMullerSampler, nominal: NominalOrbit, system: Any, spice: Any
) -> SingleSampleSimulation:
    return SingleSampleSimulation(
        nominal=nominal,
        law=_make_law(
            spec["control_mode"],
            spec["special_mode"],
            spec["special_crossings"],
            spec["feedback_arc_days"],
            spec["control_interval_days"] * _SECONDS_PER_DAY,
            spice,
            nominal.t_start,
        ),
        control_interval_sec=spec["control_interval_days"] * _SECONDS_PER_DAY,
        num_controls=spec["num_controls"],
        output_step_sec=spec["output_step_sec"],
        nav_error=NavigationErrorModel(
            position_sigma_m=spec["position_accuracy_m"],
            velocity_sigma_mps=spec["velocity_accuracy_mps"],
        ),
        thrust_error=ThrustExecutionError(
            dv_min=spec["thrust_min_mps"],
            dv_mid=spec["thrust_mean_mps"],
            dv_max=spec["thrust_max_mps"],
            abs_sigma_mps=spec["thrust_abs_err_mps"],
            rel_sigma=spec["thrust_rel_err"],
            angle_sigma_deg=spec["thrust_angle_err_deg"],
        ),
        srp_error=SrpErrorModel(error_level=spec["srp_error_level"]),
        thrust_total_mps=spec["thrust_total_mps"],
        factory=PropagatorFactory(system, observer=spec["observer"]),
        force_config_ctrl=spec["force_config_ctrl"],
        force_config_true=spec["force_config_true"],
        sampler=sampler,
        engine_layout=spec.get("engine_layout"),
        momentum_interval_sec=spec.get("momentum_interval_sec", 0.0),
        srp_offset_m=spec.get("srp_offset_m"),
        spacecraft_mass_kg=spec.get("spacecraft_mass_kg", 1000.0),
        srp_torque_nm=spec.get("srp_torque_nm"),
    )


def run_monte_carlo(
    nominal_ephemeris: EphemerisTable,
    *,
    spice: Any,
    system: Any,
    force_config_ctrl: dict[str, Any],
    force_config_true: dict[str, Any],
    control_mode: int = 1,
    special_mode: int = 1,
    special_crossings: int = 3,
    control_interval_days: float = 30.0,
    feedback_arc_days: float = 28.0,
    num_controls: int = 120,
    num_monte_carlo: int = 5,
    output_step_sec: float = 86400.0,
    position_accuracy_m: float = 1500.0,
    velocity_accuracy_mps: float = 0.002,
    thrust_angle_err_deg: float = 0.333,
    thrust_mean_mps: float = 10.0,
    thrust_rel_err: float = 0.003,
    thrust_abs_err_mps: float = 0.033,
    thrust_min_mps: float = 0.1,
    thrust_max_mps: float = 100.0,
    thrust_total_mps: float = 1000.0,
    srp_error_level: float = 0.10,
    seed: int | None = None,
    n_workers: int = 1,
    kernel_dir: str | None = None,
    engine_layout: Any = None,
    momentum_interval_days: float = 0.0,
    srp_offset_m: npt.ArrayLike | None = None,
    spacecraft_mass_kg: float = 1000.0,
    srp_torque_nm: npt.ArrayLike | None = None,
) -> MonteCarloResult:
    """运行蒙特卡洛站保仿真（DFH 功能码 2 的数值核心）。

    Args:
        nominal_ephemeris: 标称轨道星历（FR1 ``design_orbit`` 产物）
        spice: SPICE 管理器（``n_workers>1`` 时仅用于构造标称视图）
        system: 星历动力学系统（地心 ICRF，见 tests/dfh 的 system fixture）
        force_config_ctrl: 控制（理论）力模型配置（表 5-3 左列）
        force_config_true: 真实（实际）力模型配置（表 5-3 右列）
        control_mode: 1=目标点宽松、2=目标点严格、3=特征点
        special_mode: 特征点模式 1=Lissajous、2=Halo/NRHO
        special_crossings: 特征点目标穿越次数
        control_interval_days: 控制时间间隔（天）
        feedback_arc_days: 目标点模式反馈弧段（天）
        num_controls: 控制次数（总时间 = (N-1)·间隔）
        num_monte_carlo: 蒙特卡洛样本数
        output_step_sec: 受控星历输出间隔（秒）
        position_accuracy_m/velocity_accuracy_mps: 测定轨 1-sigma
        thrust_*: 分段控制误差参数（§1.5.2）
        srp_error_level: 光压弧段随机误差量级（百分比/100）
        seed: 随机种子（同种子同结果）
        n_workers: 进程池大小（>1 时样本并行；需 ``kernel_dir`` 供 worker
            重建 SPICE 上下文）
        kernel_dir: SPICE 内核目录（``n_workers>1`` 时必填）
        engine_layout: ``EngineLayout`` 实例（None 表示无角动量管理）
        momentum_interval_days: 角动量卸载间隔（天），0 表示与轨道控制同步
        srp_offset_m: SRP 压心相对质心偏移（m）
        spacecraft_mass_kg: 航天器质量（kg）
        srp_torque_nm: 常值 SRP 力矩（N·m）

    Returns:
        :class:`MonteCarloResult`（含 SK_STATISTIC/MANEUVERS 组装方法）
    """
    nominal = NominalOrbit(nominal_ephemeris, spice)
    observer = getattr(system, "origin", "EARTH")

    spec: dict[str, Any] = {
        "control_mode": control_mode,
        "special_mode": special_mode,
        "special_crossings": special_crossings,
        "control_interval_days": control_interval_days,
        "feedback_arc_days": feedback_arc_days,
        "num_controls": num_controls,
        "output_step_sec": output_step_sec,
        "position_accuracy_m": position_accuracy_m,
        "velocity_accuracy_mps": velocity_accuracy_mps,
        "thrust_angle_err_deg": thrust_angle_err_deg,
        "thrust_mean_mps": thrust_mean_mps,
        "thrust_rel_err": thrust_rel_err,
        "thrust_abs_err_mps": thrust_abs_err_mps,
        "thrust_min_mps": thrust_min_mps,
        "thrust_max_mps": thrust_max_mps,
        "thrust_total_mps": thrust_total_mps,
        "srp_error_level": srp_error_level,
        "force_config_ctrl": force_config_ctrl,
        "force_config_true": force_config_true,
        "observer": observer,
        "engine_layout": engine_layout,
        "momentum_interval_sec": momentum_interval_days * _SECONDS_PER_DAY,
        "srp_offset_m": np.asarray(srp_offset_m, dtype=float) if srp_offset_m is not None else None,
        "spacecraft_mass_kg": spacecraft_mass_kg,
        "srp_torque_nm": (
            np.asarray(srp_torque_nm, dtype=float)
            if srp_torque_nm is not None
            else None
        ),
    }

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_monte_carlo)]

    if n_workers > 1:
        if not kernel_dir:
            raise ValueError("n_workers>1 时需提供 kernel_dir（worker 重建 SPICE 上下文）")
        worker_params = {
            **spec,
            "kernel_dir": kernel_dir,
            "bodies": list(getattr(system, "bodies", [])),
            "nominal_ephemeris": nominal_ephemeris,
        }
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(worker_params,)
        ) as ex:
            results = list(ex.map(_run_sample, [dict(spec, seed=s) for s in seeds]))
    else:
        results = [
            _build_simulation(spec, BoxMullerSampler(s), nominal, system, spice).run()
            for s in seeds
        ]

    total_dv = np.array([r.total_delta_v_mps for r in results])
    max_dv = np.array([r.max_delta_v_mps for r in results])
    failed_mask = np.array([r.failed for r in results], dtype=bool)
    num_failed = int(failed_mask.sum())

    att_dv = np.array([r.attitude_delta_v_mps for r in results])
    att_dv_ind = np.array([r.attitude_delta_v_independent_mps for r in results])

    first = results[0] if results else None
    maneuvers = (
        np.column_stack([first.maneuver_times, first.maneuver_dv_mps])
        if first is not None
        else np.empty((0, 2))
    )
    controlled = _build_controlled_ephemeris(first, nominal, spice) if first is not None else None
    return MonteCarloResult(
        total_delta_v=total_dv,
        max_delta_v=max_dv,
        failed_mask=failed_mask,
        num_failed=num_failed,
        maneuvers=maneuvers,
        controlled_ephemeris=controlled,
        attitude_delta_v=att_dv,
        attitude_delta_v_independent=att_dv_ind,
    )


def _build_controlled_ephemeris(
    sample: SampleResult, nominal: NominalOrbit, spice: Any
) -> EphemerisTable | None:
    """把样本的受控真实轨道组装成 DFH 同格式星历表（UTC + GCRS + 会合系）。

    会合系列与 ``design_orbit`` 同约定：地心归一（月球在 +x），内部转换器
    输出质心归一后 x 加 mu 平移。星历为空（控制失败早退）时返回 None。
    """
    if sample.controlled_times.size == 0:
        return None
    from datetime import datetime

    from ..coordinate.synodic_j2000 import SynodicJ2000System

    cr3 = _earth_moon_system()
    syn_j2000 = SynodicJ2000System(cr3bp_system=cr3, spice=spice)

    et_grid = sample.controlled_times
    states = sample.controlled_states
    t_c = cr3.characteristic_time
    t_syn = (et_grid - nominal.t_start) / t_c
    synodic = syn_j2000.batch_j2000_to_synodic(states, t_syn, nominal.t_start)[:, :3]
    synodic[:, 0] += cr3.mu

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
