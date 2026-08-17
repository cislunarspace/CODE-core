"""
不变流形计算模块

提供 CR3BP 周期轨道不变流形（稳定/不稳定）的种子生成与批量传播功能。

算法要点
--------
单值矩阵 M（沿轨道传播一周的 STM）的实特征值给出流形方向：
稳定流形取模小于 1 的实特征向量，不稳定流形取模大于 1 的实特征向量。
沿周期轨道取 n_points 个相位点，用从轨道首点到各相位的 STM 把特征向量
转运到该相位，位置部分归一化后施加 ±ε 的无量纲扰动得到种子；
稳定流形反向积分、不稳定流形正向积分得到流形管。

数值内核在 Rust（特征分解、STM 转运、种子扰动、批量传播调度）；
Python 侧只做参数校验、领域对象组装与可选的事后截面截断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...data.types.orbit import Orbit
from ...exceptions import PropagationFailure
from ...integrators import (
    manifold_propagate_py,
    manifold_seeds_py,
    require_rust_extension,
)
from ..dynamics import CR3BP_Dynamics

if TYPE_CHECKING:
    from .sections import PoincareSection

logger = logging.getLogger(__name__)


class ManifoldKind(Enum):
    """不变流形类型枚举"""

    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass
class ManifoldTube:
    """不变流形管（一族流形弧）。

    Attributes:
        orbit: 周期轨道引用
        kind: 流形类型（稳定/不稳定）
        branch: 扰动分支（``"+"`` / ``"-"``）
        epsilon: 种子扰动幅度（无量纲）
        trajectories: 流形弧列表（无量纲 CR3BP 态）
    """

    orbit: Orbit
    kind: ManifoldKind
    branch: str
    epsilon: float
    trajectories: list[Orbit] = field(default_factory=list)


class InvariantManifold:
    """周期轨道的不变流形

    Attributes:
        orbit: 周期轨道（可只存首点，但 period 须已设置）
        kind: 流形类型（稳定/不稳定）
        branch: 扰动分支 ``"+"`` / ``"-"``
        epsilon: 无量纲扰动幅度（位置方向长度，典型取 50 km / DU）
        dynamics: CR3BP_Dynamics 对象
    """

    # 传播采样步长（无量纲时间），密采样保证事后截面检测精度
    SAMPLE_DT = 0.005

    def __init__(
        self,
        orbit: Orbit,
        kind: ManifoldKind,
        branch: Literal["+", "-"],
        epsilon: float,
    ) -> None:
        """初始化不变流形

        Args:
            orbit: 周期轨道（period 须已知；只存首点时 seeds 会先传播一周采样相位）
            kind: ManifoldKind.STABLE 或 ManifoldKind.UNSTABLE
            branch: 扰动方向分支，``"+"`` 或 ``"-"``
            epsilon: 无量纲扰动幅度（正值）

        Raises:
            ValueError: 轨道周期未知、未关联 system、branch 或 epsilon 非法
        """
        if orbit.period is None or orbit.period <= 0:
            raise ValueError("轨道周期未知，无法生成流形种子")
        system = getattr(orbit, "system", None)
        if system is None:
            raise ValueError("orbit 须关联 system 才能计算不变流形")
        if branch not in ("+", "-"):
            raise ValueError(f"branch 必须为 '+' 或 '-'，当前为 {branch}")
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须为正数，当前为 {epsilon}")

        self.orbit = orbit
        self.kind = ManifoldKind(kind)
        self.branch = branch
        self.epsilon = float(epsilon)
        self.dynamics = CR3BP_Dynamics(system)

        # 最近一次 seeds 的相位点数（供 propagate 默认 n_points）
        self._cached_n_points: int | None = None

    def seeds(self, n_points: int) -> np.ndarray:
        """生成相位扫掠种子

        沿周期轨道均匀取 n_points 个相位点，把首点处的流形特征向量
        用各段 STM 转运到该相位，位置部分归一化后施加 ±ε 扰动。

        Args:
            n_points: 相位点个数

        Returns:
            种子状态数组，形状 (n_points, 6)
        """
        if n_points < 1:
            raise ValueError(f"n_points 必须大于等于 1，当前为 {n_points}")
        require_rust_extension("manifold_seeds_py")

        assert self.orbit.period is not None
        x0 = np.asarray(self.orbit.states[0], dtype=float).reshape(6)
        branch_sign = 1.0 if self.branch == "+" else -1.0
        max_step = self.dynamics.max_step
        max_step_arg = None if not np.isfinite(max_step) or max_step <= 0 else float(max_step)

        raw = manifold_seeds_py(
            float(self.dynamics.system.mu),
            x0.tolist(),
            float(self.orbit.period),
            self.kind.value,
            branch_sign,
            self.epsilon,
            int(n_points),
            float(self.dynamics.rtol),
            float(self.dynamics.atol),
            max_step_arg,
        )
        seeds = np.asarray(raw["seeds"], dtype=float)
        if seeds.shape != (n_points, 6):
            raise RuntimeError(f"Rust 种子形状异常: {seeds.shape}，期望 ({n_points}, 6)")
        self._cached_n_points = n_points
        return seeds

    def propagate(
        self,
        t_span: float,
        section: PoincareSection | None = None,
        n_workers: int = 1,
    ) -> ManifoldTube:
        """批量传播流形弧

        积分方向由 kind 决定：STABLE 反向积分、UNSTABLE 正向积分，
        t_span 取绝对值作为积分时长。

        Args:
            t_span: 积分时长（无量纲时间，符号被忽略）
            section: 可选庞加莱截面；给定时每条弧在首次穿越截面处截断，
                并把求精后的穿越态追加为弧的末点
            n_workers: 并行 worker 数（>1 启用 Rayon）

        Returns:
            ManifoldTube: 流形管，含全部流形弧
        """
        duration = abs(float(t_span))
        if duration <= 0:
            raise ValueError(f"t_span 必须非零，当前为 {t_span}")

        seeds = self.seeds(self._default_seed_count())

        # 测试注入缝：若 dynamics.propagate 被 monkeypatch，走逐种子 Python
        # 调度以保留失败跳过语义（ADR 0020 测试注入缝豁免）。生产路径不触发。
        if self._dynamics_propagate_monkeypatched():
            return self._propagate_via_dynamics(seeds, duration, section)

        require_rust_extension("manifold_propagate_py")
        max_step = self.dynamics.max_step
        max_step_arg = None if not np.isfinite(max_step) or max_step <= 0 else float(max_step)

        parallel = n_workers > 1
        raw = manifold_propagate_py(
            float(self.dynamics.system.mu),
            seeds.reshape(-1).tolist(),
            self.kind.value,
            duration,
            float(self.SAMPLE_DT),
            float(self.dynamics.rtol),
            float(self.dynamics.atol),
            max_step_arg,
            n_workers=int(n_workers) if parallel else None,
            parallel=parallel,
        )

        trajectories: list[Orbit] = []
        for arc in raw["arcs"]:
            times = np.asarray(arc["times"], dtype=float)
            states = np.asarray(arc["states"], dtype=float)
            if section is not None:
                times, states = self._truncate_at_first_crossing(times, states, section)
            if len(states) == 0:
                logger.warning("流形弧积分返回空轨迹，跳过该种子")
                continue
            trajectories.append(Orbit(states=states, times=times, system=self.orbit.system))

        if raw.get("n_failures", 0):
            logger.warning("流形弧积分失败 %d 条，已跳过", int(raw["n_failures"]))

        return ManifoldTube(
            orbit=self.orbit,
            kind=self.kind,
            branch=self.branch,
            epsilon=self.epsilon,
            trajectories=trajectories,
        )

    # ---- 内部实现 ----

    _DEFAULT_N_POINTS = 50

    def _default_seed_count(self) -> int:
        """propagate 使用的相位点个数：沿用已有 seeds 缓存的个数，否则取默认值"""
        if self._cached_n_points is not None:
            return self._cached_n_points
        return self._DEFAULT_N_POINTS

    def _dynamics_propagate_monkeypatched(self) -> bool:
        """检测 ``dynamics.propagate`` 是否被测试 monkeypatch。

        ``monkeypatch.setattr(dynamics, "propagate", ...)`` 会在实例
        ``__dict__`` 上留下属性；生产路径不会触发。
        """
        return "propagate" in getattr(self.dynamics, "__dict__", {})

    def _propagate_via_dynamics(
        self,
        seeds: np.ndarray,
        duration: float,
        section: PoincareSection | None,
    ) -> ManifoldTube:
        """测试注入路径：逐种子调 dynamics.propagate，保留失败跳过语义。"""
        t_final = -duration if self.kind is ManifoldKind.STABLE else duration
        n_samples = max(int(np.ceil(duration / self.SAMPLE_DT)) + 1, 2)
        t_eval = np.linspace(0.0, t_final, n_samples)
        trajectories: list[Orbit] = []
        for x0 in seeds:
            try:
                result = self.dynamics.propagate(x0, (0.0, t_final), t_eval=t_eval)
            except PropagationFailure:
                logger.warning("流形弧积分失败，跳过该种子")
                continue
            times = np.asarray(result["time"], dtype=float)
            states = np.asarray(result["states"], dtype=float)
            if section is not None:
                times, states = self._truncate_at_first_crossing(times, states, section)
            if len(states) == 0:
                logger.warning("流形弧积分返回空轨迹，跳过该种子")
                continue
            trajectories.append(Orbit(states=states, times=times, system=self.orbit.system))
        return ManifoldTube(
            orbit=self.orbit,
            kind=self.kind,
            branch=self.branch,
            epsilon=self.epsilon,
            trajectories=trajectories,
        )

    def _truncate_at_first_crossing(
        self,
        times: np.ndarray,
        states: np.ndarray,
        section: PoincareSection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """在首次截面穿越处截断流形弧，穿越态求精后作为末点

        忽略第一段采样区间内的穿越：种子本身可能恰好落在截面上，
        起点的穿越不代表流形弧离开轨道后的行为。
        """
        from .sections import detect_crossings

        crossings = detect_crossings(times, states, section)
        crossings = [c for c in crossings if c[2] > 0]
        if not crossings:
            return times, states

        t_cross, state_cross, seg_idx = crossings[0]
        new_times = np.append(times[: seg_idx + 1], t_cross)
        new_states = np.vstack([states[: seg_idx + 1], state_cross])
        return new_times, new_states
