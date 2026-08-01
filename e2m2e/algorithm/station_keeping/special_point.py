"""特征点控制（《控制方案.md》§1.2）。

L3 点 Lissajous 轨道穿越地月连线坐标系的 x-z 平面时近似满足 ẋ=0（式
5.31）；NRHO/Halo 作为晕轨道的特殊成员还满足 ż=0（式 5.32）。控制过程：
在当前控制时刻 t₀ 施加 Δv，使轨道在第 1 次（或第 N 次）穿越 x-z 平面
（y=0）时满足上述约束，牛顿迭代（式 5.33/5.34）求解，雅可比为从 t₀ 至
穿越时刻状态转移矩阵的子矩阵；解不唯一时取最小范数解（本项目做法）。

实现要点：

- 传播在 GCRS 惯性系进行（Rust 42 维 STM 传播），约束在会合系评估：
  穿越检测用会合系 y 分量符号变化，约束残差用会合系速度分量
- 雅可比：``∂ẋ_syn/∂v₀ = e1ᵀ·Φ[3:6,3:6]``、``∂ż_syn/∂v₀ = e3ᵀ·Φ[3:6,3:6]``
  （会合系基向量 e1/e3 在穿越时刻求值；旋转/平移项不依赖 v₀，对雅可比
  无贡献，穿越时刻随 v₀ 的变化项按 §1.2 的"STM 子矩阵"处理忽略）
- 牛顿迭代每次重传播：初迭代粗网格全弧段找第 N 次穿越区间，后续迭代
  只在上次穿越时刻 ± 窗口内传播（穿越时刻收敛过程中移动很小）

控制量输出为 GCRS 速度增量（km/s），与传播器单位一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .target_point import NominalOrbitView

__all__ = ["SpecialPointLaw", "StmPropagator", "SynodicView"]

_SECONDS_PER_DAY = 86400.0


class StmPropagator(Protocol):
    """带 STM 的传播器（Rust 批量/单样本 STM 传播的薄封装）。"""

    def propagate_with_stm(
        self, state0: npt.ArrayLike, t0: float, t_eval: npt.ArrayLike
    ) -> dict[str, npt.NDArray[np.floating]]:
        """从 ``state0``（GCRS，km, km/s）在 ``t0`` 时刻传播到 ``t_eval``
        各时间点（秒）。

        Returns:
            ``{"time", "states" (n,6), "stm" (n,6,6)}``
        """
        ...


class SynodicView(Protocol):
    """会合系视图：GCRS 状态批量转会合系 + 会合系旋转矩阵。"""

    def to_synodic(self, states: npt.ArrayLike, ets: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """GCRS 状态（km, km/s）批量转无量纲会合系状态（n,6）。"""
        ...

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """会合系旋转矩阵 R（``r_icrf = R @ r_syn``）。"""
        ...


def _detect_crossing_intervals(y: npt.NDArray[np.floating]) -> list[tuple[int, int]]:
    """返回 y 分量符号变化的区间索引对（``y[i]*y[i+1] < 0``）。"""
    sign_change = np.where(y[:-1] * y[1:] < 0.0)[0]
    return [(int(i), int(i) + 1) for i in sign_change]


@dataclass
class SpecialPointLaw:
    """特征点控制律。

    Attributes:
        special_mode: 特征点控制模式——1=Lissajous 类型（约束 ẋ=0）；
            2=halo/NRHO 类型（约束 ẋ=0 且 ż=0）
        crossings: 目标穿越次数 N（第 N 次穿越 x-z 平面），DFH 默认 3
        tolerance: 约束残差容差（无量纲会合系速度，1e-6 约合 1e-3 m/s）
        max_iter: 牛顿迭代上限
        window_days: 后续迭代的穿越时刻搜索窗口（天）
        horizon_sec: 首迭代搜索穿越的弧段长度（秒），取控制时间间隔
        synodic: 会合系视图（GCRS↔会合系批量转换 + 旋转矩阵）
    """

    special_mode: int = 1
    crossings: int = 3
    tolerance: float = 1e-6
    max_iter: int = 8
    window_days: float = 1.0
    horizon_sec: float | None = None
    synodic: SynodicView | None = None

    def __post_init__(self) -> None:
        if self.special_mode not in (1, 2):
            raise ValueError(
                f"special_mode 必须为 1（Lissajous）或 2（halo/NRHO），当前 {self.special_mode}"
            )
        if self.crossings < 1:
            raise ValueError(f"crossings 必须 >= 1，当前 {self.crossings}")

    def _constraint(self, v_syn: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """约束残差 g：mode 1 为 [ẋ]，mode 2 为 [ẋ, ż]（无量纲）。"""
        if self.special_mode == 1:
            return np.array([v_syn[0]])
        return np.array([v_syn[0], v_syn[2]])

    def _jacobian(
        self, stm_vv: npt.NDArray[np.floating], rotation: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """约束雅可比 ∂g/∂v₀：e1ᵀ·Φ_vv（mode 1）或 [e1ᵀ; e3ᵀ]·Φ_vv（mode 2）。

        ``rotation`` 为会合系旋转矩阵 R（列 = e1/e2/e3），
        ``stm_vv`` 为 Φ(t*,t₀) 的 3×3 速度块。
        """
        e1 = rotation[:, 0]
        rows = [e1]
        if self.special_mode == 2:
            rows.append(rotation[:, 2])
        jac = np.stack(rows)
        return jac @ stm_vv

    def _find_crossing(
        self,
        state0: npt.NDArray[np.floating],
        t0: float,
        t_horizon: float,
        propagator: StmPropagator,
        synodic: SynodicView,
        grid_sec: float,
        window: tuple[float, float] | None = None,
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]] | None:
        """找第 N 次 y=0 穿越。

        粗网格（或窗口内细网格）传播 → 符号变化区间 → 区间内细传播
        插值精化。返回 (t*, 穿越处会合状态, 穿越处 STM)，找不到返回 None。

        Args:
            window: 非 None 时只在窗口内传播（迭代后期 t* 收敛后使用）
        """
        if window is not None:
            t_a, t_b = window
            # 穿越不可能早于控制时刻：窗口下限 clamp 到 t0（否则 t_eval
            # 在 prepend t0 后非单调，Rust 传播出现负步长坍缩）
            t_a = max(t_a, t0)
            if t_b <= t_a:
                return None
            # 窗口内按粗网格间隔布点（避免穿越区间被大网格吞掉）
            n_grid = int(np.clip((t_b - t_a) / grid_sec, 64, 2048))
        else:
            t_a, t_b = t0, t_horizon
            n_grid = max(2, int((t_b - t_a) / grid_sec) + 1)

        t_eval = np.linspace(t_a, t_b, n_grid)
        res = propagator.propagate_with_stm(state0, t0, t_eval)
        states = np.asarray(res["states"])
        times = np.asarray(res["time"])

        syn_states = synodic.to_synodic(states, times)
        y = syn_states[:, 1]
        intervals = _detect_crossing_intervals(y)
        if len(intervals) < self.crossings:
            return None
        i0, i1 = intervals[self.crossings - 1]

        # 区间内细传播（间隔 ~5 s）精化穿越时刻与 STM
        t_fine = np.linspace(times[i0], times[i1], 128)
        res_fine = propagator.propagate_with_stm(state0, t0, t_fine)
        states_fine = np.asarray(res_fine["states"])
        stm_fine = np.asarray(res_fine["stm"])
        y_fine = synodic.to_synodic(states_fine, t_fine)[:, 1]

        sign = y_fine[:-1] * y_fine[1:] < 0.0
        if not np.any(sign):
            # 粗网格区间在细网格上未变号（边界情形），取最接近零的点
            j = int(np.argmin(np.abs(y_fine)))
            frac = 0.0
        else:
            j = int(np.argmax(sign))
            frac = -y_fine[j] / (y_fine[j + 1] - y_fine[j]) if y_fine[j + 1] != y_fine[j] else 0.5
        t_star = t_fine[j] + frac * (t_fine[j + 1] - t_fine[j])

        # 取细网格上与 t* 最近点的状态/STM
        k = int(np.argmin(np.abs(t_fine - t_star)))
        return t_star, states_fine[k], stm_fine[k]

    def compute_maneuver(
        self,
        state0: npt.ArrayLike,
        t0: float,
        *,
        propagator: StmPropagator,
        nominal: NominalOrbitView | None = None,
        grid_sec: float = 600.0,
    ) -> npt.NDArray[np.floating] | None:
        """计算控制量 Δv（GCRS，km/s）。

        Args:
            state0: 控制时刻状态（GCRS，km, km/s；通常为测量轨道状态）
            t0: 控制时刻（秒）
            propagator: 带 STM 传播器
            nominal: 未使用（特征点控制不需要标称轨道），为接口一致保留
            grid_sec: 首迭代粗网格间隔（秒）

        Returns:
            Δv 矢量（km/s）；找不到穿越（弧段内不足 N 次）或未收敛时
            返回 None
        """
        if self.synodic is None or self.horizon_sec is None:
            raise ValueError("特征点控制需要构造时提供 synodic 与 horizon_sec")
        synodic = self.synodic
        t_horizon = t0 + self.horizon_sec

        state0 = np.asarray(state0, dtype=float)
        v0 = state0[3:].copy()
        t_star: float | None = None
        window: tuple[float, float] | None = None
        g_norm = np.inf

        for _ in range(self.max_iter):
            state_v = state0.copy()
            state_v[3:] = v0

            found = self._find_crossing(
                state_v, t0, t_horizon, propagator, synodic, grid_sec, window=window
            )
            if found is None:
                return None
            t_star, state_at, stm_at = found

            v_syn = synodic.to_synodic(state_at[np.newaxis, :], np.array([t_star]))[0, 3:]
            g = self._constraint(v_syn)
            g_norm = float(np.linalg.norm(g))
            if g_norm < self.tolerance:
                break

            rotation = synodic.rotation_matrix(t_star)
            jac = self._jacobian(stm_at[3:6, 3:6], rotation)
            # 最小范数解（式 5.33/5.34 解不唯一时的本项目做法）
            dv, *_ = np.linalg.lstsq(jac, -g, rcond=None)
            v0 = v0 + dv

            # 迭代后期只在上次穿越时刻附近搜索（穿越时刻随 v0 收敛而稳定）
            window = (
                t_star - self.window_days * _SECONDS_PER_DAY,
                t_star + self.window_days * _SECONDS_PER_DAY,
            )

        if g_norm >= self.tolerance:
            return None  # 未收敛：视为本弧段无法满足特征点约束

        return v0 - state0[3:]
