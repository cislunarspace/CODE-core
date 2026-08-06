"""
庞加莱截面工具模块

提供平面截面与近拱点截面的定义，以及两类截面穿越检测：

- **积分中检测**：``PoincareSection.event()`` 生成事件函数
  ``g(t, state) -> float``（携带 ``direction``/``terminal`` 属性），传给
  ``Dynamics.propagate(events=...)``，由积分器在步内定位穿越，无需密采样。
- **事后检测**：``crossings``/``detect_crossings``，传播时密采样
  t_eval → 逐采样点求截面函数 s(t)（平面：state[axis]-value；近拱点：
  r·v，r 为相对中心天体的位置）→ 符号变化区间内对线性插值态用二分
  法求精，穿越态残差可达 1e-10 以下。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..dynamics.cr3bp_system import CR3BP_System
    from .manifolds import ManifoldTube


@dataclass
class SectionCrossings:
    """截面穿越结果容器。

    Attributes:
        section: 产生穿越的截面对象
        states: 穿越态数组，形状 (k, 6)（插值求精后）
        times: 穿越时刻数组，形状 (k,)
        trajectory_index: 每个穿越点所属的流形弧索引，形状 (k,)
    """

    section: PoincareSection
    states: np.ndarray
    times: np.ndarray
    trajectory_index: np.ndarray


def _bisect_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-14,
    max_iter: int = 50,
) -> float:
    """纯 Python 二分求根。

    前提：``f(a)`` 与 ``f(b)`` 异号或其一为零（调用方
    :func:`detect_crossings` 已保证截面函数在采样区间端点变号）。支持
    时间递增或递减采样（``a`` 与 ``b`` 任意大小关系）。
    bisection 线性收敛，50 次迭代区间缩至 2⁻⁵⁰ ≈ 9e-16，远超截面残差
    1e-10 的需求；求精精度瓶颈是分段线性插值误差（~h²·|ÿ|/8），非求根
    器本身。
    """

    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    # 归一化为 a < b，使区间长度恒正（detect_crossings 的采样时间可递减）
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0 or (b - a) * 0.5 < xtol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def _refine_crossing(
    t0: float,
    t1: float,
    y0: np.ndarray,
    y1: np.ndarray,
    section_fn: Callable[[np.ndarray], float],
) -> tuple[float, np.ndarray]:
    """在符号变化区间 [t0, t1] 内对分段线性插值态用二分法求精穿越点"""

    def interp_state(t: float) -> np.ndarray:
        w = (t - t0) / (t1 - t0)
        return y0 + w * (y1 - y0)

    t_cross = _bisect_root(
        lambda t: section_fn(interp_state(t)),
        t0,
        t1,
        xtol=1e-14,
    )
    return t_cross, interp_state(t_cross)


def detect_crossings(
    times: np.ndarray,
    states: np.ndarray,
    section_fn: Callable[[np.ndarray], float],
) -> list[tuple[float, np.ndarray, int]]:
    """事后截面穿越检测

    逐采样点求截面函数，符号变化区间内对分段线性插值态用二分法求精。
    正向与反向（时间递减）积分均适用。

    Args:
        times: 采样时刻，形状 (n,)，单调（递增或递减）
        states: 采样状态，形状 (n, 6)
        section_fn: 截面函数，输入 6 维状态返回标量，零点即截面

    Returns:
        穿越列表，每项为 (穿越时刻, 穿越态, 区间左端采样索引)
    """
    values = np.array([section_fn(state) for state in states])
    crossings: list[tuple[float, np.ndarray, int]] = []

    for i in range(len(values) - 1):
        s0, s1 = values[i], values[i + 1]
        if s0 == 0.0 or s0 * s1 > 0:
            continue

        t_cross, state_cross = _refine_crossing(
            times[i], times[i + 1], states[i], states[i + 1], section_fn
        )
        crossings.append((t_cross, state_cross, i))

    return crossings


class PoincareSection:
    """庞加莱截面

    截面由标量函数 s(state) 的零等值面定义。提供两类常用构造：
    平面截面（某一坐标分量等于给定值）与近拱点截面（相对中心天体
    位置 r 与速度 v 的点积为零，即 r·v = 0）。
    """

    def __init__(self, section_fn: Callable[[np.ndarray], float], description: str = "") -> None:
        self._section_fn = section_fn
        self.description = description

    def __call__(self, state: np.ndarray) -> float:
        """求截面函数在给定状态处的值"""
        return float(self._section_fn(np.asarray(state, dtype=float)))

    def event(
        self, direction: int = 0, terminal: bool = False
    ) -> Callable[[float, np.ndarray], float]:
        """生成事件函数，供积分中检测使用

        返回的 callable ``g(t, state) -> float`` 可直接传给
        ``Dynamics.propagate(events=...)``；函数对象携带
        ``direction``/``terminal`` 属性。截面函数只依赖前 6 维物理状态，
        STM 增广传播（42 维状态）时自动截取前 6 维。

        Args:
            direction: 穿越方向过滤。> 0 只记上行穿越（s 由负到正），
                < 0 只记下行，0 双向
            terminal: True 时首次触发即终止积分

        Returns:
            事件函数，零点即截面
        """

        def g(t: float, state: np.ndarray) -> float:
            return self(np.asarray(state, dtype=float)[:6])

        g.direction = direction  # type: ignore[attr-defined]
        g.terminal = terminal  # type: ignore[attr-defined]
        return g

    @classmethod
    def plane(cls, axis: int, value: float) -> PoincareSection:
        """构造平面截面 s = state[axis] - value

        Args:
            axis: 状态分量索引（0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz）
            value: 平面取值

        Returns:
            PoincareSection 实例
        """
        if not 0 <= axis < 6:
            raise ValueError(f"axis 必须在 [0, 6) 内，当前为 {axis}")
        return cls(
            lambda state: state[axis] - value,
            description=f"plane(axis={axis}, value={value})",
        )

    @classmethod
    def periapsis(cls, center: str, system: CR3BP_System) -> PoincareSection:
        """构造近拱点截面 s = r·v

        r 为相对 center 天体的位置。CR3BP 会合系中主天体固定于 x=-μ，
        次天体固定于 x=1-μ。center 与 system 的主/次天体名称
        （不区分大小写）匹配，"earth"/"moon" 分别回退为主/次天体。

        Args:
            center: 中心天体名称（如 "earth" / "moon"）
            system: CR3BP_System 对象（提供 mu 与天体名称）

        Returns:
            PoincareSection 实例
        """
        mu = system.mu
        name = center.lower()
        primary = str(getattr(system, "primary_body", "")).lower()
        secondary = str(getattr(system, "secondary_body", "")).lower()

        if name in (primary, "earth"):
            center_pos = np.array([-mu, 0.0, 0.0])
        elif name in (secondary, "moon"):
            center_pos = np.array([1.0 - mu, 0.0, 0.0])
        else:
            raise ValueError(
                f"无法识别的中心天体: {center}，"
                f"须为 {system.primary_body} 或 {system.secondary_body}"
            )

        return cls(
            lambda state: float(np.dot(state[:3] - center_pos, state[3:])),
            description=f"periapsis(center={center})",
        )

    def crossings(self, tube: ManifoldTube) -> SectionCrossings:
        """检测流形管中所有流形弧的截面穿越

        Args:
            tube: 流形管（或任何带 ``trajectories`` 列表的容器，
                每条轨迹须含 ``times`` 与 ``states``）

        Returns:
            SectionCrossings: 全部穿越点（插值求精后）
        """
        all_states: list[np.ndarray] = []
        all_times: list[float] = []
        all_indices: list[int] = []

        for k, arc in enumerate(tube.trajectories):
            for t_cross, state_cross, _ in detect_crossings(arc.times, arc.states, self):
                all_states.append(state_cross)
                all_times.append(t_cross)
                all_indices.append(k)

        states = np.array(all_states).reshape(-1, 6) if all_states else np.empty((0, 6))
        return SectionCrossings(
            section=self,
            states=states,
            times=np.array(all_times, dtype=float),
            trajectory_index=np.array(all_indices, dtype=int),
        )

    def __str__(self):
        return f"PoincareSection({self.description})"

    def __repr__(self):
        return f"PoincareSection(description={self.description!r})"
