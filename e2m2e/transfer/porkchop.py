"""porkchop 图扫描：二体 Lambert 双脉冲 ΔV 网格。

对出发时间 × 飞行时间网格逐点解 Lambert 问题，得到出发/到达脉冲
及其总和的网格数据（porkchop 图的数据层）。终端（出发/到达天体或
轨道）状态经 :class:`~e2m2e.transfer.terminal.TerminalCondition`
接口提取，本模块不关心状态如何产生。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .lambert import solve_lambert_batch
from .terminal import TerminalCondition

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..core.dynamics import CR3BP_Dynamics


@dataclass
class PorkchopData:
    """porkchop 扫描结果。

    Attributes:
        t_dep: 出发时间网格，形状 ``(n,)``
        tof: 飞行时间网格，形状 ``(m,)``
        dv1: 出发脉冲大小，形状 ``(n, m)``，km/s
        dv2: 到达脉冲大小，形状 ``(n, m)``，km/s
        total: 总脉冲 ``dv1 + dv2``，形状 ``(n, m)``，km/s
    """

    t_dep: np.ndarray
    tof: np.ndarray
    dv1: np.ndarray
    dv2: np.ndarray
    total: np.ndarray

    def plot(self, ax: Axes | None = None, levels: int | npt.ArrayLike | None = None) -> Axes:
        """画总 ΔV 等值线图（porkchop 图）。

        Args:
            ax: 目标坐标轴，None 时新建
            levels: ``contour`` 的等值线层级，None 时取 20

        Returns:
            绘图所用的 Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if levels is None:
            levels = 20
        cs = ax.contour(self.t_dep, self.tof, self.total.T, levels=levels)
        ax.clabel(cs, fontsize=8)
        ax.set_xlabel("departure time")
        ax.set_ylabel("time of flight")
        return ax


def porkchop(
    dep: TerminalCondition,
    arr: TerminalCondition,
    t_dep_range: npt.ArrayLike,
    tof_range: npt.ArrayLike,
    mu: float,
    dynamics: CR3BP_Dynamics,
    direction: str = "short",
    revs: int = 0,
) -> PorkchopData:
    """扫描出发时间 × 飞行时间网格，计算双脉冲 ΔV。

    对网格点 ``(t_dep, tof)``：出发终端状态取 ``t_dep`` 时刻，到达终端
    状态取 ``t_dep + tof`` 时刻，解二体 Lambert 得转移速度，脉冲为转移
    速度与终端轨道速度之差。

    Args:
        dep: 出发终端（如 :class:`OrbitTerminal`）
        arr: 到达终端
        t_dep_range: 出发时间网格 ``(n,)``，与终端时间坐标一致
        tof_range: 飞行时间网格 ``(m,)``，s
        mu: 中心天体 GM，km³/s²
        dynamics: 传播对象，透传给终端的 ``get_arrival_state``
        direction: ``"short"`` 或 ``"long"``
        revs: 完整圈数

    Returns:
        :class:`PorkchopData`
    """
    t_dep = np.atleast_1d(np.asarray(t_dep_range, dtype=float))
    tof = np.atleast_1d(np.asarray(tof_range, dtype=float))
    n, m = t_dep.shape[0], tof.shape[0]

    # 出发端状态只依赖 t_dep，循环外取一次
    r0_grid = np.empty((n, 3))
    v_dep_grid = np.empty((n, 3))
    for i, td in enumerate(t_dep):
        r0_grid[i], v_dep_grid[i] = dep.get_arrival_state(float(td), dynamics)

    dv1 = np.full((n, m), np.nan)
    dv2 = np.full((n, m), np.nan)
    # 同一 tof 下各出发时刻的几何组成一列，逐列批量求解
    for j, t in enumerate(tof):
        rf_col = np.empty((n, 3))
        v_arr_col = np.empty((n, 3))
        for i, td in enumerate(t_dep):
            rf_col[i], v_arr_col[i] = arr.get_arrival_state(float(td + t), dynamics)
        velocities = solve_lambert_batch(
            r0_grid, rf_col, [t], mu, direction=direction, revs=revs
        )
        col = velocities[:, 0, :, :]  # (n, 2, 3)
        valid = ~np.isnan(col[:, 0, 0])
        dv1[valid, j] = np.linalg.norm(col[valid, 0, :] - v_dep_grid[valid], axis=1)
        dv2[valid, j] = np.linalg.norm(v_arr_col[valid] - col[valid, 1, :], axis=1)

    return PorkchopData(t_dep=t_dep, tof=tof, dv1=dv1, dv2=dv2, total=dv1 + dv2)
