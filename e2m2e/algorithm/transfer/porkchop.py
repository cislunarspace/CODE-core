"""porkchop 图扫描：二体 Lambert 双脉冲 ΔV 网格。

对出发时间 × 飞行时间网格逐点解 Lambert 问题，得到出发/到达脉冲
及其总和的网格数据（porkchop 图的数据层）。终端（出发/到达天体或
轨道）状态经 :class:`~e2m2e.algorithm.transfer.terminal.TerminalCondition`
接口提取，本模块不关心状态如何产生。

除内存网格与绘图外，本模块还提供两项任务分析能力（主题 8）：

- **SQLite 存档**：:meth:`PorkchopData.to_sqlite` / :meth:`PorkchopData.from_sqlite`
  把网格落盘为关系表，多次扫描可累积、可查询（stdlib sqlite3，无新增依赖）。
- **ΔV–TOF Pareto 前沿**：:func:`pareto_front` 用经典非支配排序（Deb 2002）
  从网格中提取 Pareto 前沿（Topputo 2013 双目标范式）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from e2m2e import integrators

from ..dynamics import CR3BP_Dynamics
from .lambert import _parse_direction
from .terminal import OrbitTerminal, StateTerminal, TerminalCondition

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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

    # ------------------------------------------------------------------
    # 持久化：SQLite 存档
    # ------------------------------------------------------------------

    def to_sqlite(
        self,
        path: str | Path,
        orbit_pair: str,
        *,
        direction: str = "",
        revs: int = 0,
        note: str = "",
    ) -> int:
        """把本网格写入 SQLite 解数据库，返回本次扫描的 ``scan_id``。

        库结构两张表：

        - ``scans(scan_id, orbit_pair, direction, revs, n_t_dep, n_tof,
          created_at, note)``：每次调用存一行元数据。
        - ``design_points(scan_id, t_dep, tof, dv1, dv2, total)``：网格展平
          成 ``n×m`` 行，NaN（无解组合）存为 NULL。

        同一文件多次调用以自增 ``scan_id`` 累积，形成可查询的转移代价
        数据库（主题 8）。使用 stdlib ``sqlite3``，不引入新依赖。

        Args:
            path: SQLite 文件路径；父目录自动创建。
            orbit_pair: 轨道对标识，如 ``"LEO->GEO"``。
            direction: Lambert 转移方向（``"short"``/``"long"``），记入元数据。
            revs: 完整圈数，记入元数据。
            note: 自由备注。

        Returns:
            本次扫描的 ``scan_id``。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n, m = self.total.shape

        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            cur.executescript(_SCHEMA_SQL)
            cur.execute(
                "INSERT INTO scans (orbit_pair, direction, revs, n_t_dep, n_tof,"
                " created_at, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    orbit_pair,
                    direction,
                    int(revs),
                    n,
                    m,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    note,
                ),
            )
            scan_id = cur.lastrowid
            assert scan_id is not None

            rows = []
            for i in range(n):
                for j in range(m):
                    rows.append(
                        (
                            scan_id,
                            float(self.t_dep[i]),
                            float(self.tof[j]),
                            _nan_to_none(self.dv1[i, j]),
                            _nan_to_none(self.dv2[i, j]),
                            _nan_to_none(self.total[i, j]),
                        )
                    )
            cur.executemany(
                "INSERT INTO design_points (scan_id, t_dep, tof, dv1, dv2, total)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return scan_id

    @classmethod
    def from_sqlite(cls, path: str | Path, scan_id: int) -> PorkchopData:
        """按 ``scan_id`` 从 SQLite 解数据库重建网格（NULL 还原为 NaN）。

        Args:
            path: SQLite 文件路径。
            scan_id: :meth:`to_sqlite` 返回的扫描编号。

        Returns:
            重建的 :class:`PorkchopData`，与写入时数值等价。

        Raises:
            ValueError: ``scan_id`` 不存在。
        """
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT n_t_dep, n_tof FROM scans WHERE scan_id = ?", (scan_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"scan_id {scan_id} 不存在于 {path}")
            n, m = int(row[0]), int(row[1])

            t_dep = np.empty(n)
            tof = np.empty(m)
            dv1 = np.full((n, m), np.nan)
            dv2 = np.full((n, m), np.nan)
            total = np.full((n, m), np.nan)
            # 行序即写入顺序（i 主序、j 次序），按 rowid 还原
            for idx, (_sid, td, tf, d1, d2, tt) in enumerate(
                cur.execute(
                    "SELECT scan_id, t_dep, tof, dv1, dv2, total FROM design_points"
                    " WHERE scan_id = ? ORDER BY rowid",
                    (scan_id,),
                )
            ):
                i, j = divmod(idx, m)
                t_dep[i] = td
                tof[j] = tf
                dv1[i, j] = d1 if d1 is not None else np.nan
                dv2[i, j] = d2 if d2 is not None else np.nan
                total[i, j] = tt if tt is not None else np.nan
        finally:
            conn.close()
        return cls(t_dep=t_dep, tof=tof, dv1=dv1, dv2=dv2, total=total)

    # ------------------------------------------------------------------
    # 插值代价查询
    # ------------------------------------------------------------------

    def query(self, t_dep: float, tof: float) -> float:
        """双线性插值查询转移代价（总 ΔV）。

        在规则网格上定位 ``(t_dep, tof)`` 所在的单元格，用四角点双线性
        插值估计 ``total``。若四角点含 NaN（无解组合），返回 NaN——调用方
        应检查返回值或先用 :func:`pareto_front` 筛选有效区域。

        对应规划文档「宋亮俊数据库的在线查询」（主题 8）：预计算网格 +
        双线性插值，替代逐点重算 Lambert。

        Args:
            t_dep: 出发时间（须在本网格 ``t_dep`` 范围内）
            tof: 飞行时间（须在本网格 ``tof`` 范围内）

        Returns:
            插值得到的总 ΔV（km/s），或 NaN（格点含无解组合）。

        Raises:
            ValueError: 查询点超出网格范围。
        """
        i, di = _grid_locate(self.t_dep, t_dep)
        j, dj = _grid_locate(self.tof, tof)

        # 双线性插值：f(i+di, j+dj) = (1-di)(1-dj)f(i,j) + di(1-dj)f(i+1,j)
        #                              + (1-di)dj f(i,j+1) + di·dj f(i+1,j+1)
        c00 = self.total[i, j]
        c10 = self.total[i + 1, j]
        c01 = self.total[i, j + 1]
        c11 = self.total[i + 1, j + 1]
        if np.isnan(c00) or np.isnan(c10) or np.isnan(c01) or np.isnan(c11):
            return float("nan")
        return float(
            (1 - di) * (1 - dj) * c00 + di * (1 - dj) * c10 + (1 - di) * dj * c01 + di * dj * c11
        )

    @classmethod
    def query_scan(cls, path: str | Path, scan_id: int, t_dep: float, tof: float) -> float:
        """从 SQLite 解数据库读网格并插值查询。

        等价于 ``cls.from_sqlite(path, scan_id).query(t_dep, tof)``。

        Args:
            path: SQLite 文件路径。
            scan_id: 扫描编号。
            t_dep, tof: 查询点（同 :meth:`query`）。

        Returns:
            插值得到的总 ΔV（km/s），或 NaN。
        """
        return cls.from_sqlite(path, scan_id).query(t_dep, tof)


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS scans (
    scan_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    orbit_pair TEXT NOT NULL,
    direction  TEXT NOT NULL DEFAULT '',
    revs       INTEGER NOT NULL DEFAULT 0,
    n_t_dep    INTEGER NOT NULL,
    n_tof      INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS design_points (
    scan_id INTEGER NOT NULL REFERENCES scans(scan_id),
    t_dep   REAL NOT NULL,
    tof     REAL NOT NULL,
    dv1     REAL,
    dv2     REAL,
    total   REAL
);
CREATE INDEX IF NOT EXISTS idx_points_scan ON design_points(scan_id);
CREATE INDEX IF NOT EXISTS idx_points_grid ON design_points(scan_id, t_dep, tof);
"""


def _grid_locate(axis: np.ndarray, x: float) -> tuple[int, float]:
    """在单调递增坐标轴上定位 x，返回 (i, dx) 使 x ∈ [axis[i], axis[i+1]]。

    dx ∈ [0, 1] 为归一化偏移。x 恰好等于末端点时返回 (n-2, 1.0)。
    """
    n = axis.shape[0]
    if n < 2:
        raise ValueError(f"坐标轴长度 {n} < 2，无法插值")
    x = float(x)
    if x < axis[0] or x > axis[-1]:
        raise ValueError(f"查询点 {x} 超出网格范围 [{axis[0]}, {axis[-1]}]")
    i = int(np.searchsorted(axis, x, side="right")) - 1
    i = min(i, n - 2)  # x == axis[-1] 时 i = n-1 → 钳到 n-2
    dx = (x - axis[i]) / (axis[i + 1] - axis[i])
    return i, dx


def _nan_to_none(x: float) -> float | None:
    """NaN → None（SQLite NULL）；有限值原样返回。"""
    v = float(x)
    return None if np.isnan(v) else v


# ----------------------------------------------------------------------
# Pareto 前沿：ΔV–TOF 双目标非支配排序
# ----------------------------------------------------------------------


@dataclass
class ParetoFront:
    """porkchop 网格的 Pareto 前沿（rank 0 非支配点集）。

    Attributes:
        t_dep: 前沿点的出发时间，形状 ``(k,)``
        tof: 前沿点的飞行时间，形状 ``(k,)``
        dv1: 前沿点的出发脉冲，形状 ``(k,)``，km/s
        dv2: 前沿点的到达脉冲，形状 ``(k,)``，km/s
        total: 前沿点的总脉冲，形状 ``(k,)``，km/s
        rank: 各前沿点的非支配层级，形状 ``(k,)``，前沿点恒为 0
    """

    t_dep: np.ndarray
    tof: np.ndarray
    dv1: np.ndarray
    dv2: np.ndarray
    total: np.ndarray
    rank: np.ndarray

    def plot(self, ax: Axes | None = None) -> Axes:
        """画 Pareto 前沿（TOF vs 总 ΔV 散点 + 连线）。

        Args:
            ax: 目标坐标轴，None 时新建

        Returns:
            绘图所用的 Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        order = np.argsort(self.tof)
        ax.plot(self.tof[order], self.total[order], "o-", markersize=4)
        ax.set_xlabel("time of flight")
        ax.set_ylabel("total ΔV (km/s)")
        return ax


def pareto_front(
    data: PorkchopData,
    *,
    objectives: tuple[str, str] = ("total", "tof"),
) -> ParetoFront:
    """从 porkchop 网格提取 ΔV–TOF Pareto 前沿（经典非支配排序）。

    点 A 支配点 B 当且仅当 A 在两个目标上都不劣于 B、且至少一个目标
    严格更优。前沿（rank 0）即不被任何有效点支配的点集。仅对有效点
    （``total`` 非 NaN）排序；NaN 组合不参与。

    默认目标为 ``min(total)`` 与 ``min(tof)`` （Topputo 2013 双目标范式）。
    也支持 ``("dv1", "tof")`` 等其它字段组合；字段名必须是
    :class:`PorkchopData` 的 ndarray 字段。

    复杂度 O(M·N²)，N 为有效点数。porkchop 网格通常数百至数千点，
    纯 Python 足够；万级以上解集再考虑下沉 Rust。

    Args:
        data: porkchop 扫描结果。
        objectives: 两个目标字段名（越小越优）。

    Returns:
        :class:`ParetoFront` （仅 rank 0 点）。

    Raises:
        ValueError: 目标字段名不存在，或无有效点。
    """
    n, m = data.total.shape
    flat = []
    for name in objectives:
        arr = getattr(data, name, None)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"目标字段 {name!r} 不存在于 PorkchopData")
        if name == "t_dep":
            flat.append(np.broadcast_to(arr[:, None], (n, m)).ravel())
        elif name == "tof":
            flat.append(np.broadcast_to(arr[None, :], (n, m)).ravel())
        else:
            flat.append(arr.ravel())
    obj_mat = np.column_stack(flat)  # (n*m, 2)

    total_flat = data.total.ravel()
    valid = ~np.isnan(total_flat)
    pts = obj_mat[valid]  # (k, 2)

    if pts.shape[0] == 0:
        raise ValueError("porkchop 网格无有效点（total 全为 NaN）")

    rank = _non_dominated_sort(pts)
    front_mask = rank == 0

    # 还原到网格字段
    valid_idx = np.flatnonzero(valid)
    front_idx = valid_idx[front_mask]
    t_dep_grid = np.broadcast_to(data.t_dep[:, None], (n, m))
    tof_grid = np.broadcast_to(data.tof[None, :], (n, m))
    return ParetoFront(
        t_dep=t_dep_grid.ravel()[front_idx],
        tof=tof_grid.ravel()[front_idx],
        dv1=data.dv1.ravel()[front_idx],
        dv2=data.dv2.ravel()[front_idx],
        total=data.total.ravel()[front_idx],
        rank=rank[front_mask],
    )


def _non_dominated_sort(pts: np.ndarray) -> np.ndarray:
    """经典非支配排序（Deb 2002），返回各点的 rank（0 = 前沿）。

    目标均为最小化。复杂度 O(M·N²)，M=2。
    """
    k = pts.shape[0]
    # domination_count[p] = 支配 p 的点数
    domination_count = np.zeros(k, dtype=int)
    # dominated[p] = 被 p 支配的点列表
    dominated: list[list[int]] = [[] for _ in range(k)]
    rank = np.full(k, -1, dtype=int)
    current_front: list[int] = []

    for p in range(k):
        for q in range(k):
            if p == q:
                continue
            if _dominates(pts[p], pts[q]):
                dominated[p].append(q)
            elif _dominates(pts[q], pts[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            rank[p] = 0
            current_front.append(p)

    r = 0
    while current_front:
        next_front: list[int] = []
        for p in current_front:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = r + 1
                    next_front.append(q)
        r += 1
        current_front = next_front
    return rank


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 支配 b：a 所有目标 ≤ b 且至少一个 < b（最小化）。"""
    return bool(np.all(a <= b) and np.any(a < b))


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

    数值网格评估（终端传播 + Lambert + ΔV 组装 + 分发）全部在 Rust
    （#446，ADR 0017 范式）；Python 只做问题构造与结果解释：

    - **规格路径**：两端均为内置终端（``OrbitTerminal``/``StateTerminal``
      且未被 monkeypatch）、涉及轨道终端时 dynamics 为未 patch 的
      ``CR3BP_Dynamics``——终端规格直接交给 Rust，轨道终端的传播在
      Rust 内逐点并行进行。
    - **协议路径**：其余情况（自定义 ``TerminalCondition`` 子类、终端
      方法被 patch、动力学不满足上条）由 Python 按 ``get_arrival_state``
      协议逐点提取状态网格，再交给同一 Rust 评估核。patch 语义因此
      保持有效。

    两条路径共用同一 Rust Lambert/ΔV 核，无 Python 数值回退；扩展缺失
    按 #378 抛 ``RustExtensionUnavailableError``。并行由 Rayon 执行，
    ``E2M2E_PORKCHOP_PARALLEL=0`` 可强制串行（与并行逐位一致）。

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

    integrators.require_rust_extension("porkchop_grid_py", "porkchop_grid_states_py")

    spec = _builtin_grid_spec(dep, arr, dynamics)
    if spec is not None:
        long_way = _parse_direction(direction)
        dv1_flat, dv2_flat = integrators.porkchop_grid_py(
            t_dep.tolist(),
            tof.tolist(),
            spec.dep.kind,
            spec.dep.state.tolist(),
            spec.dep.t0,
            spec.dep.period,
            spec.arr.kind,
            spec.arr.state.tolist(),
            spec.arr.t0,
            spec.arr.period,
            spec.mu_cr3bp,
            spec.rtol,
            spec.atol,
            spec.max_step,
            float(mu),
            long_way,
            int(revs),
        )
    else:
        # 保持原路径的错误优先级：无效轨道/终端先在状态提取时上抛，
        # direction 仅在即将调用 Lambert 前校验。
        dep_states, arr_states = _extract_state_grids(dep, arr, t_dep, tof, dynamics)
        long_way = _parse_direction(direction)
        dv1_flat, dv2_flat = integrators.porkchop_grid_states_py(
            dep_states.ravel().tolist(),
            arr_states.ravel().tolist(),
            tof.tolist(),
            float(mu),
            long_way,
            int(revs),
        )

    dv1 = np.asarray(dv1_flat, dtype=float).reshape(n, m)
    dv2 = np.asarray(dv2_flat, dtype=float).reshape(n, m)
    return PorkchopData(t_dep=t_dep, tof=tof, dv1=dv1, dv2=dv2, total=dv1 + dv2)


@dataclass
class _TerminalSpec:
    """内置终端的展平规格（传给 Rust 规格路径）。"""

    kind: str  # "orbit" | "state"
    state: np.ndarray  # (6,)：orbit 为首点状态，state 为固定状态
    t0: float
    period: float


@dataclass
class _BuiltinGridSpec:
    """规格路径的全部输入：两端内置终端规格 + CR3BP 传播配置。"""

    dep: _TerminalSpec
    arr: _TerminalSpec
    # 两端均为 state 终端时全为 None（无需传播）。
    mu_cr3bp: float | None
    rtol: float | None
    atol: float | None
    max_step: float | None


def _terminal_unpatched(term: TerminalCondition, cls: type) -> bool:
    """检测终端的 ``get_arrival_state`` 是否保持类的原始实现。

    Rust 规格路径不经过 Python 方法分发；若方法被 monkeypatch（类级或
    实例级），必须走协议路径让 patch 生效。判定依据 ``__qualname__``
    （类级 patch）与绑定方法的 ``__func__`` 身份（实例级 patch），
    与 ``search_parallel._geometry_methods_monkeypatched`` 同范式。
    """
    method = getattr(cls, "get_arrival_state", None)
    if getattr(method, "__qualname__", None) != f"{cls.__name__}.get_arrival_state":
        return False
    bound = getattr(term, "get_arrival_state", None)
    return getattr(bound, "__func__", None) is method


def _terminal_spec(term: TerminalCondition) -> _TerminalSpec | None:
    """提取内置终端规格；非内置类型、被 patch 或规格无效时返回 None。

    规格无效（如轨道无周期）返回 None 走协议路径，保留原
    ``propagate_orbit_state_at_time`` 的 ValueError 语义。
    """
    if type(term) is OrbitTerminal and _terminal_unpatched(term, OrbitTerminal):
        orbit = term.orbit
        if orbit.states.shape[0] < 1 or orbit.period is None or orbit.period <= 0:
            return None
        state0 = np.asarray(orbit.states[0], dtype=float)
        if state0.shape != (6,):
            return None
        return _TerminalSpec("orbit", state0, float(orbit.times[0]), float(orbit.period))
    if type(term) is StateTerminal and _terminal_unpatched(term, StateTerminal):
        state = np.asarray(term.state, dtype=float)
        if state.shape != (6,):
            return None
        return _TerminalSpec("state", state, 0.0, 0.0)
    return None


def _builtin_grid_spec(
    dep: TerminalCondition,
    arr: TerminalCondition,
    dynamics: CR3BP_Dynamics,
) -> _BuiltinGridSpec | None:
    """判断是否可走 Rust 规格路径；可以则返回全部输入规格。

    条件：两端均为内置终端且未 patch；涉及轨道终端时 dynamics 是
    未 patch 的 ``CR3BP_Dynamics``（星历/BCR4BP 动力学或自定义子类
    一律走协议路径，行为与原 Python 路径一致）。检测缝限于终端的
    ``get_arrival_state`` 与 dynamics 的 ``propagate_orbit_state_at_time``；
    更深的 ``propagate`` 实现是该公开协议内部细节，不作为 monkeypatch 缝。
    """
    dep_spec = _terminal_spec(dep)
    arr_spec = _terminal_spec(arr)
    if dep_spec is None or arr_spec is None:
        return None
    if "orbit" not in (dep_spec.kind, arr_spec.kind):
        return _BuiltinGridSpec(dep_spec, arr_spec, None, None, None, None)
    if type(dynamics) is not CR3BP_Dynamics:
        return None
    method = CR3BP_Dynamics.propagate_orbit_state_at_time
    if getattr(method, "__qualname__", None) != "CR3BP_Dynamics.propagate_orbit_state_at_time":
        return None
    bound = getattr(dynamics, "propagate_orbit_state_at_time", None)
    if getattr(bound, "__func__", None) is not method:
        return None
    return _BuiltinGridSpec(
        dep_spec,
        arr_spec,
        float(dynamics.system.mu),
        float(dynamics.rtol),
        float(dynamics.atol),
        float(dynamics.max_step),
    )


def _extract_state_grids(
    dep: TerminalCondition,
    arr: TerminalCondition,
    t_dep: np.ndarray,
    tof: np.ndarray,
    dynamics: CR3BP_Dynamics,
) -> tuple[np.ndarray, np.ndarray]:
    """按 ``get_arrival_state`` 协议提取出发/到达状态网格（协议路径）。

    返回 ``(dep_states, arr_states)``：``dep_states[i]`` 为 ``t_dep[i]``
    时刻出发状态，形状 ``(n, 6)``；``arr_states[i*m+j]`` 为
    ``t_dep[i] + tof[j]`` 时刻到达状态，形状 ``(n*m, 6)``（行优先）。
    """
    n, m = t_dep.shape[0], tof.shape[0]
    dep_states = np.empty((n, 6))
    for i, td in enumerate(t_dep):
        r, v = dep.get_arrival_state(float(td), dynamics)
        dep_states[i, :3] = r
        dep_states[i, 3:] = v
    arr_states = np.empty((n * m, 6))
    for i, td in enumerate(t_dep):
        for j, t in enumerate(tof):
            r, v = arr.get_arrival_state(float(td + t), dynamics)
            arr_states[i * m + j, :3] = r
            arr_states[i * m + j, 3:] = v
    return dep_states, arr_states
