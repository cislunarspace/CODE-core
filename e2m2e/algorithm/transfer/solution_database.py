"""解数据库：多 scan 聚合查询与筛选（主题 8）。

在 porkchop SQLite 存档基础上提供统一查询接口：
- 多 scan 聚合（跨扫描合并查询）
- 双线性插值代价查询（复用 :meth:`PorkchopData.query`）
- 筛选钩子（预留 Grossi 式主矢量筛选接口）

用法::

    from e2m2e.transfer.solution_database import SolutionDatabase

    db = SolutionDatabase("porkchop.db")
    db.add_scan(data, orbit_pair="LEO->GEO", direction="short")
    dv = db.query(scan_id=1, t_dep=1000.0, tof=3600.0)
    front = db.pareto_front(scan_id=1)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .porkchop import ParetoFront, PorkchopData, pareto_front


@dataclass
class SolutionDatabase:
    """porkchop 解数据库（SQLite 后端）。

    封装多 scan 聚合查询，比裸用 :class:`PorkchopData` 的类方法
    更方便。不引入新依赖（stdlib sqlite3）。

    Attributes:
        path: SQLite 文件路径。
    """

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def add_scan(
        self,
        data: PorkchopData,
        orbit_pair: str,
        *,
        direction: str = "",
        revs: int = 0,
        note: str = "",
    ) -> int:
        """把一次 porkchop 扫描加入数据库，返回 scan_id。"""
        return data.to_sqlite(self.path, orbit_pair, direction=direction, revs=revs, note=note)

    def get_scan(self, scan_id: int) -> PorkchopData:
        """按 scan_id 取完整网格。"""
        return PorkchopData.from_sqlite(self.path, scan_id)

    def query(self, scan_id: int, t_dep: float, tof: float) -> float:
        """插值查询指定 scan 的转移代价。"""
        return PorkchopData.query_scan(self.path, scan_id, t_dep, tof)

    def pareto_front(
        self, scan_id: int, *, objectives: tuple[str, str] = ("total", "tof")
    ) -> ParetoFront:
        """提取指定 scan 的 Pareto 前沿。"""
        data = self.get_scan(scan_id)
        return pareto_front(data, objectives=objectives)

    def list_scans(self) -> list[dict]:
        """列出所有扫描的元数据。"""
        conn = sqlite3.connect(str(self.path))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT scan_id, orbit_pair, direction, revs, n_t_dep, n_tof,"
                " created_at, note FROM scans ORDER BY scan_id"
            )
            return [
                {
                    "scan_id": row[0],
                    "orbit_pair": row[1],
                    "direction": row[2],
                    "revs": row[3],
                    "n_t_dep": row[4],
                    "n_tof": row[5],
                    "created_at": row[6],
                    "note": row[7],
                }
                for row in cur.fetchall()
            ]
        finally:
            conn.close()

    def filter(
        self,
        scan_id: int,
        criterion: Callable[[float, float, float], bool],
    ) -> npt.NDArray[np.bool_]:
        """按判据筛选网格点（预留 Grossi 式主矢量筛选接口）。

        Args:
            scan_id: 扫描编号。
            criterion: 判据函数 ``f(t_dep, tof, total_dv) -> bool``，
                True 表示保留。

        Returns:
            布尔掩码数组，形状 ``(n, m)``，与网格同形。
        """
        data = self.get_scan(scan_id)
        n, m = data.total.shape
        mask = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                if not np.isnan(data.total[i, j]):
                    mask[i, j] = criterion(
                        float(data.t_dep[i]),
                        float(data.tof[j]),
                        float(data.total[i, j]),
                    )
        return mask

