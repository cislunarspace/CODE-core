"""预插值星历缓存 — 积分前批量预采样天体位置，避免逐步调 SPICE。

ForceModel 满配直推的性能瓶颈在于每步每力调
``SPICEManager.get_body_position`` / ``get_body_state``——无缓存、每次跨
Python↔C 边界查 SPICE。本模块在积分前用 SPICE 在均匀网格上预采样所有
相关天体的 (R, V)，建 ``scipy.interpolate.CubicSpline`` （C² 连续），之后
查询走样条插值（纯数值）。源：``core/ephem_cache.py`` （ADR 0011 迁移，
数据层自足）。

为什么用三次样条而不是线性插值：线性插值 C⁰ 连续，网格点处导数不连续，
会让自适应积分器的误差估计器不断缩步长（实测 93 倍 RHS 调用）。三次样条
C² 连续消除此问题（经验来自 qiao 仓库
``Python/crtbp/subfunction/ephfunc/ephem_table.py:10-12``）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from .manager import SPICEManager

# 默认网格步长（秒）。qiao 经验值每小时一点。月球轨道周期 ~27 天，1 小时
# 间隔下三次样条位置误差 < 1 km。
DEFAULT_DT_SECONDS: float = 3600.0

# 两端外推 margin（倍数）。积分器可能在边界附近因步长越界查到端点外，
# 留 margin 避免外推。
_DEFAULT_MARGIN_STEPS: int = 5


class EphemCache:
    """预插值星历缓存。

    持有多个天体在 ``[et_start, et_end]`` 上的 (位置, 速度) 三次样条。
    查询时纯数值插值，不碰 SPICE。

    Attributes:
        bodies: 缓存覆盖的天体名列表（大写）。
        et_start, et_end: 缓存覆盖的 ET 秒范围（含 margin）。
        frame: 采样时用的坐标系（查询必须匹配）。
        observer: 采样时用的观察者（查询必须匹配）。
    """

    def __init__(
        self,
        bodies: list[str],
        pos_splines: dict[str, CubicSpline],
        vel_splines: dict[str, CubicSpline],
        et_start: float,
        et_end: float,
        frame: str,
        observer: str,
    ) -> None:
        self.bodies = [b.upper() for b in bodies]
        self._pos_splines = {k.upper(): v for k, v in pos_splines.items()}
        self._vel_splines = {k.upper(): v for k, v in vel_splines.items()}
        self.et_start = et_start
        self.et_end = et_end
        self.frame = frame
        self.observer = observer.upper()

    def covers(self, body: str, et: float, frame: str, observer: str) -> bool:
        """查询是否在缓存覆盖范围内（天体名 + 时间 + 坐标系都匹配）。"""
        return (
            body.upper() in self._pos_splines
            and self.et_start <= et <= self.et_end
            and frame == self.frame
            and observer.upper() == self.observer
        )

    def get_body_position(self, body: str, et: float) -> npt.NDArray[np.floating]:
        """插值查询天体位置 (3,) km。调用前应先 covers() 判定。"""
        return np.asarray(self._pos_splines[body.upper()](et), dtype=float)

    def get_body_state(self, body: str, et: float) -> npt.NDArray[np.floating]:
        """插值查询天体状态 (6,) [km, km/s]。调用前应先 covers() 判定。"""
        pos = self._pos_splines[body.upper()](et)
        vel = self._vel_splines[body.upper()](et)
        return np.concatenate([pos, vel])


def build_ephem_cache(
    spice: SPICEManager,
    bodies: list[str],
    et_start: float,
    et_end: float,
    *,
    dt: float = DEFAULT_DT_SECONDS,
    frame: str = "J2000",
    observer: str = "EARTH",
    margin_steps: int = _DEFAULT_MARGIN_STEPS,
) -> EphemCache:
    """构建预插值星历缓存。

    在 ``[et_start, et_end]`` （加两端 margin）上以 ``dt`` 步长均匀采样，
    对每个天体在每个网格点调 ``spice.get_body_state`` （此时走 SPICE），
    建 CubicSpline。
    """
    if et_end <= et_start:
        raise ValueError(f"et_end ({et_end}) 必须大于 et_start ({et_start})")
    if dt <= 0:
        raise ValueError(f"dt ({dt}) 必须为正")

    bodies_up = [b.upper() for b in bodies]
    margin = margin_steps * dt
    t0 = et_start - margin
    t1 = et_end + margin

    n = int(np.ceil((t1 - t0) / dt)) + 1
    t_grid = t0 + np.arange(n) * dt
    if t_grid[-1] > t1:
        t_grid[-1] = t1

    pos_grids: dict[str, np.ndarray] = {b: np.empty((n, 3)) for b in bodies_up}
    vel_grids: dict[str, np.ndarray] = {b: np.empty((n, 3)) for b in bodies_up}

    for i, t in enumerate(t_grid):
        for b in bodies_up:
            st = spice.get_body_state(b, float(t), frame, observer)
            pos_grids[b][i] = st[:3]
            vel_grids[b][i] = st[3:6]

    pos_splines = {b: CubicSpline(t_grid, pos_grids[b], bc_type="natural") for b in bodies_up}
    vel_splines = {b: CubicSpline(t_grid, vel_grids[b], bc_type="natural") for b in bodies_up}

    return EphemCache(
        bodies=bodies_up,
        pos_splines=pos_splines,
        vel_splines=vel_splines,
        et_start=float(t_grid[0]),
        et_end=float(t_grid[-1]),
        frame=frame,
        observer=observer,
    )
