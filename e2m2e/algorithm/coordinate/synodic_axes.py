"""由 SPICE 月球瞬时状态驱动的会合（synodic）坐标轴。

该轴在 ICRF/J2000 下的基向量由 ``spice.get_body_state("MOON", et, "J2000", "EARTH")``
实时确定：
- e1：地月连线方向（指向月球）
- e3：瞬时轨道角动量方向
- e2：右手系补齐

约定 ``r_icrf = R @ r_axes``，``R = column_stack([e1, e2, e3])``。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .axes import Axes


class SynodicAxes(Axes):
    """SPICE 驱动的会合旋转坐标轴。"""

    # 数值微分步长 (秒)。1e-5 会严重吃有效数字并放大 SPICE 星历插值噪声
    # （Cdot 相对误差 ~1e-3）；1.0s 处于舍入-截断平衡平台，误差降到 ~1e-7。
    _DEFAULT_RATE_STEP = 1.0
    _CACHE_CAPACITY = 256  # R / Rdot 缓存容量

    def __init__(self, spice, cache_capacity: int = 256) -> None:
        self._spice = spice
        self._CACHE_CAPACITY = max(1, cache_capacity)
        self._rotation_cache: dict[float, npt.NDArray[np.floating]] = {}
        self._rate_cache: dict[float, npt.NDArray[np.floating]] = {}

    @staticmethod
    def _build_rotation_matrix(
        r_m: npt.NDArray[np.floating], v_m: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """由月球相对地球的位置/速度构造瞬时旋转矩阵。"""
        e1 = r_m / np.linalg.norm(r_m)
        h = np.cross(r_m, v_m)
        e3 = h / np.linalg.norm(h)
        e2 = np.cross(e3, e1)
        return np.column_stack([e1, e2, e3])

    def _evict_oldest(self) -> None:
        # 两个缓存的 key 集合并不同步（rotation_matrix 塞 et，rotation_and_rate
        # 还塞 et±step），需各自独立判容量与判空，否则对空 rate_cache 取
        # next(iter(...)) 会抛 StopIteration（step4 长期预报 876 点远超 256 容量）。
        if len(self._rotation_cache) >= self._CACHE_CAPACITY and self._rotation_cache:
            self._rotation_cache.pop(next(iter(self._rotation_cache)))
        if len(self._rate_cache) >= self._CACHE_CAPACITY and self._rate_cache:
            self._rate_cache.pop(next(iter(self._rate_cache)))

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        cached = self._rotation_cache.get(et)
        if cached is not None:
            return cached
        moon_state = self._spice.get_body_state("MOON", et, "J2000", "EARTH")
        rotation = self._build_rotation_matrix(moon_state[:3], moon_state[3:])
        self._evict_oldest()
        self._rotation_cache[et] = rotation
        return rotation

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        cached_rate = self._rate_cache.get(et)
        if cached_rate is not None:
            # rotation 可能已被 evict（容量 256），不能直接索引
            rotation = self.rotation_matrix(et)
            return rotation, cached_rate
        rotation = self.rotation_matrix(et)
        step = self._DEFAULT_RATE_STEP
        before = self.rotation_matrix(et - step)
        after = self.rotation_matrix(et + step)
        rate = (after - before) / (2.0 * step)
        self._evict_oldest()
        self._rate_cache[et] = rate
        return rotation, rate

    def characteristic_length(self, et: float) -> float:
        """返回当前时刻的地月距离 (km)。"""
        moon_state = self._spice.get_body_state("MOON", et, "J2000", "EARTH")
        return float(np.linalg.norm(moon_state[:3]))
