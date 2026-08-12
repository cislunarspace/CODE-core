"""星历数据提供者抽象：对上层屏蔽数据来源（SPICE/r2s2）。

ADR 0015：时间尺度并入 EphemerisProvider（不单独 TimeSystem 类）。三类
方法——时间（utc_to_tdb/et_to_utc/utc_to_tai/tai_to_tt/tt_to_tdb/
jd_tdb_to_et）、状态（body_position/body_state/body_rotation）、帧
（pxform），单点 + 批量。

实现：
- SPICE 实现 = :class:`SPICEManager` （``data/kernels/manager.py``）；
- r2s2 实现 = ``data/frames/r2s2.py`` 的适配器（句柄管理 + TT↔TDB）。

批量方法（body_states/body_positions 等）为后续 Rust 注入数据预留接口，
当前为占位（ADR 0011：未实现能力保留占位 + NotImplementedError）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["EphemerisProvider"]


class EphemerisProvider:
    """星历数据提供者：对上层屏蔽数据来源。

    单点 + 批量两类方法；时间（utc_to_tdb/et_to_utc/utc_to_tai/tai_to_tt/
    tt_to_tdb/jd_tdb_to_et）、状态（body_position/body_state/body_rotation）、
    帧（pxform）三类。SPICE 和 r2s2 分别实现；Rust 侧"注入数据"（星历缓存
    样条表）从批量查询构建。
    """

    # ---- 时间（TDB 作动力学统一时间，ADR 0015）----

    def utc_to_tdb(self, utc: str) -> float:
        """UTC → TDB（ET 秒）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def et_to_utc(self, et: float) -> str:
        """TDB（ET 秒）→ UTC。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def utc_to_tai(self, utc: str) -> float:
        """UTC → TAI（秒）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def tai_to_tt(self, tai: float) -> float:
        """TAI → TT（秒）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def tt_to_tdb(self, tt: float) -> float:
        """TT → TDB（秒）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def jd_tdb_to_et(self, jd_tdb: float) -> float:
        """JD(TDB) → ET 秒。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    # ---- 状态 ----

    def body_position(
        self, body: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> npt.NDArray[np.floating]:
        """天体位置（km，形状 (3,)）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def body_state(
        self, body: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> npt.NDArray[np.floating]:
        """天体状态（km, km/s，形状 (6,)）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    def body_rotation(
        self, body: str, et: float, frame_from: str, frame_to: str
    ) -> npt.NDArray[np.floating]:
        """天体本体固定帧旋转矩阵（3×3）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写

    # ---- 帧 ----

    def pxform(self, frame_from: str, frame_to: str, et: float) -> npt.NDArray[np.floating]:
        """帧旋转矩阵（3×3）。"""
        raise NotImplementedError  # pragma: no cover - 由实现类覆写
