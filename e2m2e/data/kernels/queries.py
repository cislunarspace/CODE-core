"""SPICEManager 的星历查询 mixin：时间转换、天体状态/位置、帧旋转与 GM。

从 manager.py 逐字迁出的无状态查询面（#751）：utc_to_et/utc_to_tdb/
et_to_utc、get_body_state/body_state、get_body_position/body_position、
pxform、get_gm（其中多数实现 EphemerisProvider 的对应接口方法）。
SPICE 调用一律经 ``_spice_loader.get_spiceypy`` 惰性加载，不经
manager——本模块被 manager 在运行时导入作基类，反向导入会成环，
**运行时不得 import manager**。

本 mixin 不持有簿记与锁（双侧 furnsh/unload、datum 簿记与告警归
manager，ADR 0048）：实例态仅 ``_ephem_cache``（缓存拦截位置/状态
查询）与 ``_gm_fallback_warned``（回退告警去重），均由
``SPICEManager.__init__`` 创建；当前 GM 口径 ``ephemeris_datum`` 是
SPICEManager 上的只读 property，mixin 只声明依赖。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.constants.bodies import _BODIES_BY_NAME
from ._spice_loader import get_spiceypy
from .registry import _DEFAULT_EPHEMERIS_DATUM, _NAIF_IDS

if TYPE_CHECKING:
    from .ephem_cache import EphemCache

_logger = logging.getLogger(__name__)


class KernelQueryMixin:
    """SPICE 查询面 mixin：时间/状态/帧/GM 查询。

    供 ``class SPICEManager(KernelQueryMixin, EphemerisProvider)`` 组入。
    mixin 必须排在 EphemerisProvider 之前：MRO 自左向右解析，顺序颠倒
    会让 provider 的 NotImplementedError 桩遮蔽本 mixin 的全部实现。
    """

    # ---- 实例态声明（值由 SPICEManager.__init__ 创建，此处仅声明类型）----
    #: 预插值星历缓存；None 表示未启用（构建/开关在 manager）。
    _ephem_cache: EphemCache | None
    #: 已就 (body, 请求 datum, 实际 datum) 回退告警过的组合：每组合一次（实例级）。
    _gm_fallback_warned: set[tuple[str, str, str]]

    @property
    def ephemeris_datum(self) -> str:
        """当前 GM 基准（声明位：实际实现为 SPICEManager.ephemeris_datum）。"""
        raise NotImplementedError  # pragma: no cover - 由 SPICEManager 覆写

    # ---- EphemerisProvider 时间方法 ----

    def utc_to_et(self, utc_str: str) -> float:
        """将 UTC 时间字符串转换为 Ephemeris Time（TDB 秒）。

        SPICE 的 ET 即 TDB 时间尺度（ADR 0015：TDB 作动力学统一时间）。
        """
        return float(get_spiceypy().str2et(utc_str))

    def utc_to_tdb(self, utc: str) -> float:
        """UTC → TDB（ET 秒）。同 :meth:`utc_to_et`。"""
        return self.utc_to_et(utc)

    def et_to_utc(self, et: float) -> str:
        """将 Ephemeris Time（TDB 秒）转换为 UTC 时间字符串。"""
        return str(get_spiceypy().et2utc(et, "ISOC", 0))

    # ---- EphemerisProvider 状态方法 ----

    def get_body_state(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        """查询目标天体相对于观察者的状态向量（位置 + 速度）。"""
        if self._ephem_cache is not None and self._ephem_cache.covers(target, et, frame, observer):
            return self._ephem_cache.get_body_state(target, et)
        state, _lt = get_spiceypy().spkezr(target, et, frame, "NONE", observer)
        return np.array(state)

    def body_state(
        self, body: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> npt.NDArray[np.floating]:
        """EphemerisProvider 接口：天体状态（6,）。同 :meth:`get_body_state`。"""
        return self.get_body_state(body, et, frame, observer)

    def get_body_position(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        """查询目标天体相对于观察者的位置向量。"""
        if self._ephem_cache is not None and self._ephem_cache.covers(target, et, frame, observer):
            return self._ephem_cache.get_body_position(target, et)
        position, _lt = get_spiceypy().spkpos(target, et, frame, "NONE", observer)
        return np.array(position)

    def body_position(
        self, body: str, et: float, frame: str = "J2000", observer: str = "EARTH"
    ) -> npt.NDArray[np.floating]:
        """EphemerisProvider 接口：天体位置（3,）。同 :meth:`get_body_position`。"""
        return self.get_body_position(body, et, frame, observer)

    def pxform(self, frame_from: str, frame_to: str, et: float) -> npt.NDArray[np.floating]:
        """SPICE 帧旋转矩阵（EphemerisProvider 帧方法）。"""
        return np.array(get_spiceypy().pxform(frame_from, frame_to, et))

    # ---- GM 查询（ADR 0048 口径配对）----

    def get_gm(self, body: str, datum: str | None = None) -> float:
        """获取天体的引力参数 GM（km³/s²）。

        GM 基准默认取 :attr:`ephemeris_datum`（跟随已加载星历内核，ADR 0048）；
        可用 ``datum`` 显式覆盖。该基准下无记录时回退 DE440 并按
        (天体, 基准) 组合告警一次——不静默混用口径。

        若天体不在 ``data.constants.bodies`` 的 GM 表中（如未收录的小天体），
        则通过 SPICE 内核实时读取（原始行为不变）。

        Args:
            body: 天体名称（大小写不敏感）。
            datum: 显式 GM 基准（如 ``"DE421"``/``"DE440"``）；None 用当前口径。
        """
        requested = datum.upper() if datum is not None else self.ephemeris_datum
        name_upper = body.upper()
        body_obj = _BODIES_BY_NAME.get(name_upper)
        if body_obj is not None:
            if requested in body_obj.gm_by_datum:
                return body_obj.gm_by_datum[requested]
            if _DEFAULT_EPHEMERIS_DATUM in body_obj.gm_by_datum:
                key = (name_upper, requested, _DEFAULT_EPHEMERIS_DATUM)
                if key not in self._gm_fallback_warned:
                    self._gm_fallback_warned.add(key)
                    _logger.warning(
                        "天体 %s 无 %s 基准 GM，回退 %s 值：位置与 GM 口径不一致。"
                        "需按 %s 口径复算时请先补齐该基准的权威 GM（见 ADR 0048）。",
                        name_upper,
                        requested,
                        _DEFAULT_EPHEMERIS_DATUM,
                        requested,
                    )
                return body_obj.gm_by_datum[_DEFAULT_EPHEMERIS_DATUM]
        body_id = _NAIF_IDS.get(name_upper, body)
        vals = get_spiceypy().bodvrd(str(body_id), "GM", 1)
        return float(vals[1][0])
