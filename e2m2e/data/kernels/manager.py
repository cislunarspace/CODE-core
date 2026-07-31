"""SPICE 内核管理器：加载/缓存/校验。

数据层 SPICE 实现（ADR 0011 迁移，源：``core/spice.py``）。职责：内核
加载/卸载/缓存、UTC↔ET 时间转换、天体状态/位置查询、引力参数查询，
并实现 :class:`EphemerisProvider` 接口（时间/状态/帧三类，见
``provider.py``）。

SPICE 内核文件说明：

1. **闰秒内核**（``.tls``）：提供 UTC ↔ ET 时间转换所需的闰秒表。自动在
   ``kernels/`` 目录和 ``SPICE_KERNEL_DIR`` 环境变量指定的路径中搜索。
2. **星历内核**（``.bsp``）：包含天体位置/速度数据（如 JPL DE440）。
   需要手动加载，可通过 :meth:`SPICEManager.find_ephemeris_kernel` 搜索或
   :meth:`SPICEManager.load_kernel` 加载。

依赖方向：数据层只依赖外部库（numpy/spiceypy/scipy），不依赖 e2m2e 其他层。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._spice_loader import get_spiceypy
from .provider import EphemerisProvider

if TYPE_CHECKING:
    from .ephem_cache import EphemCache

# 常用天体的引力参数 GM（km³/s²）。
# 键名为 NAIF 标准天体名称的大写形式。
# 数值来源：JPL DE440 行星历表（TDB 时间框架）。
_GM_VALUES: dict[str, float] = {
    "SUN": 1.32712440018e11,
    "MERCURY": 22031.868551,
    "VENUS": 324858.592000,
    "EARTH": 398600.435507,
    "MOON": 4902.800118,
    "MARS": 42828.375816,
    "JUPITER": 126712764.100000,
    "SATURN": 37940584.841800,
    "URANUS": 5794556.400000,
    "NEPTUNE": 6836527.100580,
    "EMB": 403503.235502,  # 地月质心（Earth-Moon Barycenter）
    "PLUTO": 975.500000,  # 矮行星，供扩展使用
}

# 常用天体的 NAIF ID 映射表，用于将天体名称转换为 SPICE 所需的整数 ID。
_NAIF_IDS: dict[str, int] = {
    "SUN": 10,
    "MERCURY": 199,
    "VENUS": 299,
    "EARTH": 399,
    "MOON": 301,
    "MARS": 499,
    "JUPITER": 599,
    "SATURN": 699,
    "URANUS": 799,
    "NEPTUNE": 899,
    "EMB": 3,
    "PLUTO": 999,
}

# 闰秒内核（.tls 文件）的搜索路径列表。
# 按优先级依次搜索：项目内置 kernels 目录 → 环境变量 SPICE_KERNEL_DIR。
# 注意：data/kernels/ 比旧 core/ 深一级，仓库根用 parents[3]。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEAPSECOND_SEARCH_PATHS: list[str] = [
    str(_REPO_ROOT / "kernels"),
    os.environ.get("SPICE_KERNEL_DIR", ""),
]


def _find_leapseconds_kernel(search_paths: list[str] | None = None) -> str | None:
    """在预定义的搜索路径中查找闰秒内核文件（.tls）。"""
    paths = _LEAPSECOND_SEARCH_PATHS if search_paths is None else search_paths
    for search_dir in paths:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        for root, _dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".tls"):
                    return os.path.join(root, f)
    return None


class SPICEManager(EphemerisProvider):
    """SPICE 内核管理器：SPICE 星历数据提供者实现。

    统一管理内核加载与天体状态查询：自动加载闰秒内核、提供星历查询接口
    （位置/状态）、时间格式转换（UTC ↔ ET）以及天体引力参数查询。

    使用流程::

        mgr = SPICEManager()
        kernel = mgr.find_ephemeris_kernel("/path/to/kernels")
        mgr.load_kernel(kernel)
        et = mgr.utc_to_et("2025-06-21T11:00:00")
        state = mgr.get_body_state("MOON", et, "J2000", "EARTH")
        mgr.unload_kernel(kernel)

    Attributes:
        _leapseconds_loaded: 标记闰秒内核是否已加载，避免重复加载。
    """

    _leapseconds_loaded: bool = False
    _leapseconds_lock = threading.Lock()

    def __init__(self) -> None:
        """初始化 SPICE 管理器。"""
        # 预插值星历缓存（enable_ephem_cache 后生效；get_body_position/state
        # 优先走 cache，避免逐步跨 Python↔C 边界查 SPICE）。见 ephem_cache.py。
        self._ephem_cache: EphemCache | None = None

    def _ensure_leapseconds(self):
        """确保闰秒内核已加载（线程安全）。"""
        if SPICEManager._leapseconds_loaded:
            return
        with SPICEManager._leapseconds_lock:
            if SPICEManager._leapseconds_loaded:
                return
            path = _find_leapseconds_kernel()
            if path:
                get_spiceypy().furnsh(path)
                SPICEManager._leapseconds_loaded = True

    def load_kernel(self, path: str) -> None:
        """加载一个 SPICE 内核文件（.bsp / .bpc / .tf 等）。

        加载前会自动确保闰秒内核已就绪。

        Raises:
            FileNotFoundError: 当指定路径的文件不存在时。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Kernel file not found: {path}")
        self._ensure_leapseconds()
        get_spiceypy().furnsh(path)
        # Rust cspice 与 Python spiceypy 是独立 CSPICE 实例（静态链接，
        # 内核池不共享）。spice feature 启用时双 furnsh，让下沉到 Rust
        # 的力（ThirdBody/Indirect/...）也能查到。Rust 绑定是数值层，
        # 此处为内核加载的跨层桥接（ADR 0012 的 data/ → 仅外部库例外，
        # 迁移期保留，第 5 批评估归属）。
        try:
            from e2m2e._integrators import spice_poc_furnsh  # noqa: F401

            spice_poc_furnsh(path)
        except ImportError:
            pass

    def unload_kernel(self, path: str) -> None:
        """卸载一个已加载的 SPICE 内核文件，释放相关资源。"""
        get_spiceypy().unload(path)
        # Rust cspice 侧没有 unload 包装，依赖进程退出释放（cspice 0.1 限制）。

    def enable_ephem_cache(
        self,
        bodies: list[str],
        et_start: float,
        et_end: float,
        *,
        dt: float = 3600.0,
        frame: str = "J2000",
        observer: str = "EARTH",
    ) -> None:
        """构建并启用预插值星历缓存。"""
        from .ephem_cache import build_ephem_cache

        self._ephem_cache = build_ephem_cache(
            self,
            bodies,
            et_start,
            et_end,
            dt=dt,
            frame=frame,
            observer=observer,
        )

    def disable_ephem_cache(self) -> None:
        """关闭预插值星历缓存，回退到逐步 SPICE 查询。"""
        self._ephem_cache = None

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

    _EPHEMERIS_KERNEL_PRIORITY = ["de440.bsp", "de440s.bsp", "de435.bsp", "de438.bsp"]

    def find_ephemeris_kernel(self, search_dir: str) -> str:
        """在指定目录中按优先级搜索星历内核文件（.bsp）。

        优先级：de440.bsp > de440s.bsp > de435.bsp > de438.bsp。

        Raises:
            FileNotFoundError: 目录不存在或其中无匹配的内核文件。
        """
        if not os.path.isdir(search_dir):
            raise FileNotFoundError(
                f"Ephemeris kernel search directory does not exist: {search_dir}"
            )
        for candidate in self._EPHEMERIS_KERNEL_PRIORITY:
            path = os.path.join(search_dir, candidate)
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise FileNotFoundError(f"No ephemeris kernel found in {search_dir}")

    def get_gm(self, body: str) -> float:
        """获取天体的引力参数 GM（km³/s²）。

        优先从本地缓存字典中查找；若未命中，则通过 SPICE 内核实时读取。
        """
        name_upper = body.upper()
        if name_upper in _GM_VALUES:
            return _GM_VALUES[name_upper]
        body_id = _NAIF_IDS.get(name_upper, body)
        vals = get_spiceypy().bodvrd(str(body_id), "GM", 1)
        return float(vals[1][0])
