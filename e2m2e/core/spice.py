"""SPICE 工具封装模块。

本模块对 spiceypy（NASA SPICE 工具包的 Python 绑定）进行二次封装，
提供天体星历查询、时间转换、以及内核文件管理等常用功能，
使其更易于在轨道设计流程中使用。

SPICE 内核文件说明
==================

SPICE 内核是 NASA NAIF 提供的数据文件，包含天体星历、姿态、时间转换等信息。
本模块使用两类内核：

1. **闰秒内核**（``.tls``）：提供 UTC ↔ ET 时间转换所需的闰秒表。
   模块会自动在 ``kernels/`` 目录和 ``SPICE_KERNEL_DIR`` 环境变量指定的路径中
   搜索并加载 ``.tls`` 文件，无需手动操作。

2. **星历内核**（``.bsp``）：包含天体位置/速度数据（如 JPL DE440）。
   需要手动加载，可通过 ``find_ephemeris_kernel()`` 搜索或 ``load_kernel()`` 加载。

支持的星历内核（按推荐优先级）：

    ============  ====================================
    文件名        说明
    ============  ====================================
    de440.bsp     JPL DE440（推荐，覆盖 1550–2650 年）
    de440s.bsp    JPL DE440 精简版（覆盖 1849–2150 年）
    de435.bsp     JPL DE435（覆盖 1550–2650 年）
    de438.bsp     JPL DE438（覆盖 1550–2650 年）
    ============  ====================================

内核文件获取：从 `NASA NAIF <https://naif.jpl.nasa.gov/naif/data.html>`_ 下载，
或设置 ``SPICE_KERNEL_DIR`` 环境变量指向已下载的内核目录。

典型用法
========

手动指定路径加载::

    from e2m2e.core import SPICEManager

    mgr = SPICEManager()
    mgr.load_kernel("path/to/de440.bsp")

    et = mgr.utc_to_et("2024-01-01T00:00:00")
    state = mgr.get_body_state("MOON", et, "J2000", "EARTH")
    mgr.unload_kernel("path/to/de440.bsp")

自动搜索内核文件::

    from e2m2e.core import SPICEManager

    mgr = SPICEManager()
    kernel = mgr.find_ephemeris_kernel("/path/to/kernels")
    mgr.load_kernel(kernel)

    # ... 使用完毕后卸载
    mgr.unload_kernel(kernel)
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import numpy.typing as npt
import spiceypy

# 常用天体的引力参数 GM（km³/s²）。
# 键名为 NAIF 标准天体名称的大写形式。
_GM_VALUES: Dict[str, float] = {
    "EARTH": 398600.436,
    "MOON": 4902.8,
    "SUN": 1.32712440018e11,
    "EMB": 398600.436,
}

# 常用天体的 NAIF ID 映射表。
# 用于将天体名称转换为 SPICE 所需的整数 ID。
_NAIF_IDS: Dict[str, int] = {
    "EARTH": 399,
    "MOON": 301,
    "SUN": 10,
    "EMB": 3,
}

# 闰秒内核（.tls 文件）的搜索路径列表。
# 按优先级依次搜索：项目内置 kernels 目录 → 环境变量 SPICE_KERNEL_DIR。
_LEAPSECOND_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "kernels"),
    os.environ.get("SPICE_KERNEL_DIR", ""),
]


def _find_leapseconds_kernel():
    """在预定义的搜索路径中查找闰秒内核文件（.tls）。

    Returns:
        str | None: 找到的闰秒内核文件的绝对路径；若未找到则返回 None。
    """
    for search_dir in _LEAPSECOND_SEARCH_PATHS:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".tls"):
                    return os.path.join(root, f)
    return None


class SPICEManager:
    """SPICE 内核管理器，统一管理内核加载与天体状态查询。

    负责自动加载闰秒内核、提供星历查询接口（位置/状态）、
    时间格式转换（UTC ↔ ET）以及天体引力参数查询。

    使用流程::

        mgr = SPICEManager()
        # 搜索并加载星历内核
        kernel = mgr.find_ephemeris_kernel("/path/to/kernels")
        mgr.load_kernel(kernel)

        # 查询天体状态
        et = mgr.utc_to_et("2025-06-21T11:00:00")
        state = mgr.get_body_state("MOON", et, "J2000", "EARTH")

        # 使用完毕后卸载
        mgr.unload_kernel(kernel)

    Attributes:
        _leapseconds_loaded: 标记闰秒内核是否已加载，避免重复加载。
    """

    def __init__(self) -> None:
        self._leapseconds_loaded = False

    def _ensure_leapseconds(self):
        """确保闰秒内核已加载。

        首次调用时会自动搜索并加载 .tls 闰秒内核文件，
        后续调用直接跳过。这是 load_kernel 的前置步骤。
        """
        if self._leapseconds_loaded:
            return
        path = _find_leapseconds_kernel()
        if path:
            spiceypy.furnsh(path)
            self._leapseconds_loaded = True

    def load_kernel(self, path: str) -> None:
        """加载一个 SPICE 内核文件（.bsp / .bpc / .tf 等）。

        加载前会自动确保闰秒内核已就绪。

        Args:
            path: 内核文件的路径。

        Raises:
            FileNotFoundError: 当指定路径的文件不存在时。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Kernel file not found: {path}")
        self._ensure_leapseconds()
        spiceypy.furnsh(path)

    def unload_kernel(self, path: str) -> None:
        """卸载一个已加载的 SPICE 内核文件，释放相关资源。

        Args:
            path: 之前通过 load_kernel 加载的内核文件路径。
        """
        spiceypy.unload(path)

    def utc_to_et(self, utc_str: str) -> float:
        """将 UTC 时间字符串转换为 Ephemeris Time（历书时，单位秒）。

        Args:
            utc_str: ISO 格式的 UTC 时间字符串，如 ``"2024-01-01T00:00:00"``。

        Returns:
            对应的 ET 值（秒）。
        """
        return float(spiceypy.str2et(utc_str))

    def et_to_utc(self, et: float) -> str:
        """将 Ephemeris Time（历书时）转换为 UTC 时间字符串。

        Args:
            et: 历书时（秒）。

        Returns:
            ISO 格式的 UTC 时间字符串。
        """
        return spiceypy.et2utc(et, "ISOC", 0)

    def get_body_state(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        """查询目标天体相对于观察者的状态向量（位置 + 速度）。

        Args:
            target: 目标天体名称或 NAIF ID，如 ``"MOON"``。
            et: 历书时（秒）。
            frame: 参考坐标系名称，如 ``"J2000"``、``"ECLIPJ2000"``。
            observer: 观察者天体名称或 NAIF ID，如 ``"EARTH"``。

        Returns:
            长度为 6 的 NumPy 数组，前 3 个元素为位置 (km)，后 3 个为速度 (km/s)。
        """
        state, _lt = spiceypy.spkezr(target, et, frame, "NONE", observer)
        return np.array(state)

    def get_body_position(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        """查询目标天体相对于观察者的位置向量。

        Args:
            target: 目标天体名称或 NAIF ID。
            et: 历书时（秒）。
            frame: 参考坐标系名称。
            observer: 观察者天体名称或 NAIF ID。

        Returns:
            长度为 3 的 NumPy 数组，表示位置 (km)。
        """
        position, _lt = spiceypy.spkpos(target, et, frame, "NONE", observer)
        return np.array(position)

    _EPHEMERIS_KERNEL_PRIORITY = ["de440.bsp", "de440s.bsp", "de435.bsp", "de438.bsp"]

    def find_ephemeris_kernel(self, search_dir: str) -> str:
        """在指定目录中按优先级搜索星历内核文件（.bsp）。

        优先级：de440.bsp > de440s.bsp > de435.bsp > de438.bsp。

        Args:
            search_dir: 要搜索的目录路径。

        Returns:
            找到的第一个 .bsp 内核文件的绝对路径。

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

        Args:
            body: 天体名称（如 ``"EARTH"``、``"MOON"``）或 NAIF ID。

        Returns:
            该天体的 GM 值（km³/s²）。
        """
        name_upper = body.upper()
        if name_upper in _GM_VALUES:
            return _GM_VALUES[name_upper]
        body_id = _NAIF_IDS.get(name_upper, body)
        vals = spiceypy.bodvrd(body_id, "GM", 1)
        return float(vals[0][0])
