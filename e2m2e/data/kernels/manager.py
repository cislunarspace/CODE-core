"""SPICE 内核管理器：加载/缓存/校验。

数据层 SPICE 实现（ADR 0011 迁移，源：``core/spice.py``）。职责：内核
加载/卸载/缓存、UTC↔ET 时间转换、天体状态/位置查询、引力参数查询，
并实现 :class:`EphemerisProvider` 接口（时间/状态/帧三类，见
``provider.py``）。

SPICE 内核文件说明：

1. **闰秒内核** （``.tls``）：提供 UTC ↔ ET 时间转换所需的闰秒表。自动
   在被加载内核的同级目录、仓库内置 ``kernels/`` 目录、
   ``SPICE_KERNEL_DIR`` 环境变量指定的路径中按序搜索。
2. **星历内核** （``.bsp``）：包含天体位置/速度数据（如 JPL DE440）。
   需要手动加载，可通过 :meth:`SPICEManager.find_ephemeris_kernel` 搜索或
   :meth:`SPICEManager.load_kernel` 加载。

依赖方向：数据层只依赖外部库（numpy/spiceypy/scipy）与包根共享内核叶
（exceptions/spice_ext，ADR 0039）——SPICE 双实例桥接经 spice_ext 直达
Rust 扩展，不穿数值层门面。
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.constants.bodies import _BODIES_BY_NAME
from ._spice_loader import get_spiceypy
from .provider import EphemerisProvider

if TYPE_CHECKING:
    from .ephem_cache import EphemCache

_logger = logging.getLogger(__name__)

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

# 星历（SPK）内核 → 物理常数基准（datum）映射白名单。
#
# 依据（ADR 0048）：
# - **SPK 内核本身不携带 GM**（实测：de421.bsp + pck00010.tpc 下对所有天体
#   ``bodvrd(..., "GM")`` 均报 KERNELVARNOTFOUND），GM 只能来自
#   ``constants.toml`` 的声明式 body 表；故 datum 由内核**文件名**推断，
#   不是从内核内容读取。
# - 只有 ``de421.bsp`` 会改变 GM 口径：仓库只为 DE421/DE440 维护 GM 表。
#   de441/de442 的 GM 与 DE440 确有差异，但补全它们需要的权威来源不在本仓库
#   现有证据内，因此一律按 DE440 处理并告警一次，绝不静默混用。
# - 配对取"最后一个成功加载的星历内核"：与 SPICE 对重叠覆盖段"后加载者生效"
#   的优先级规则一致，避免出现"位置按 de440s、GM 按 de421"的静默错配。
_SPK_DATUM_PATTERN = re.compile(r"(?i)^(de\d+s?)\.bsp$")

_KERNEL_DATUM_BY_SPK: dict[str, str] = {
    "de421": "DE421",
    "de430": "DE440",
    "de435": "DE440",
    "de438": "DE440",
    "de440": "DE440",
    "de440s": "DE440",
    "de441": "DE440",
    "de442": "DE440",
    "de442s": "DE440",
}

#: 无星历内核（或内核未收录）时的 GM 基准，与 ADR 0022 决策 4 的星历动力学默认一致。
_DEFAULT_EPHEMERIS_DATUM = "DE440"

#: 白名单中 GM 被近似到其它基准的内核（ADR 0048 承认其 DE440 差异但无权威表）：
#: 加载时按内核名告警一次，落实「绝不静默混用」。
_APPROXIMATED_KERNELS: frozenset[str] = frozenset(
    {"de430", "de435", "de438", "de441", "de442", "de442s"}
)


def _spk_kernel_name(path: str) -> str | None:
    """星历内核名（小写、去扩展名）；非 DE 系列命名返回 ``None``。"""
    match = _SPK_DATUM_PATTERN.match(os.path.basename(path))
    return match.group(1).lower() if match is not None else None


def _datum_for_kernel(path: str) -> str | None:
    """由星历内核文件名推断 GM 基准；非星历内核或未收录者返回 ``None``。"""
    name = _spk_kernel_name(path)
    return _KERNEL_DATUM_BY_SPK.get(name) if name is not None else None


# 闰秒内核（.tls 文件）的搜索路径列表。
# 按优先级依次搜索：项目内置 kernels 目录 → 环境变量 SPICE_KERNEL_DIR。
# 注意：data/kernels/ 距仓库根三级，仓库根用 parents[3]。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEAPSECOND_SEARCH_PATHS: list[str] = [
    str(_REPO_ROOT / "kernels"),
    os.environ.get("SPICE_KERNEL_DIR", ""),
]

#: 行星名→质心/本体 NAIF ID 别名表。de440s/de430/de440 全本只含行星**质心**
#: 段 + 地球族本体 + 月球 + 太阳，不含行星本体段（499/599/…）；CSPICE 默认表
#: 把 "MARS" 解析成本体 499（de440s 不含）而非质心 4。本表把这些名字注册到
#: 质心/本体 ID，使 Python spiceypy 实例与 Rust cspice 实例（那边在
#: ``spice_ffi::register_bodies`` 注册同一份表）解析一致。
#:
#: 单一归属 SPICEManager 模块。两份表（Python 这里 +
#: Rust ``BODY_ALIASES``）保持一致，不做跨语言单源。
_BODY_ID_ALIASES: list[tuple[str, int]] = [
    ("MERCURY", 1),
    ("VENUS", 2),
    ("EARTH", 399),
    ("MARS", 4),
    ("JUPITER", 5),
    ("SATURN", 6),
    ("URANUS", 7),
    ("NEPTUNE", 8),
    ("MOON", 301),
    ("SUN", 10),
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


_STALE_BINARY_HINT = (
    "e2m2e._integrators 编译产物（.pyd/.so）落后于源码："
    "Rust 函数 {fn_name!r} 缺少关键字参数 {missing}。"
    "请重建 Rust 扩展：make dev（等价于 uv run maturin develop --release）"
)


def _call_rust_or_compat_error(
    fn,
    /,
    *args,
    fn_name: str,
    required_kwargs: tuple[str, ...],
    **kwargs,
):
    """调用 Rust pyfunction，把编译产物过期导致的签名漂移转成可操作错误。

    ``.pyd``/``.so`` 落后于源码时，PyO3 在参数绑定阶段抛 ``TypeError`` （如
    ``got an unexpected keyword argument 'sxform_pairs'``），错误信息毫无指向、
    栈顶远离调用点。本函数先用 :func:`inspect.signature` 主动比对所需
    keyword-only 参数，缺失即抛带请重建提示的 :class:`RuntimeError`；无法
    内省（无 ``__text_signature__``）时退化为在调用点捕获 ``TypeError``，锚定
    PyO3 漂移模板（"unexpected keyword argument"）并按参数名命中重映射：仅
    漂移型错误被重映射，余者原样上抛，避免掩盖真实 bug（如 dt 参数类型错误）。

    与 :meth:`SPICEManager.enable_ephem_cache` 外层的 ``except ImportError``
    （覆盖扩展未编译/未开 spice feature）互补，共同覆盖扩展存在但过期
    这一会以裸 ``TypeError`` 冒泡的情形。属靶向加固，非 ABI 版本戳（见
    ``docs/plans`` 下 #3 计划）。
    """
    # 主路径：签名内省预检。
    try:
        available = inspect.signature(fn).parameters
        missing = [k for k in required_kwargs if k not in available]
    except (ValueError, TypeError):
        # PyO3 未暴露 __text_signature__ 等情形：交由调用点兜底。
        missing = []
    if missing:
        raise RuntimeError(_STALE_BINARY_HINT.format(fn_name=fn_name, missing=missing))

    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        # 兜底：内省不可用时的签名漂移。锚定 PyO3 漂移模板
        # "unexpected keyword argument"，避免把合法类型错误（如 dt 参数类型
        # 不对 → "must be real number, not str"）误判为编译产物过期。
        msg = str(exc)
        if "unexpected keyword argument" in msg and any(name in msg for name in required_kwargs):
            raise RuntimeError(
                _STALE_BINARY_HINT.format(fn_name=fn_name, missing=list(required_kwargs))
            ) from exc
        raise  # 与漂移无关的 TypeError：原样上抛，不掩盖


class SPICEManager(EphemerisProvider):
    """Wrapper around the NASA SPICE toolkit
    (ephemeris queries, time conversion, kernel management).

    SPICE 内核管理器：SPICE 星历数据提供者实现。

    SPICE 内核管理器：SPICE 星历数据提供者实现。

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
    _bodies_registered: bool = False

    #: 已加载的星历内核 (绝对路径, datum)，按加载顺序；末项 datum 即当前 GM 口径。
    #: **类级共享**——CSPICE 内核池是进程级全局的（ADR 0048）：任一实例的
    #: load/unload 都会改变"重叠覆盖段取后加载者"的实际口径，故簿记必须与池同域。
    #: 实例级簿记会在同进程另一 manager 加载内核时静默错配（de440s 位置 + DE421 GM）。
    #: 按绝对路径去重（同一文件重复加载只留一条）；改列表一律持 ``_bookkeeping_lock``。
    _loaded_ephemeris: list[tuple[str, str]] = []
    _bookkeeping_lock = threading.Lock()

    def __init__(self) -> None:
        """初始化 SPICE 管理器。"""
        # 预插值星历缓存（enable_ephem_cache 后生效；get_body_position/state
        # 优先走 cache，避免逐步跨 Python↔C 边界查 SPICE）。见 ephem_cache.py。
        self._ephem_cache: EphemCache | None = None
        # 本实例已就 (body, 请求 datum, 实际 datum) 回退告警过的组合：每组合一次。
        # 实例级（非模块全局），避免跨测试/跨 manager 的状态耦合。
        self._gm_fallback_warned: set[tuple[str, str, str]] = set()
        # 本实例已就"按 DE440 近似"或"未收录"告警过的星历内核名。
        self._kernel_datum_warned: set[str] = set()

    def _ensure_leapseconds(self, search_dir: str | None = None):
        """确保闰秒内核已加载（线程安全）。

        Args:
            search_dir: 额外的搜索目录（如被加载内核的同级目录），优先级
                高于 ``_LEAPSECOND_SEARCH_PATHS``。传 None 或空串时仅搜索
                默认路径。找不到时发告警但不 raise（保留用户自行 furnsh 的
                可能），``_leapseconds_loaded`` 保持 False 以便后续重试。
        """
        if SPICEManager._leapseconds_loaded:
            return
        with SPICEManager._leapseconds_lock:
            if SPICEManager._leapseconds_loaded:
                return
            # search_dir 优先级最高，置列表首位；空串回退默认搜索路径。
            search_paths = (
                [search_dir, *_LEAPSECOND_SEARCH_PATHS] if search_dir else _LEAPSECOND_SEARCH_PATHS
            )
            path = _find_leapseconds_kernel(search_paths)
            if path:
                get_spiceypy().furnsh(path)
                # Rust cspice 与 Python spiceypy 是独立 CSPICE 实例（内核池
                # 不共享，见 load_kernel 注释）。下沉到 Rust 的批量 ET→UTC
                # （frame_convert.batch_et_to_utc_py）在 Rust 实例查闰秒表，
                # 缺 LSK 报 MISSINGTIMEINFO；此处双 furnsh 补齐。
                from e2m2e.spice_ext import spice_furnsh

                if spice_furnsh is not None:
                    spice_furnsh(path)
                SPICEManager._leapseconds_loaded = True
            else:
                _logger.warning(
                    "未找到闰秒内核（.tls）：UTC↔ET 时间转换将失败"
                    "（SPICE NOLEAPSECONDS）。请设置 SPICE_KERNEL_DIR 环境变量，"
                    "或将 naif0012.tls 放入内核目录。"
                )

    def _warn_approximated_kernel(self, name: str | None, datum: str) -> None:
        """对「GM 按其它基准近似」的内核按内核名告警一次（落实 ADR 0048 契约）。"""
        if name is None or name not in _APPROXIMATED_KERNELS:
            return
        if name in self._kernel_datum_warned:
            return
        self._kernel_datum_warned.add(name)
        _logger.warning(
            "星历内核 %s.bsp 无自有权威 GM 表，GM 按 %s 口径近似：位置取该内核、"
            "GM 取 %s，二者非同代。需按该内核口径复算时请先补齐权威 GM 与白名单"
            "（见 ADR 0048）。",
            name,
            datum,
            datum,
        )

    def _warn_unregistered_kernel(self, name: str) -> None:
        """对匹配 DE 命名但未收录的内核告警一次：不改变当前 GM 口径。"""
        if name in self._kernel_datum_warned:
            return
        self._kernel_datum_warned.add(name)
        _logger.warning(
            "星历内核 %s.bsp 未收录于 GM 基准白名单：不改变当前 GM 口径（停留 %s），"
            "位置取该内核——口径可能与该内核非同代。需按该口径复算时请补白名单与"
            "权威 GM（见 ADR 0048）。",
            name,
            self.ephemeris_datum,
        )

    def load_kernel(self, path: str) -> None:
        """加载一个 SPICE 内核文件（.bsp / .bpc / .tf 等）。

        加载前会自动确保闰秒内核已就绪。星历（SPK）内核加载后即成为当前
        GM 口径（:attr:`ephemeris_datum`），配对规则见模块级
        ``_KERNEL_DATUM_BY_SPK``。

        Raises:
            FileNotFoundError: 当指定路径的文件不存在时。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Kernel file not found: {path}")
        # 把被加载内核的同级目录纳入闰秒搜索（最高优先级），覆盖 PyPI 安装或
        # 非 repo 环境下用户把 naif0012.tls 与 .bsp 放一起的常见用法。
        self._ensure_leapseconds(search_dir=os.path.dirname(path))
        # 首次加载时在本（Python spiceypy）实例注册行星名→质心/本体 ID 别名
        # （类级 once 标志，对称于 Rust 侧 furnsh 时 register_bodies 的 Once）。
        # boddef 对同一 (name, id) 幂等，重复调用无害。
        if not SPICEManager._bodies_registered:
            for name, naif_id in _BODY_ID_ALIASES:
                get_spiceypy().boddef(name, naif_id)
            SPICEManager._bodies_registered = True
        get_spiceypy().furnsh(path)
        # Rust cspice 与 Python spiceypy 是独立 CSPICE 实例（静态链接，
        # 内核池不共享）。spice feature 启用时双 furnsh，让下沉到 Rust
        # 的力（ThirdBody/Indirect/...）也能查到。桥接经共享内核叶
        # spice_ext 直达 Rust 扩展（ADR 0039）。
        from e2m2e.spice_ext import spice_furnsh

        if spice_furnsh is not None:
            spice_furnsh(path)
        # 星历内核簿记：仅在 furnsh 成功后登记，失败不污染当前口径。列表类级共享
        # （与进程级 CSPICE 池同域，ADR 0048），就地改类列表，避免实例阴影。
        kernel_name = _spk_kernel_name(path)
        datum = _datum_for_kernel(path)
        if datum is None:
            if kernel_name is not None:
                self._warn_unregistered_kernel(kernel_name)
            return
        abspath = os.path.abspath(path)
        with SPICEManager._bookkeeping_lock:
            loaded = SPICEManager._loaded_ephemeris
            loaded[:] = [entry for entry in loaded if entry[0] != abspath]
            loaded.append((abspath, datum))
        self._warn_approximated_kernel(kernel_name, datum)

    def unload_kernel(self, path: str) -> None:
        """卸载一个已加载的 SPICE 内核文件，释放相关资源。

        卸载星历内核后，当前 GM 口径回落到仍加载的最后一个星历内核；
        无星历内核时回到默认 DE440。
        """
        get_spiceypy().unload(path)
        # Rust cspice 与 Python spiceypy 是独立 CSPICE 实例（静态链接，
        # 内核池不共享）。load_kernel 双 furnsh，此处对称卸载 Rust 侧，
        # 避免 Rust 内核池残留导致测试结果依赖执行顺序。
        # Rust 侧只卸载确经 spice_furnsh 加载过的文件，未加载时静默跳过
        # （保持重复 unload 幂等）。
        from e2m2e.spice_ext import spice_unload

        if spice_unload is not None:
            spice_unload(path)
        abspath = os.path.abspath(path)
        with SPICEManager._bookkeeping_lock:
            loaded = SPICEManager._loaded_ephemeris
            loaded[:] = [entry for entry in loaded if entry[0] != abspath]

    @property
    def ephemeris_datum(self) -> str:
        """当前 GM 基准：最后一个已加载星历内核的 datum，无则 DE440（ADR 0048）。

        簿记是**类级**的（ADR 0048）：CSPICE 内核池进程级全局，任一实例加载/卸载
        的星历内核都改变"重叠覆盖段取后加载者"的实际口径，故这里读的是进程内
        真实生效的末位内核，位置与 GM 天然同口径、不需调用方手工配对。
        """
        with SPICEManager._bookkeeping_lock:
            entries = SPICEManager._loaded_ephemeris
            if entries:
                return entries[-1][1]
            return _DEFAULT_EPHEMERIS_DATUM

    def enable_ephem_cache(
        self,
        bodies: list[str],
        et_start: float,
        et_end: float,
        *,
        dt: float = 3600.0,
        frame: str = "J2000",
        observer: str = "EARTH",
        frame_pairs: list[tuple[str, str]] | None = None,
        sxform_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        """构建并启用预插值星历缓存（Python 层 + Rust 积分层）。

        Python 侧 ``EphemCache`` 拦 ``get_body_position/state``；Rust 侧
        （``_integrators.enable_ephem_cache``）给 ``compiled_stm``/多重打靶
        积分内循环的三次样条查表。两套缓存独立构建但同源（同网格采样），
        保证 Rust 积分不逐次调 cspice（消除 DAFFRNOTFOUND 与每步 FFI 开销）。

        反向传播契约：反向段（``direction="backward"`` 预报、递减 t_patch 的
        多重/分段打靶）要求缓存窗口 ``[et_start, et_end]`` 覆盖反向段到达的最早
        时刻。Rust 积分内循环的窗口外查询是确定性硬失败
        （``CacheMissError::OutOfRange``，信封短码 ``EPHEM_CACHE_MISS``，见
        ``e2m2e-spice`` ``ephem_cache.rs``），不静默外推；Python 层
        ``get_body_state/position`` 在缓存未覆盖时回退直接 SPICE 查询，成败
        取决于内核覆盖。

        Args:
            bodies: 需缓存的天体名列表（EARTH/MOON/SUN/行星）。
            et_start/et_end: 缓存覆盖的 ET 秒范围；调用方负责让窗口覆盖全部
                查询时刻（含反向段）。
            dt: 预采样网格步长（秒），默认 3600。
            frame: Python 层缓存采样坐标系（J2000）。
            observer: Python 层缓存采样原点（EARTH）。
            frame_pairs: Rust 层帧旋转对（(from, to)）。GravityField 需
                body-fixed→J2000（如 ("ITRF93","J2000")、("MOON_PA","J2000")）。
                缺省注册 (frame, "J2000")。
            sxform_pairs: Rust 层 6×6 状态变换对（(from, to)）。Lense-Thirring
                需 body-fixed→J2000（如 ("ITRF93","J2000")）。可为空列表。
        """
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
        # 同步启用 Rust 侧缓存。Rust 力模型（compiled.rs）查天体用两种
        # observer：第三体/间接项用传播系原点（EARTH），GravityField 用
        # SSB（需把原点也换算到 SSB 平移）。故每个 body 同时注册
        # (body, observer) 与 (body, "SOLAR SYSTEM BARYCENTER")；帧对注册
        # 传入的 frame_pairs（缺省 (frame, "J2000")）。
        #
        # 关键：缓存键是**精确字符串**（ephem_cache.rs L233），而第三体力的
        # to_rust_spec 把天体转成 NAIF-ID 字符串（如 "MOON"→"301"）。故每个
        # body 同时注册名字与 NAIF-ID 两种键，保证 ThirdBody 查询（用 ID）
        # 与 GravityField 查询（用名字）都命中。Rust 采样同样走 cspice，
        # 之后积分查表。
        from e2m2e.spice_ext import enable_ephem_cache as _rust_enable
        from e2m2e.spice_ext import require_rust_extension

        require_rust_extension("enable_ephem_cache")
        # 天体名 → NAIF-ID 字符串（与 third_body_gravity.py 的
        # _name_or_id 一致；失败保留名字）
        try:
            import spiceypy as _sp

            id_keys = [str(_sp.bods2c(b)) if _sp.bods2c(b) > 0 else b.upper() for b in bodies]
        except Exception:
            id_keys = [b.upper() for b in bodies]

        frame_pairs = frame_pairs or [(frame, "J2000")]
        sxform_pairs = sxform_pairs or []
        _call_rust_or_compat_error(
            _rust_enable,
            [
                (k, observer.upper())
                for b, kid in zip(bodies, id_keys, strict=True)
                for k in (b.upper(), kid)
            ]
            + [
                (k, "SOLAR SYSTEM BARYCENTER")
                for b, kid in zip(bodies, id_keys, strict=True)
                if b.upper() != "SOLAR SYSTEM BARYCENTER"
                for k in (b.upper(), kid)
            ],
            frame_pairs,
            et_start,
            et_end,
            fn_name="enable_ephem_cache",
            required_kwargs=("dt", "sxform_pairs"),
            dt=dt,
            sxform_pairs=sxform_pairs,
        )

    def disable_ephem_cache(self) -> None:
        """关闭预插值星历缓存（Python 层 + Rust 层），回退到逐步 SPICE 查询。"""
        self._ephem_cache = None
        from e2m2e.spice_ext import disable_ephem_cache as _rust_disable
        from e2m2e.spice_ext import require_rust_extension

        require_rust_extension("disable_ephem_cache")
        _rust_disable()

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

    #: 各 GM 基准的星历内核候选（按偏好排序）。唯一来源：design 链路同引用本表。
    #: 同一基准可有多个等价内核（如 de440/de440s 的 GM 表相同）。
    _DATUM_KERNEL_PREFERENCE: dict[str, tuple[str, ...]] = {
        "DE421": ("de421.bsp",),
        "DE440": ("de440.bsp", "de440s.bsp"),
    }

    @classmethod
    def datum_kernel_names(cls, datum: str) -> tuple[str, ...]:
        """GM 基准的星历内核候选名；未知基准返回空元组（ADR 0048 单一来源）。"""
        return cls._DATUM_KERNEL_PREFERENCE.get(datum.upper(), ())

    def find_ephemeris_kernel(self, search_dir: str, preferred: str | None = None) -> str:
        """在指定目录中按优先级搜索星历内核文件（.bsp）。

        默认优先级：de440.bsp > de440s.bsp > de435.bsp > de438.bsp。
        ``preferred`` 指定 GM 基准（如 ``"DE421"``）时，该基准的候选内核
        （同一 datum 的多个内核等价）置于候选首位；**显式请求而全部缺失即
        报错**，不静默降级到其它 DE 系列——降级会让「请求 DE421 口径」变成
        「悄悄用 DE440 口径」，比失败更糟。

        Raises:
            FileNotFoundError: 目录不存在、显式 preferred 的内核全缺失，或其中无匹配的内核文件。
        """
        if not os.path.isdir(search_dir):
            raise FileNotFoundError(
                f"Ephemeris kernel search directory does not exist: {search_dir}"
            )
        candidates = list(self._EPHEMERIS_KERNEL_PRIORITY)
        if preferred is not None:
            names = self.datum_kernel_names(preferred)
            if not names:
                raise ValueError(f"未知的星历基准（无偏好内核）: {preferred}")
            present = [name for name in names if os.path.isfile(os.path.join(search_dir, name))]
            if not present:
                wanted = " 或 ".join(names)
                raise FileNotFoundError(
                    f"请求 {preferred.upper()} 口径但内核全缺失（{wanted}）：{search_dir}"
                    "（内核由 kernels-v1 release 分发，跑 make kernels 获取）"
                )
            # 该口径的全部可用内核置于候选首位（声明顺序即偏好顺序）。
            candidates = present + candidates
        for candidate in candidates:
            path = os.path.join(search_dir, candidate)
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise FileNotFoundError(f"No ephemeris kernel found in {search_dir}")

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
