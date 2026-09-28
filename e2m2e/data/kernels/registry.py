"""SPICE 内核静态登记表：纯数据表与纯函数（无 IO、无状态、无线程同步）。

从 manager.py 拆出的静态查表层（#751）：常用天体 NAIF ID、SPK 内核
文件名 → GM 基准（datum）白名单（ADR 0048）、行星名→质心/本体 ID
别名表、星历内核搜索偏好。全部为模块级常量与纯函数——不 import
spiceypy、不碰内核池；供 manager（加载/簿记/搜索/告警）与 queries
（GM 查询）共同引用，自身不得反向依赖它们（避免导入环）。

依赖方向：仅标准库（os/re）。
"""

from __future__ import annotations

import os
import re

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
#: 加载时按内核名告警一次，落实"绝不静默混用"。
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


#: 行星名→质心/本体 NAIF ID 别名表。de440s/de430/de440 全本只含行星**质心**
#: 段 + 地球族本体 + 月球 + 太阳，不含行星本体段（499/599/…）；CSPICE 默认表
#: 把 "MARS" 解析成本体 499（de440s 不含）而非质心 4。本表把这些名字注册到
#: 质心/本体 ID，使 Python spiceypy 实例与 Rust cspice 实例（那边在
#: ``spice_ffi::register_bodies`` 注册同一份表）解析一致。
#:
#: 单一归属 registry 模块。两份表（Python 这里 +
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


#: 星历内核搜索默认优先级（find_ephemeris_kernel 缺省顺序）。
_EPHEMERIS_KERNEL_PRIORITY: list[str] = ["de440.bsp", "de440s.bsp", "de435.bsp", "de438.bsp"]

#: 各 GM 基准的星历内核候选（按偏好排序）。唯一来源：design 链路同引用本表。
#: 同一基准可有多个等价内核（如 de440/de440s 的 GM 表相同）。
_DATUM_KERNEL_PREFERENCE: dict[str, tuple[str, ...]] = {
    "DE421": ("de421.bsp",),
    "DE440": ("de440.bsp", "de440s.bsp"),
}
