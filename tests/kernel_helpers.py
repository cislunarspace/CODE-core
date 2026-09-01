"""测试用的 SPICE 内核加载/卸载辅助模块。"""

import importlib
import os
import sys

import pytest

SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"),
)


def spice_kernels_available() -> bool:
    """通用 SPICE 内核（``.tls`` 闰秒 + ``.bsp`` 星历）是否齐备。

    全套件唯一的通用可用性探测实现（ADR 0025 决策 4）；特定资源
    （如 ``de440t.bsp`` 时间星历、ITRF93 二进制 PCK）的探测不归此处。
    """
    if not os.path.isdir(SPICE_KERNEL_DIR):
        return False
    names = os.listdir(SPICE_KERNEL_DIR)
    has_tls = any(f.endswith(".tls") for f in names)
    has_bsp = any(f.endswith(".bsp") for f in names)
    return has_tls and has_bsp


#: 通用 SPICE 可用性 skip 标记；与 ``pytest.mark.spice`` 搭配使用。
requires_spice = pytest.mark.skipif(
    not spice_kernels_available(),
    reason="SPICE kernels (.tls + .bsp) not available",
)


# 定义 body-fixed 帧所需的 SPICE 内核文件名：
# 地球 ITRF93 需要二进制 PCK（earth_latest_high_prec.bpc），未来历元段由
# 预测 PCK（SPICEEarthPredictedKernel.bpc，须先加载，使重叠段取历史高精度
# 数据）补齐；月球 MOON_PA 需要文本 PCK（pck00010.tpc）、二进制 PCK
# （SPICELunaCurrentKernel.bpc）与帧内核（SPICELunaFrameKernel.tf）。
BODY_FIXED_KERNELS = [
    "SPICEEarthPredictedKernel.bpc",
    "earth_latest_high_prec.bpc",
    "pck00010.tpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
]


def load_body_fixed_kernels(spice) -> list[str]:
    """向 ``spice``(SPICEManager)furnsh 所有可用的 body-fixed 内核。

    加载 ``kernels/`` 下定义 ITRF93 / MOON_PA 帧所需的 BPC/TPC/TF 内核
    (见 :data:`BODY_FIXED_KERNELS`),使 ``GravityField`` 的 body-fixed 坐标轴
    (ITRFSpiceAxes)能解析 SPICE 旋转。文件不存在时静默跳过该项。

    Args:
        spice: 已初始化的 :class:`~e2m2e.data.kernels.manager.SPICEManager`。

    Returns:
        实际加载（furnsh）的内核绝对路径列表。调用方应在 teardown 时对其逆序卸载。
    """
    loaded: list[str] = []
    if not os.path.isdir(SPICE_KERNEL_DIR):
        return loaded
    for name in BODY_FIXED_KERNELS:
        path = os.path.join(SPICE_KERNEL_DIR, name)
        if os.path.exists(path):
            spice.load_kernel(path)
            loaded.append(path)
    return loaded


def unload_kernels(spice, paths: list[str]) -> None:
    """逆序卸载 ``load_body_fixed_kernels`` 之类返回的内核路径列表。"""
    for path in reversed(paths):
        spice.unload_kernel(path)


# =============================================================================
# Rust 扩展符号守卫（#603）：native 符号缺失时跳过而非硬失败
#
# 判据口径对齐 e2m2e/spice_ext.py 的 _ensure_symbols 报错文案（被现存
# match="make dev" 测试钉住，勿另造表述）；与 requires_spice 同款先例：
# 环境能力不足统一表达为 skip，读日志的人据此分清环境问题与代码问题。
# =============================================================================


def native_symbols_available(*symbols: str) -> bool:
    """Rust 扩展可导入且指定符号在 integrators 门面命名空间非 None。

    门面（``e2m2e.integrators``）在导入时从 ``e2m2e._integrators`` 装载
    扩展符号，缺失的符号为 ``None``——与本套件其他可用性探测一样做
    存在性检查，不触发真实计算。查 sys.modules 优先于 import 语句：
    ``import a.b as x`` 绑定时父包属性优先，会拿到测试替换不掉的真模块。
    """
    module = sys.modules.get("e2m2e.integrators")
    if module is None:
        try:
            module = importlib.import_module("e2m2e.integrators")
        except ImportError:
            return False
    return all(getattr(module, symbol, None) is not None for symbol in symbols)


def requires_native_symbols(*symbols: str):
    """native 符号守卫：缺符号的用例跳过并说明重建指引（#603）。

    用法与 :data:`requires_spice` 相同：``@requires_native_symbols("foo_py")``
    叠加在测试或类上。reason 口径与 spice_ext 的报错一致。
    """
    return pytest.mark.skipif(
        not native_symbols_available(*symbols),
        reason=(
            "e2m2e._integrators 缺少所需符号："
            + ", ".join(symbols)
            + "。spice 是默认且唯一支持的 feature；请用 make dev 重建扩展"
        ),
    )
