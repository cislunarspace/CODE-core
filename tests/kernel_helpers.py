"""测试用的 SPICE 内核加载/卸载辅助模块。"""

import os

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
# 地球 ITRF93 需要二进制 PCK（earth_latest_high_prec.bpc），
# 月球 MOON_PA 需要文本 PCK（pck00010.tpc）、二进制 PCK（SPICELunaCurrentKernel.bpc）
# 与帧内核（SPICELunaFrameKernel.tf）。
BODY_FIXED_KERNELS = [
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
