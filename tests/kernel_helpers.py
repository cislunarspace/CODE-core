"""测试用的 SPICE 内核加载/卸载辅助模块。"""

import os

SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"),
)

# 定义 body-fixed 帧所需的 SPICE 内核文件名（issue #187）：
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
