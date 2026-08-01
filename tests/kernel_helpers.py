"""SPICE 内核加载/卸载辅助函数。

从 tests/conftest.py 抽出为独立模块：测试文件若用 ``from conftest import ...``
导入这些函数，在 pytest-xdist 并行下会与其他目录的 conftest.py 撞名
（``sys.modules['conftest']`` 被先加载者占据），解析到错误的模块。
改为从唯一命名的 ``kernel_helpers`` 导入（配合 pyproject 的
``pythonpath = ["tests"]``）消除歧义。
"""

import os

SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"),
)

# 定义 body-fixed 帧所需的 SPICE 内核文件名(issue #187):
# 地球 ITRF93 需要 earth_latest_high_prec.bpc;text PCK 与月球 MOON_PA 需要
# pck00010.tpc、SPICELunaCurrentKernel.bpc、SPICELunaFrameKernel.tf。
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
        实际 furnsh 的内核绝对路径列表;调用方应在 teardown 时对其逆序 unload。
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
