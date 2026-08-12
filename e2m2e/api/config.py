"""运行配置：内核路径、精度阈值、日志级别。

只管**运行环境** （ADR 0014）；物理常量归 data/templates/。构造注入 Facade
（``Facade(config=Config(...))``），内部默认从环境变量读。SPICEManager 全局句柄、
r2s2 进程单例作为已知限制用 Config 显式管理。

实现状态：骨架。字段待定稿。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["Config"]


@dataclass
class Config:
    """e2m2e 运行配置。

    Attributes:
        kernel_dir: SPICE 内核目录（默认 $SPICE_KERNEL_DIR 或仓库 kernels/）。
        log_level: 日志级别。
        tolerance: 默认数值容差（积分 rtol/atol）。
    """

    kernel_dir: str = field(default_factory=lambda: os.environ.get("SPICE_KERNEL_DIR", "kernels"))
    log_level: str = "WARNING"
    rtol: float = 1e-12
    atol: float = 1e-12
