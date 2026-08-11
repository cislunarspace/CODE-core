"""e2m2e 统一异常层次。

所有 e2m2e 抛出的异常都以 :class:`E2M2EError` 为共同基类，
便于调用方用 ``except E2M2EError`` 统一捕获库内部错误。

异常不属于任何一层，放顶层供 data/algorithm/api/tools 共享（ADR 0011 五层
结构落地后，旧路径 ``e2m2e.core.exceptions`` 经 shim 重导出保持兼容）。
"""


class E2M2EError(Exception):
    """所有 e2m2e 异常的共同基类。"""

    pass


class RustExtensionUnavailableError(E2M2EError, RuntimeError):
    """需要使用 Rust 扩展但扩展不可用/缺少所需符号时抛出。

    spice 是默认且唯一支持的 feature（ADR 0009）：核心计算路径必须由
    Rust 扩展承载，不允许静默回退到 Python/scipy（issue #378）。扩展
    未构建、构建不含 spice feature、或符号缺失时在使用处抛本异常，
    错误信息含 ``make dev`` 修复指引。

    同时继承 :class:`E2M2EError`（库内统一捕获）与 :class:`RuntimeError`
    （兼容既有裸 ``RuntimeError`` 捕获点）。
    """

    pass
