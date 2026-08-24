"""e2m2e 统一异常层次。

所有 e2m2e 抛出的异常都以 :class:`E2M2EError` 为共同基类，
便于调用方用 ``except E2M2EError`` 统一捕获库内部错误。

异常不属于任何一层，放顶层供 data/algorithm/api/tools 共享（ADR 0011 五层
结构）。
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

    同时继承 :class:`E2M2EError` （库内统一捕获）与 :class:`RuntimeError`
    （兼容既有裸 ``RuntimeError`` 捕获点）。
    """

    pass


class PropagationFailure(E2M2EError):
    """传播失败（ADR 0020 决策 2，取代字符串前缀匹配契约）。

    步长塌缩到机器精度地板等确定性传播失败，在 Rust→Python FFI 边界
    翻译成本异常，供下游 ``except PropagationFailure`` 精确捕获；取代对
    ``"step size collapsed"`` 错误消息前缀匹配的依赖（issue #317 第
    3.1 项），改写 Rust 侧错误消息措辞不影响捕获。

    继承 :class:`E2M2EError` （统一捕获契约），但不继承 :class:`RuntimeError`
    （与通用运行时错误区分；既有裸 ``except RuntimeError`` 不再兜住它）。
    """

    pass
