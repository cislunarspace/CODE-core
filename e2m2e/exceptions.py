"""e2m2e 统一异常层次 / Unified exception hierarchy.

[English]

All exceptions raised by e2m2e share :class:`E2M2EError` as their common base,
so callers can catch library-internal errors uniformly via
``except E2M2EError``. Exceptions belong to no single layer; they live at the
top level for data/algorithm/api/tools to share (ADR 0011).

[简体中文]

所有 e2m2e 抛出的异常都以 :class:`E2M2EError` 为共同基类，
便于调用方用 ``except E2M2EError`` 统一捕获库内部错误。

异常不属于任何一层，放顶层供 data/algorithm/api/tools 共享（ADR 0011 五层
结构）。
"""


class E2M2EError(Exception):
    """所有 e2m2e 异常的共同基类。/ Common base of all e2m2e exceptions."""

    pass


class RustExtensionUnavailableError(E2M2EError, RuntimeError):
    """需要使用 Rust 扩展但扩展不可用/缺少所需符号时抛出。

    [English] Raised when the Rust extension is required but unavailable or
    missing needed symbols. spice is the default and only supported feature
    (ADR 0009): core compute paths must run on the Rust extension with no
    silent fallback to Python/scipy. Message includes a ``make dev`` fix hint.
    Also subclasses :class:`RuntimeError` so bare ``except RuntimeError`` sites
    keep working.

    spice 是默认且唯一支持的 feature（ADR 0009）：核心计算路径必须由
    Rust 扩展承载，不允许静默回退到 Python/scipy。扩展未构建、构建
    不含 spice feature、或符号缺失时在使用处抛本异常，错误信息含
    ``make dev`` 修复指引。

    同时继承 :class:`E2M2EError` （库内统一捕获）与 :class:`RuntimeError`
    （兼容裸 ``RuntimeError`` 捕获点）。
    """

    pass


class PropagationFailure(E2M2EError):
    """传播失败（确定性传播错误，ADR 0020 决策 2）。

    [English] Deterministic propagation failures — step-size collapse to the
    machine floor etc. — surface as this exception at the Rust→Python FFI
    boundary for precise downstream capture via ``except PropagationFailure``;
    Rust-side message wording may change without breaking type-based capture.
    Inherits only :class:`E2M2EError` (not ``RuntimeError``) so bare
    ``except RuntimeError`` does not swallow it.

    步长塌缩到机器精度地板等确定性传播失败，在 Rust→Python FFI 边界
    翻译成本异常，供下游 ``except PropagationFailure`` 精确捕获；
    Rust 侧错误消息措辞可自由改写，不影响按异常类型的捕获契约。

    继承 :class:`E2M2EError` （统一捕获契约），但不继承 :class:`RuntimeError`
    （与通用运行时错误区分，裸 ``except RuntimeError`` 不捕获本异常）。
    """

    pass
