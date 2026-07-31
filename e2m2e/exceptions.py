"""e2m2e 统一异常层次。

所有 e2m2e 抛出的异常都以 :class:`E2M2EError` 为共同基类，
便于调用方用 ``except E2M2EError`` 统一捕获库内部错误。

异常不属于任何一层，放顶层供 data/algorithm/api/tools 共享（ADR 0011 五层
结构落地后，旧路径 ``e2m2e.core.exceptions`` 经 shim 重导出保持兼容）。
"""


class E2M2EError(Exception):
    """所有 e2m2e 异常的共同基类。"""

    pass
