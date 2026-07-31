"""异常基类 shim：``E2M2EError`` 已迁至顶层 ``e2m2e.exceptions``。

旧路径 ``e2m2e.core.exceptions.E2M2EError`` 在迁移期继续可用（ADR 0011
过渡策略：旧包位置 re-export 新模块）。
"""

from ..exceptions import E2M2EError

__all__ = ["E2M2EError"]
