"""共享内核叶（ADR 0039）的架构元测试。

re-export 必须保持对象身份——旧路径是永久稳定别名，不是复制。
"""

from __future__ import annotations

import pytest

from e2m2e import status
from e2m2e.algorithm.results import CAUSE_STATUS, ResultStatus
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.aux


def test_status_reexports_preserve_identity():
    """旧路径（data.templates / algorithm.results）与 e2m2e.status 是同一批对象。"""
    assert status.ConvergenceState is ConvergenceState
    assert status.FailureCause is FailureCause
    assert status.ResultStatus is ResultStatus
    assert status.CAUSE_STATUS is CAUSE_STATUS
    # 防"复制而非 re-export"回归：定义点必须是共享内核叶本身。
    assert ResultStatus.__module__ == "e2m2e.status"
    assert ConvergenceState.__module__ == "e2m2e.status"
