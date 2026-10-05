"""execution 工具子集清单与工具元数据的对账测试（#805）。

``LONG_RUNNING_TOOLS``（worker 子进程执行策略）与 ``BINARY_FRAME_TOOLS``
（sidecar 帧契约）是执行核心的手写工具名子集，被 CLI、sidecar、MCP worker
共用。facade 方法改名或新增长任务时这些清单靠人工同步，本文件断言清单
与 ``tool_inventory()`` 派生面双向对账：清单里的名字必须是真实工具，
``long_running=True`` 标记的工具必须进 ``LONG_RUNNING_TOOLS``。
"""

from __future__ import annotations

import pytest

from e2m2e.api.execution import BINARY_FRAME_TOOLS, LONG_RUNNING_TOOLS
from e2m2e.api.facade import Facade, tool_inventory

pytestmark = pytest.mark.aux


@pytest.fixture(scope="module")
def inventory():
    return {info.name: info for info in tool_inventory(Facade())}


def test_long_running_subset_matches_metadata(inventory):
    """LONG_RUNNING_TOOLS 与 mcp_exposed(long_running=True) 标记双向一致。"""
    marked = {name for name, info in inventory.items() if info.long_running}
    assert marked == LONG_RUNNING_TOOLS
    for name in LONG_RUNNING_TOOLS:
        assert name in inventory, f"LONG_RUNNING_TOOLS 含不存在的工具 {name}"


def test_binary_frame_subset_matches_inventory(inventory):
    """BINARY_FRAME_TOOLS 里的名字都是真实工具。"""
    for name in BINARY_FRAME_TOOLS:
        assert name in inventory, f"BINARY_FRAME_TOOLS 含不存在的工具 {name}"
