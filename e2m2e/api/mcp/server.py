"""MCP 服务：LLM 工具入口。

进程内库为主体 + CLI 薄包装 mcp-serve（ADR 0014）：``create_server(facade)``
函数（进程内、可测试）+ ``e2m2e mcp-serve`` 子命令。一个 Facade 实例 = 一个
server。MCP 工具 = facade 上 mcp_exposed=True 的方法（纯派生，见 tools.py），
传输层包统一信封（见 envelope.py）。

依赖 ``[mcp]`` extra：本模块在缺 ``mcp`` 库时导入即失败，调用方（CLI）负责
给出安装提示。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import anyio.to_thread
from mcp.server import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from . import envelope, tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anyio.abc import ByteReceiveStream

    from ..facade import Facade

__all__ = [
    "LONG_RUNNING_TOOLS",
    "create_server",
    "handle_list_tools",
    "handle_call_tool",
    "make_progress_reporter",
    "run_tool_in_worker",
]

# 中间通知的最小发送间隔（秒）：高粒度回调（WSB 网格搜索可达数千次）
# 节流成客户端可消化的节奏；fraction=1.0 收尾通知永不节流。
_PROGRESS_MIN_INTERVAL_SEC = 0.1
# 单次投递等待事件循环的上限（秒）：事件循环停摆/已关时不无限阻塞
# worker 线程；超时按投递失败处理（静默降级为无进度）。
_PROGRESS_SEND_TIMEOUT_SEC = 5.0

# 长任务工具（#588 / #576 Phase 2，子进程隔离架构）：分钟级计算改跑
# worker 子进程（见 worker.py），使 MCP 取消（notifications/cancelled，
# interrupt 模式取消 handler 作用域）与客户端断连（EOF 取消任务组）都能
# 可靠传播为进程 kill。其余工具线程池直跑，行为不变。执行策略是传输层
# 关注点，放这里而不入 Facade 元数据。
LONG_RUNNING_TOOLS = frozenset({"transfer_design", "orbit_family_generation"})

# worker 子进程命令（模块常量：测试注入 fake worker 用，同
# _PROGRESS_MIN_INTERVAL_SEC 的 monkeypatch 先例）。
_WORKER_ARGV = [sys.executable, "-m", "e2m2e.api.mcp.worker"]
# 取消后 kill+收尸的时限（秒）：Windows TerminateProcess 即时生效，此为
# 病理情形兜底，超时也不再阻塞取消收尾。
_KILL_REAP_TIMEOUT_SEC = 10.0


def make_progress_reporter(context: Any) -> Any:
    """把 MCP 请求上下文包成线程安全的进度回调（Facade 签名 ``(fraction, message)``）。

    客户端未在请求 meta 带 ``progressToken`` 时返回 None——零开销直通。
    工具本体在 anyio worker 线程执行，而算法层回调可能从任意线程触发
    （WSB 网格搜索的 Rust drainer 线程，见 ``spawn_progress_drainer``），
    因此投递走创建时捕获的 asyncio 事件循环（``run_coroutine_threadsafe``）；
    ``e2m2e mcp-serve`` 经 ``anyio.run`` 默认 asyncio 后端，无运行中
    asyncio 事件循环时同样返回 None。阻塞等待投递完成以施加背压并保持
    通知顺序；投递失败静默吞掉——进度失败不得影响计算。
    """
    meta = getattr(context, "meta", None)
    if getattr(meta, "progress_token", None) is None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    session = context.session
    last_sent = 0.0

    def reporter(fraction: float, message: str | None = None) -> None:
        nonlocal last_sent
        now = time.monotonic()
        if fraction < 1.0 and now - last_sent < _PROGRESS_MIN_INTERVAL_SEC:
            return
        last_sent = now
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                session.report_progress(fraction, 1.0, message), loop
            ).result(timeout=_PROGRESS_SEND_TIMEOUT_SEC)

    return reporter


def handle_list_tools(facade: Facade) -> list[Tool]:
    """列出工具（纯函数，便于测试）。"""
    return [
        Tool(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in tools.tool_specs(facade)
    ]


def _to_result(env: envelope.Envelope) -> CallToolResult:
    """信封 → MCP CallToolResult（JSON 文本 + isError 标志）。"""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(env, ensure_ascii=False))],
        is_error=env["status"] == "error",
    )


async def _iter_lines(stream: ByteReceiveStream) -> AsyncIterator[bytes]:
    """从字节流逐行产出（不含换行符）；EOF（EndOfStream）自然结束。"""
    buffer = bytearray()
    while True:
        try:
            chunk = await stream.receive()
        except anyio.EndOfStream:
            return
        buffer += chunk
        while (newline := buffer.find(b"\n")) >= 0:
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            yield line


async def run_tool_in_worker(
    tool_name: str, arguments: dict[str, Any], context: Any
) -> CallToolResult:
    """在 worker 子进程中执行长任务工具（#588 子进程隔离）。

    请求经 stdin 一行 JSON 下发，进度/结果行经 stdout 回流（协议见
    worker.py）。等待全部走可取消的 await：MCP 请求取消（interrupt 模式
    取消 handler 作用域）与客户端断连（EOF 取消任务组）在此汇成同一
    处理——kill 子进程、shield 内收尸、重抛取消。kill 只丢当前任务：
    catalog 落盘是一次性原子写（tmp + os.replace），已入库记录不受影响。
    worker 异常退出（无结果行）译为 WORKER_CRASHED 结构化错误。
    """
    proc = await anyio.open_process(
        _WORKER_ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None
    )
    assert proc.stdin is not None and proc.stdout is not None  # PIPE 已请求
    request = json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=True)
    await proc.stdin.send(request.encode("ascii") + b"\n")
    await proc.stdin.aclose()

    meta = getattr(context, "meta", None)
    token = getattr(meta, "progress_token", None)
    session = getattr(context, "session", None)
    result_env: envelope.Envelope | None = None
    try:
        async for line in _iter_lines(proc.stdout):
            try:
                message: dict[str, Any] = json.loads(line)
            except ValueError:
                continue  # 非法行（非 JSON）忽略：worker 崩溃即无结果行
            if message.get("type") == "progress" and token is not None and session is not None:
                # 事件循环内直投（与线程路径的 make_progress_reporter 互斥：
                # 那个经 run_coroutine_threadsafe，在循环线程内会死锁）；
                # worker 侧已节流，收到即转发。
                with contextlib.suppress(Exception):
                    await session.report_progress(
                        message.get("fraction", 0.0), 1.0, message.get("message")
                    )
            elif message.get("type") == "result":
                result_env = message.get("envelope")
        await proc.wait()
    except BaseException:
        # 取消（或传输层异常）：kill + 受限时收尸后重抛。shield 保证外层
        # 取消不打断收尾；kill 是唯一可靠打断长计算的手段（线程不可杀）。
        with anyio.CancelScope(shield=True):
            with contextlib.suppress(TimeoutError):
                with anyio.fail_after(_KILL_REAP_TIMEOUT_SEC):
                    proc.kill()
                    await proc.wait()
        raise
    if result_env is None:
        result_env = envelope.error_envelope(
            "WORKER_CRASHED", f"worker 进程未返回结果（exit={proc.returncode}）"
        )
    return _to_result(result_env)


def handle_call_tool(
    facade: Facade, name: str, arguments: dict[str, Any], extra_kwargs: dict[str, Any] | None = None
) -> CallToolResult:
    """调用工具并包信封（纯函数，便于测试）。

    ``extra_kwargs`` 是传输层注入的额外协作者（如进度回调），信封层按
    方法签名过滤（见 :func:`envelope.dispatch_tool`），未接受的工具零影响。
    """
    spec = next((s for s in tools.tool_specs(facade) if s.name == name), None)
    if spec is None:
        env = envelope.tool_not_found(name)
    else:
        env = envelope.invoke_tool(spec.method, arguments, extra_kwargs=extra_kwargs)
    return _to_result(env)


def create_server(facade: Facade) -> Server:
    """创建 MCP 服务器（绑定传入的 Facade）。

    一个 Facade 实例 = 一个 server；工具清单在每次 tools/list 时由 Facade
    纯派生，与 ``tool_inventory()`` 单一同源。注册走
    ``add_request_handler`` （mcp 1.x/2.x 兼容：2.0 移除了装饰器 API）。

    Args:
        facade: Facade 实例。

    Returns:
        ``mcp.server.Server`` 对象（配合 ``mcp.server.stdio.stdio_server`` 运行）。
    """
    server: Server = Server("e2m2e")

    async def _list_tools(context: Any, params: Any) -> ListToolsResult:
        # 纯派生不走 Facade 方法本体，直接在事件循环里做即可。
        return ListToolsResult(tools=handle_list_tools(facade))

    async def _call_tool(context: Any, params: Any) -> CallToolResult:
        # Facade 方法是同步长计算，放线程池避免阻塞事件循环；客户端
        # 请求进度时把线程安全回调作为额外协作者注入（未请求时 None，
        # 注入层按方法签名过滤，其余工具零影响）。长任务工具例外：改跑
        # worker 子进程，使取消/断连可靠传播为进程 kill（#588）。
        arguments = dict(params.arguments or {})
        if params.name in LONG_RUNNING_TOOLS:
            return await run_tool_in_worker(params.name, arguments, context)
        reporter = make_progress_reporter(context)
        extra = {"progress_callback": reporter} if reporter is not None else None
        return await anyio.to_thread.run_sync(
            lambda: handle_call_tool(facade, params.name, arguments, extra_kwargs=extra)
        )

    server.add_request_handler("tools/list", PaginatedRequestParams, _list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, _call_tool)
    return server
