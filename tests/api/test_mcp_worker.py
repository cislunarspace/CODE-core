"""长任务工具取消——子进程隔离（#588 / #576 Phase 2）。

三层接缝：
1. worker 本体：真实子进程 spawn，JSON 行协议往返（成功/错误翻译）。
2. server 侧 run_tool_in_worker：fake worker 注入 argv——进度转发、取消
   kill 收尸、崩溃信封、kill 后 catalog 完整性。
3. 协议端到端：真实 dispatcher（内存流）驱动 ``notifications/cancelled``
   与客户端断连（EOF）两条取消路径。

worker 是通用执行器（任意工具名），路由决策（哪些工具走子进程）在
server 层的 LONG_RUNNING_TOOLS——测试用便宜工具直接打 worker。
"""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [
    pytest.mark.interface,
    # kill 子进程后 asyncio proactor 管道 __del__ 的 GC 噪声（CPython 已知
    # 现象，非资源泄漏），触发时机随 GC 漂移：文件级静音
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

pytest.importorskip("mcp")  # [mcp] extra 未装时整文件跳过（协议层依赖）

from e2m2e.api.config import Config  # noqa: E402
from e2m2e.api.facade import Facade  # noqa: E402
from e2m2e.api.mcp import server as mcp_server  # noqa: E402

# ---------------------------------------------------------------------------
# 替身与辅助
# ---------------------------------------------------------------------------


class _Meta:
    def __init__(self, token: Any) -> None:
        self.progress_token = token


class _ProgressSession:
    """记录 report_progress 调用的 Session 替身（进度到达时置事件）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.progress_seen: Any = None  # anyio.Event，在事件循环内创建

    async def report_progress(
        self, progress: float, total: float | None = None, message: Any = None
    ) -> None:
        self.calls.append((progress, total, message))
        if self.progress_seen is not None:
            self.progress_seen.set()


class _Ctx:
    """MCP 请求上下文替身（meta.progress_token / session.report_progress）。"""

    def __init__(self, token: Any = None, session: Any = None) -> None:
        self.meta = _Meta(token)
        self.session = session


# fake worker：pidfile 报到 → 进度行 → 按 FAKE_WORKER_MODE 收场。
# ok：结果行后退出；sleep：睡 60s（模拟跑偏的长计算）；crash：exit(3)。
_FAKE_WORKER_SCRIPT = """
import json, os, sys, time
pidfile = os.environ.get("FAKE_WORKER_PIDFILE")
if pidfile:
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
mode = os.environ.get("FAKE_WORKER_MODE", "ok")
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
emit({"type": "progress", "fraction": 0.5, "message": "half"})
if mode == "crash":
    sys.exit(3)
if mode == "sleep":
    time.sleep(60)
emit({"type": "result", "envelope": {"status": "ok", "data": {"mode": mode},
        "error": None, "meta": {}}})
"""


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    """把 worker 命令换成 ``python -c`` fake 脚本，可切模式。"""
    pidfile = tmp_path / "fake-worker.pid"

    def use(mode: str) -> None:
        monkeypatch.setenv("FAKE_WORKER_MODE", mode)
        monkeypatch.setenv("FAKE_WORKER_PIDFILE", str(pidfile))
        monkeypatch.setattr(mcp_server, "_WORKER_ARGV", [sys.executable, "-c", _FAKE_WORKER_SCRIPT])

    use("ok")
    return SimpleNamespace(pidfile=pidfile, use=use)


def _pid_alive(pid: int) -> bool:
    """探测进程是否仍在运行（跨平台，不发送信号）。"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    import errno

    try:
        import os

        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _pid_from_file(path) -> int:
    return int(path.read_text(encoding="ascii").strip())


@pytest.fixture
def facade() -> Facade:
    # tests/api/conftest.py 已把 E2M2E_CATALOG_DIR 重定向到独立临时目录
    return Facade(Config())


# ---------------------------------------------------------------------------
# 1. 真实 worker 往返（JSON 行协议 + 信封）
# ---------------------------------------------------------------------------


def test_real_worker_roundtrip_ok():
    """真实子进程：便宜工具经 worker 往返，信封 status=ok。"""
    import anyio

    from e2m2e.api.mcp import run_tool_in_worker

    result = anyio.run(run_tool_in_worker, "catalog_query", {}, _Ctx())
    assert not result.is_error
    env = json.loads(result.content[0].text)
    assert env["status"] == "ok"
    assert set(env) == {"status", "data", "error", "meta"}


def test_real_worker_translates_validation_errors():
    """worker 侧错误翻译照旧：非法参数 → INVALID_PARAMS（无 traceback 泄漏）。"""
    import anyio

    from e2m2e.api.mcp import run_tool_in_worker

    result = anyio.run(run_tool_in_worker, "catalog_query", {"libration_point": 99}, _Ctx())
    assert result.is_error
    env = json.loads(result.content[0].text)
    assert env["status"] == "error"
    assert env["error"]["code"] == "INVALID_PARAMS"
    assert "Traceback" not in result.content[0].text


def test_real_worker_unknown_tool():
    import anyio

    from e2m2e.api.mcp import run_tool_in_worker

    result = anyio.run(run_tool_in_worker, "no_such_tool", {}, _Ctx())
    env = json.loads(result.content[0].text)
    assert env["error"]["code"] == "TOOL_NOT_FOUND"


# ---------------------------------------------------------------------------
# 2. server 侧接缝（fake worker）
# ---------------------------------------------------------------------------


def test_progress_forwarded_with_token(fake_worker):
    """worker 进度行 → session.report_progress（带 token 时转发，保序）。"""
    import anyio

    session = _ProgressSession()
    ctx = _Ctx(token="tok", session=session)

    async def scenario():
        session.progress_seen = anyio.Event()
        return await mcp_server.run_tool_in_worker("transfer_design", {}, ctx)

    result = anyio.run(scenario)
    assert [(p, t, m) for p, t, m in session.calls] == [(0.5, 1.0, "half")]
    env = json.loads(result.content[0].text)
    assert env["status"] == "ok"
    assert env["data"]["mode"] == "ok"


def test_progress_gated_without_token(fake_worker):
    """客户端未带 progressToken：进度行不转发（零开销直通），结果照常。"""
    import anyio

    session = _ProgressSession()

    async def scenario():
        return await mcp_server.run_tool_in_worker(
            "transfer_design", {}, _Ctx(token=None, session=session)
        )

    result = anyio.run(scenario)
    assert session.calls == []
    assert not result.is_error


def test_cancellation_kills_worker(fake_worker):
    """取消传播：handler 被取消 → kill 子进程并收尸，快速收尾而非等满 60s。"""
    import anyio

    fake_worker.use("sleep")
    session = _ProgressSession()
    ctx = _Ctx(token="tok", session=session)
    completed: dict[str, Any] = {}

    async def scenario() -> float:
        session.progress_seen = anyio.Event()
        started = time.monotonic()
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tg:

                async def cancel_on_progress() -> None:
                    await session.progress_seen.wait()
                    scope.cancel()

                tg.start_soon(cancel_on_progress)
                completed["result"] = await mcp_server.run_tool_in_worker(
                    "transfer_design", {}, ctx
                )
        return time.monotonic() - started

    elapsed = anyio.run(scenario)
    assert "result" not in completed, "被取消的调用不得产出结果"
    assert elapsed < 20, f"取消后须快速收尾（实测 {elapsed:.1f}s）——kill 未生效？"
    pid = _pid_from_file(fake_worker.pidfile)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    assert not _pid_alive(pid), "worker 子进程应已被终止"


def test_worker_crash_yields_error_envelope(fake_worker):
    """worker 异常退出（无结果行）→ WORKER_CRASHED 结构化错误，不炸穿 handler。"""
    import anyio

    fake_worker.use("crash")
    result = anyio.run(mcp_server.run_tool_in_worker, "transfer_design", {}, _Ctx())
    assert result.is_error
    env = json.loads(result.content[0].text)
    assert env["status"] == "error"
    assert env["error"]["code"] == "WORKER_CRASHED"


def test_catalog_records_survive_worker_kill():
    """数据安全：真实 worker 被 kill 后，已入库 catalog 记录完好可读。"""
    import anyio

    from tests.api.test_facade_catalog import _fake_design, _make_design_result

    # 经公共缝播种一条记录（fake 算法结果 → facade 自动入库）；fake 只作用于
    # 播种阶段，不影响后续真实路径与 conftest 的 catalog 目录隔离。
    with pytest.MonkeyPatch.context() as mp:
        _fake_design(mp, _make_design_result(orbit_type="DRO"))
        seed_facade = Facade(Config())
        seed_facade.design_orbit(orbit_type="DRO")

    started = time.monotonic()

    async def scenario() -> None:
        # 真实 worker（catalog_query）在任意阶段被取消（多半还在 import 期）：
        # kill 不应波及已入库记录
        with anyio.move_on_after(0.15):
            await mcp_server.run_tool_in_worker("catalog_query", {}, _Ctx())

    anyio.run(scenario)
    assert time.monotonic() - started < 20, "取消后须快速收尾"

    # kill 后重新打开库：播种记录仍在、可查（store 无跨进程内存缓存）
    env = Facade(Config()).catalog_query(orbit_family="dro")
    assert env.records, "kill 后已入库记录应完好可查"


# ---------------------------------------------------------------------------
# 3. 协议端到端：真实 dispatcher（内存流）
# ---------------------------------------------------------------------------


async def _recv_message(write_recv, timeout: float = 10.0):
    """收一条服务端消息（超时抛 TimeoutError）。"""
    import anyio

    with anyio.fail_after(timeout):
        sm = await write_recv.receive()
    return sm.message


async def _wait_pidfile(path, timeout: float = 15.0) -> int:
    import anyio

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return _pid_from_file(path)
        await anyio.sleep(0.05)
    raise AssertionError("fake worker pidfile 未出现（worker 未启动）")


async def _wait_pid_dead(pid: int, timeout: float = 20.0) -> None:
    import anyio

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        await anyio.sleep(0.1)
    raise AssertionError(f"worker pid={pid} 未被终止")


class _E2E:
    """内存流上驱动真实 server.run 的最小 MCP 客户端。"""

    def __init__(self, server) -> None:
        import anyio

        self.server = server
        self.read_send, self.read_recv = anyio.create_memory_object_stream(16)
        self.write_send, self.write_recv = anyio.create_memory_object_stream(16)

    async def send(self, message) -> None:
        from mcp.shared.message import SessionMessage

        await self.read_send.send(SessionMessage(message=message))

    async def request(self, rid: int, method: str, params: dict) -> Any:
        from mcp.types import JSONRPCRequest

        await self.send(JSONRPCRequest(jsonrpc="2.0", id=rid, method=method, params=params))
        # 跳过其间穿插的通知（进度等），等到目标响应为止
        while True:
            msg = await _recv_message(self.write_recv)
            if getattr(msg, "id", None) == rid:
                return msg

    async def notify(self, method: str, params: dict) -> None:
        from mcp.types import JSONRPCNotification

        await self.send(JSONRPCNotification(jsonrpc="2.0", method=method, params=params))

    async def handshake(self) -> None:
        from mcp.types import LATEST_PROTOCOL_VERSION

        result = await self.request(
            1,
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        assert not hasattr(result, "error") or result.error is None
        await self.notify("notifications/initialized", {})


def test_e2e_cancelled_notification_kills_worker(fake_worker, facade):
    """验收 1：客户端发 notifications/cancelled → 长任务进程被终止，
    被取消的请求不回结果，server 之后仍健康应答。"""
    import anyio
    from mcp.types import JSONRPCRequest

    from e2m2e.api.mcp import create_server

    fake_worker.use("sleep")
    server = create_server(facade)

    async def client() -> None:
        e2e = _E2E(server)
        options = server.create_initialization_options()

        async with anyio.create_task_group() as outer:

            async def serve() -> None:
                await server.run(e2e.read_recv, e2e.write_send, options)

            outer.start_soon(serve)
            await e2e.handshake()
            await e2e.send(
                JSONRPCRequest(
                    jsonrpc="2.0",
                    id=2,
                    method="tools/call",
                    params={"name": "transfer_design", "arguments": {}},
                )
            )
            pid = await _wait_pidfile(fake_worker.pidfile)
            await anyio.sleep(0.3)  # worker 进入睡眠（模拟长计算中段）
            await e2e.notify("notifications/cancelled", {"requestId": 2})
            await _wait_pid_dead(pid)
            # 被取消的请求不回结果：等 1s 只有闭流/超时
            import contextlib

            with contextlib.suppress(Exception):
                with anyio.fail_after(1.0):
                    while True:
                        msg = await e2e.write_recv.receive()
                        assert getattr(msg.message, "id", None) != 2, "取消的请求不得回结果"
            # server 仍健康
            pong = await e2e.request(3, "ping", {})
            assert getattr(pong, "result", None) is not None or not hasattr(pong, "error")
            await e2e.read_send.aclose()
            # server.run 随 EOF 返回，任务组自然收束

    anyio.run(client)


def test_e2e_client_disconnect_kills_worker(fake_worker, facade):
    """验收 2：客户端断连（EOF）同样触发取消传播，worker 被终止。"""
    import anyio
    from mcp.types import JSONRPCRequest

    from e2m2e.api.mcp import create_server

    fake_worker.use("sleep")
    server = create_server(facade)

    async def client() -> None:
        e2e = _E2E(server)
        options = server.create_initialization_options()
        served: dict[str, Any] = {}

        async with anyio.create_task_group() as outer:

            async def serve() -> None:
                await server.run(e2e.read_recv, e2e.write_send, options)
                served["done"] = True

            outer.start_soon(serve)
            await e2e.handshake()
            await e2e.send(
                JSONRPCRequest(
                    jsonrpc="2.0",
                    id=2,
                    method="tools/call",
                    params={"name": "orbit_family_generation", "arguments": {}},
                )
            )
            pid = await _wait_pidfile(fake_worker.pidfile)
            await anyio.sleep(0.3)
            await e2e.read_send.aclose()  # 断连
            await _wait_pid_dead(pid)
            assert served.get("done") is True, "EOF 后 server.run 应返回"

    anyio.run(client)
