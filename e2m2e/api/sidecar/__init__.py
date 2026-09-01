"""GUI sidecar stdio 协议（ADR 0035；#607 增补任务取消与真实进度）。

tod 的 Tauri 壳以常驻子进程驱动 e2m2e：请求/响应/进度是 JSON 文本行，
复用 MCP 方向的统一信封（``{status, data, error, meta}``，单一来源在
``e2m2e/api/mcp/envelope.py``）；响应含大数组且请求声明 ``binary_dtype``
时，JSON 行带 ``"binary_frames": N``，换行符后紧跟 N 个二进制帧，帧后恢复
JSON 行流（帧格式见 ``frames.py``）。工具面 = Facade 上 ``mcp_exposed``
的方法（纯派生，ADR 0014），不新增业务逻辑。

并发模型（#607）：短任务进程内同步执行；长任务（执行核心
``LONG_RUNNING_TOOLS``）在 worker 子进程 + 后台线程执行，读环持续消费
stdin——取消消息（``{"cancel": "<job_id>"}`` 行）中途可达，kill 子进程
即取消，回 ``status="cancelled"`` 行；被取消的原请求不回结果行（对齐
MCP 语义）。单 job 的输出字节（响应行 + 帧）在锁内原子写出。

本模块是薄适配器（#601）：工具面、校验规则、画布帧抽取的单一来源在执行
核心 ``e2m2e/api/execution.py``，这里只做协议形状转换——JSON 行切分、
job_id 进度行、``binary_frames`` 计数与帧字节排序、取消与生命周期。
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import TYPE_CHECKING, Any

from .. import execution
from ..execution import (
    BINARY_FRAME_TOOLS,
    LONG_RUNNING_TOOLS,
    execute_tool,
    preflight,
    worker_request_payload,
)
from ..frames import FrameError, read_raw_frame
from ..mcp import envelope

if TYPE_CHECKING:
    from typing import BinaryIO

    from ..facade import Facade

__all__ = ["handle_request", "run_loop"]


def _line(payload: Any) -> bytes:
    """JSON 行（含换行符）。"""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _progress_line(job_id: Any, percent: float, message: str) -> bytes:
    """进度行：可丢弃的信封 JSON 行（ADR 0035 决策 3）。"""
    return _line(
        {
            "status": "progress",
            "data": None,
            "error": None,
            "meta": {"job_id": job_id, "percent": percent, "message": message},
        }
    )


def _cancelled_line(job_id: Any) -> bytes:
    """取消确认行（#607）：幂等——未知/已结束的 job 也回此行。"""
    return _line(
        {
            "status": "cancelled",
            "data": None,
            "error": None,
            "meta": {"job_id": job_id},
        }
    )


def _request_error(facade: Facade, request: Any) -> envelope.Envelope | None:
    """请求级校验（单一规则来源在执行核心，此处只拼协议错误码）。

    形状非法 / 未知工具 / 非法 binary_dtype / 帧工具缺 dtype，返回错误
    信封；合法返回 None。
    """
    if not isinstance(request, dict) or not isinstance(request.get("tool"), str):
        return envelope.error_envelope("INVALID_PARAMS", "请求必须是含 tool 字段的 JSON 对象")
    tool: str = request["tool"]
    dtype = request.get("binary_dtype")
    err = preflight(facade, tool, dtype)
    if err is None and dtype is None and tool in BINARY_FRAME_TOOLS:
        # 协议约定（非参数校验）：响应含 ndarray/Orbit 对象，本协议要求
        # 走帧（执行核心在缺省 dtype 时走内联降级，MCP 通道即此用法），
        # 缺省 dtype 视为协议用法错误，借用 INVALID_PARAMS 错误码。
        err = envelope.error_envelope(
            "INVALID_PARAMS", f"工具 {tool} 的响应含大数组，请求必须声明 binary_dtype（f32/f64）"
        )
    return err


def handle_request(facade: Facade, request: Any) -> list[bytes]:
    """处理一个已解析的请求（内联执行），返回待写出的字节块。

    字节块是零个或多个 JSON 行（含换行符）加原始帧字节；调用方顺序写出
    即得到协议流。校验与执行委托执行核心。注意：长任务的内联执行不可
    取消——并发读环（:func:`run_loop`）会把长任务路由到 worker 子进程，
    本函数是单请求缝（测试与无取消消费方用）。
    """
    err = _request_error(facade, request)
    if err is not None:
        return [_line(err)]
    tool: str = request["tool"]
    dtype = request.get("binary_dtype")
    job_id = request.get("job_id")
    started = _progress_line(job_id, 0.0, f"开始 {tool}")

    env, frames = execute_tool(facade, tool, request.get("arguments") or {}, binary_dtype=dtype)
    if frames:
        env["binary_frames"] = len(frames)
    return [started, _line(env), *frames]


# ---------------------------------------------------------------------------
# 并发读环（#607）
# ---------------------------------------------------------------------------


class _Job:
    """一个在飞长任务：取消事件 + 子进程槽位（读环同步登记，杜绝竞态）。"""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.proc: subprocess.Popen | None = None


def _read_worker_message(stdout: Any) -> tuple[dict[str, Any] | None, list[bytes]]:
    """从 worker stdout 读一条 JSON 行及其后声明的全部原始帧。

    EOF 返回 ``(None, [])``；JSON 行坏则跳过继续读（worker 崩溃即无
    结果行）。帧按结果行 ``binary_frames`` 声明消费。
    """
    raw = stdout.readline()
    if not raw:
        return None, []
    try:
        message = json.loads(raw)
    except ValueError:
        return None, []
    if not isinstance(message, dict):
        return None, []
    frames: list[bytes] = []
    for _ in range(message.get("binary_frames", 0) or 0):
        frames.append(read_raw_frame(stdout))
    return message, frames


def _run_long_job(
    facade: Facade, request: dict[str, Any], job: _Job, job_id: Any, write: Any
) -> None:
    """长任务的 worker 子进程泵（后台线程）：进度行转发，结果行 + 帧原子写出。

    崩溃（无结果行）回 WORKER_CRASHED；取消（kill 致 EOF）静默收场——
    cancelled 行已由取消处理器发出。帧解码失败回 INTERNAL_ERROR。
    """
    tool: str = request["tool"]
    try:
        # 运行时属性访问（非 from-import）：测试 patch execution.WORKER_ARGV
        # 注入 fake worker 时必须生效。
        proc = subprocess.Popen(
            execution.WORKER_ARGV,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
    except Exception as exc:
        write(_line(envelope.error_envelope("WORKER_CRASHED", f"worker 启动失败：{exc}")))
        return
    job.proc = proc
    if job.cancel.is_set():
        # 取消在读环同步登记后、子进程创建前就已到达：立即收场。
        proc.kill()
        proc.wait()
        return
    assert proc.stdin is not None and proc.stdout is not None  # PIPE 已请求
    payload = worker_request_payload(tool, request.get("arguments") or {}, facade.config)
    payload["binary_dtype"] = request.get("binary_dtype")
    try:
        proc.stdin.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("ascii"))
        proc.stdin.close()
    except Exception as exc:
        proc.kill()
        proc.wait()
        write(_line(envelope.error_envelope("WORKER_CRASHED", f"worker 请求下发失败：{exc}")))
        return

    result_sent = False
    try:
        while True:
            message, frames = _read_worker_message(proc.stdout)
            if message is None:
                break
            if message.get("type") == "progress":
                write(
                    _progress_line(
                        job_id, message.get("fraction", 0.0), message.get("message") or ""
                    )
                )
            elif message.get("type") == "result":
                env = message.get("envelope")
                assert env is not None, "result 行必带 envelope（worker 协议）"
                if frames:
                    env["binary_frames"] = len(frames)
                write(_line(env), *frames)
                result_sent = True
                break
        proc.wait()
    except FrameError as exc:
        write(_line(envelope.error_envelope("INTERNAL_ERROR", f"帧解码失败：{exc}")))
    finally:
        if not job.cancel.is_set() and proc.poll() is None:
            proc.kill()
            proc.wait()
    if not result_sent and not job.cancel.is_set():
        write(
            _line(
                envelope.error_envelope(
                    "WORKER_CRASHED", f"worker 进程未返回结果（exit={proc.returncode}）"
                )
            )
        )


def _handle_cancel(request: dict[str, Any], jobs: dict[str, _Job], write: Any) -> None:
    """取消行处理：kill 在飞子进程（幂等），回 cancelled 行。"""
    job_id = request.get("cancel")
    job = jobs.pop(job_id, None) if isinstance(job_id, str) else None
    if job is not None:
        job.cancel.set()
        proc = job.proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
    write(_cancelled_line(job_id))


def run_loop(facade: Facade, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """stdio 主循环：逐行读请求 JSON，写出进度/响应行与二进制帧。

    坏 JSON 行返回错误信封并继续，不中断循环。请求处理的任何未预期
    异常（含响应信封化失败）同样兑成 INTERNAL_ERROR 信封。长任务在
    worker 子进程 + 后台线程执行（读环持续可读取消行）；单 job 的输出
    字节在锁内原子写出，帧永不交叉。
    """
    out_lock = threading.Lock()
    jobs: dict[str, _Job] = {}

    def write(*chunks: bytes) -> None:
        with out_lock:
            for chunk in chunks:
                stdout.write(chunk)
            stdout.flush()

    for raw in stdin:
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            write(_line(envelope.error_envelope("INVALID_PARAMS", "请求行不是合法 JSON")))
            continue
        if isinstance(request, dict) and "cancel" in request:
            _handle_cancel(request, jobs, write)
            continue
        err = _request_error(facade, request)
        if err is not None:
            write(_line(err))
            continue
        tool: str = request["tool"]
        if tool in LONG_RUNNING_TOOLS:
            job_id = request.get("job_id")
            job = _Job()
            if isinstance(job_id, str):
                jobs[job_id] = job  # 读环同步登记：后续取消行必然命中
            threading.Thread(
                target=_run_long_job,
                args=(facade, request, job, job_id, write),
                daemon=True,
            ).start()
            continue
        try:
            chunks = handle_request(facade, request)
        except Exception as exc:
            chunks = [
                _line(
                    envelope.error_envelope(
                        "INTERNAL_ERROR",
                        f"响应处理失败（{type(exc).__name__}）",
                    )
                )
            ]
        write(*chunks)
