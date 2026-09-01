"""GUI sidecar stdio 协议（ADR 0035）。

tod 的 Tauri 壳以常驻子进程驱动 e2m2e：请求/响应/进度是 JSON 文本行，
复用 MCP 方向的统一信封（``{status, data, error, meta}``，单一来源在
``e2m2e/api/mcp/envelope.py``）；响应含大数组且请求声明 ``binary_dtype``
时，JSON 行带 ``"binary_frames": N``，换行符后紧跟 N 个二进制帧，帧后恢复
JSON 行流（帧格式见 ``frames.py``）。工具面 = Facade 上 ``mcp_exposed``
的方法（纯派生，ADR 0014），不新增业务逻辑。

本模块是薄适配器（#601）：工具面、校验规则、画布帧抽取的单一来源在执行
核心 ``e2m2e/api/execution.py``，这里只做协议形状转换——JSON 行切分、
job_id 进度行、``binary_frames`` 计数与帧字节排序。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..execution import BINARY_FRAME_TOOLS, execute_tool, preflight
from ..mcp import envelope

if TYPE_CHECKING:
    from typing import BinaryIO

    from ..facade import Facade

__all__ = ["handle_request", "run_loop"]


def _line(payload: Any) -> bytes:
    """JSON 行（含换行符）。"""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _progress_line(job_id: Any, percent: float, message: str) -> bytes:
    """进度行：可丢弃的信封 JSON 行（ADR 0035 决策 3）。

    当前进度行只有任务开始事件（percent=0）：真进度需要算法层回调
    通道，Facade 尚未提供；待消费端（tod）需要时再接，不在本层虚构。
    """
    return _line(
        {
            "status": "progress",
            "data": None,
            "error": None,
            "meta": {"job_id": job_id, "percent": percent, "message": message},
        }
    )


def handle_request(facade: Facade, request: Any) -> list[bytes]:
    """处理一个已解析的请求，返回待写出的字节块。

    字节块是零个或多个 JSON 行（含换行符）加原始帧字节；调用方顺序写出
    即得到协议流。校验与执行委托执行核心；本模块保有行数策略——前置
    校验失败时单行返回（不产出进度行），执行结果前总是先产出进度行。
    """
    if not isinstance(request, dict) or not isinstance(request.get("tool"), str):
        return [
            _line(envelope.error_envelope("INVALID_PARAMS", "请求必须是含 tool 字段的 JSON 对象"))
        ]
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
    if err is not None:
        return [_line(err)]
    job_id = request.get("job_id")
    started = _progress_line(job_id, 0.0, f"开始 {tool}")

    env, frames = execute_tool(facade, tool, request.get("arguments") or {}, binary_dtype=dtype)
    if frames:
        env["binary_frames"] = len(frames)
    return [started, _line(env), *frames]


def run_loop(facade: Facade, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """std io 主循环：逐行读请求 JSON，写出进度/响应行与二进制帧。

    坏 JSON 行返回错误信封并继续，不中断循环。请求处理的任何未预期
    异常（含响应信封化失败）同样兑成 INTERNAL_ERROR 信封：
    字节块在 handle_request 返回前不落盘，故不存在半写出的帧污染流。
    """
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            chunks = [_line(envelope.error_envelope("INVALID_PARAMS", "请求行不是合法 JSON"))]
        else:
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
        for chunk in chunks:
            stdout.write(chunk)
        stdout.flush()
