"""GUI sidecar stdio 协议（issue #518 / ADR 0035）。

tod 的 Tauri 壳以常驻子进程驱动 e2m2e：请求/响应/进度是 JSON 文本行，
复用 MCP 方向的统一信封（``{status, data, error, meta}``，单一来源在
``e2m2e/api/mcp/envelope.py``）；响应含大数组且请求声明 ``binary_dtype``
时，JSON 行带 ``"binary_frames": N``，换行符后紧跟 N 个二进制帧，帧后恢复
JSON 行流（帧格式见 ``frames.py``）。工具面 = Facade 上 ``mcp_exposed``
的方法（纯派生，ADR 0014），不新增业务逻辑。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..mcp import envelope, tools
from .frames import FrameError, encode_frame

if TYPE_CHECKING:
    from typing import BinaryIO

    from ..facade import Facade

__all__ = ["handle_request", "run_loop"]

# 请求方可声明的二进制 dtype（ADR 0035 决策 1：渲染 f32，复算量 f64）。
_BINARY_DTYPES = ("f32", "f64")

# 大数组走二进制帧的工具→响应字段映射。首个落地的画布契约：族生成响应的
# 成员状态数组（Orbit 对象不可 JSON 化，本就必须走帧）。其余工具等消费端
# （tod 阶段 2）出现真实需求再补，不预先铺开。
_BINARY_TOOLS = frozenset({"orbit_family_generation"})


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


def _family_binary_payload(result: Any, dtype: str) -> tuple[dict[str, Any], list[bytes]]:
    """族生成响应的画布契约：成员状态数组出帧，JSON 行留占位。

    帧序 = ``data.orbits`` 成员序；每帧是该成员的 ``(n, 6)`` 状态数组。
    times 等小数组留在 JSON。``states`` 置 None 占位，帧是唯一真身。
    """
    frames = [encode_frame(orbit.states, dtype) for orbit in result.orbits]
    data = {
        "status": result.status.value,
        "cause": result.cause.value,
        "message": result.message,
        "family_type": result.family_type,
        "requested_members": result.requested_members,
        "generated_members": result.generated_members,
        "record_id": result.record_id,
        "orbits": [
            {"states": None, "times": [float(t) for t in orbit.times]} for orbit in result.orbits
        ],
    }
    return data, frames


def handle_request(facade: Facade, request: Any) -> list[bytes]:
    """处理一个已解析的请求，返回待写出的字节块。

    字节块是零个或多个 JSON 行（含换行符）加原始帧字节；调用方顺序写出
    即得到协议流。
    """
    if not isinstance(request, dict) or not isinstance(request.get("tool"), str):
        return [
            _line(envelope.error_envelope("INVALID_PARAMS", "请求必须是含 tool 字段的 JSON 对象"))
        ]
    tool: str = request["tool"]
    spec = next((s for s in tools.tool_specs(facade) if s.name == tool), None)
    if spec is None:
        return [_line(envelope.error_envelope("UNKNOWN_TOOL", f"未知工具 {tool!r}"))]
    dtype = request.get("binary_dtype")
    if dtype is not None and dtype not in _BINARY_DTYPES:
        return [
            _line(
                envelope.error_envelope(
                    "INVALID_PARAMS",
                    f"binary_dtype 必须是 {_BINARY_DTYPES} 之一或缺省，得到 {dtype!r}",
                )
            )
        ]
    job_id = request.get("job_id")
    arguments = request.get("arguments") or {}
    started = _progress_line(job_id, 0.0, f"开始 {tool}")
    chunks = [started]

    if tool in _BINARY_TOOLS:
        if dtype is None:
            # 协议约定（非参数校验）：响应含 Orbit 对象不可 JSON 化（MCP
            # 传输层同此限制），该工具必须走帧，缺省 dtype 视为协议用法
            # 错误，借用 INVALID_PARAMS 错误码。
            return [
                _line(
                    envelope.error_envelope(
                        "INVALID_PARAMS",
                        f"工具 {tool} 的响应含大数组，请求必须声明 binary_dtype（f32/f64）",
                    )
                )
            ]
        result, err = envelope.dispatch_tool(spec.method, arguments)
        if err is not None:
            return [started, _line(err)]
        try:
            data, frames = _family_binary_payload(result, dtype)
        except FrameError as exc:
            return [started, _line(envelope.error_envelope("INTERNAL_ERROR", f"帧编码失败：{exc}"))]
        response = envelope.ok_envelope(data)
        response["binary_frames"] = len(frames)
        return [*chunks, _line(response), *frames]

    return [*chunks, _line(envelope.invoke_tool(spec.method, arguments))]


def run_loop(facade: Facade, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """std io 主循环：逐行读请求 JSON，写出进度/响应行与二进制帧。

    坏 JSON 行返回错误信封并继续，不中断循环。
    """
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            chunks = [_line(envelope.error_envelope("INVALID_PARAMS", "请求行不是合法 JSON"))]
        else:
            chunks = handle_request(facade, request)
        for chunk in chunks:
            stdout.write(chunk)
        stdout.flush()
