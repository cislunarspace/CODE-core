"""GUI sidecar stdio 协议（ADR 0035）。

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

import numpy as np

from ..catalog_ingest import _finite_or_none
from ..mcp import envelope, tools
from .frames import FrameError, encode_frame

if TYPE_CHECKING:
    from typing import BinaryIO

    from ..facade import Facade

__all__ = ["handle_request", "run_loop"]

# 请求方可声明的二进制 dtype（ADR 0035 决策 1：渲染 f32，复算量 f64）。
_BINARY_DTYPES = ("f32", "f64")


# 大数组走二进制帧的工具→响应帧抽取函数映射。
# 族生成响应的成员状态数组出帧（Orbit 对象不可 JSON 化，本就必须走帧）；
# catalog_get 的记录数组段同理。其余工具等消费端（tod）出现真实需求再补，
# 不预先铺开。
def _family_binary_payload(result: Any, dtype: str) -> tuple[dict[str, Any], list[bytes]]:
    """族生成响应的画布契约：成员状态数组出帧，JSON 行留占位。

    帧序 = ``data.orbits`` 成员序；每帧是该成员的 ``(n, 6)`` 状态数组。
    times 等小数组留在 JSON。``states`` 置 None 占位，帧是唯一真身。
    成员级透传 ``period``、响应级透传 ``mu``（原值，归一化单位），供
    消费端从 ``(1, 6)`` 初态重采样整条轨迹。
    """
    frames = [encode_frame(orbit.states, dtype) for orbit in result.orbits]
    data = {
        "status": result.status.value,
        "cause": result.cause.value,
        "message": result.message,
        "family_type": result.family_type,
        "mu": _finite_or_none(getattr(result.system, "mu", None)),
        "requested_members": result.requested_members,
        "generated_members": result.generated_members,
        "record_id": result.record_id,
        "orbits": [
            {
                "states": None,
                "times": [float(t) for t in orbit.times],
                "period": _finite_or_none(orbit.period),
            }
            for orbit in result.orbits
        ],
    }
    return data, frames


def _catalog_binary_payload(result: Any, dtype: str) -> tuple[dict[str, Any], list[bytes]]:
    """catalog_get 响应的画布契约：数组段 ndarray 出帧。

    帧序 = JSON 行 ``data.arrays`` 中 None 占位键的顺序；每帧是对应键的
    数组。非数组值原样留 JSON。元数据、标量段、成员参数表不受影响。
    本函数与族生成的帧抽取共用 states/times/period/mu 同一套画布契约
    字段（当前各自独立实现，待第二个消费点出现再收拢）。
    """
    frames = []
    arrays: dict[str, Any] = {}
    for key, value in result.arrays.items():
        if isinstance(value, np.ndarray):
            frames.append(encode_frame(value, dtype))
            arrays[key] = None
        else:
            arrays[key] = value
    data = result.model_dump(mode="json", exclude={"arrays"})
    data["arrays"] = arrays
    return data, frames


def _map_binary_payload(result: Any, dtype: str) -> tuple[dict[str, Any], list[bytes]]:
    """spatiography_dynamical_map 响应的画布契约：五个场出帧，JSON 留占位。

    帧序 = ``ybar_field`` / ``fate_ids``（类 id 以 f32 编码）/ ``t_escape_years_field``
    / ``min_r_sel_km_field`` / ``min_r_geo_km_field``，均为 (n_a, n_e)，
    行序 = a 轴；NaN 帧内保留（终端短路格无变分值）。两轴、图例、阈值、
    场景与详情留 JSON。
    """
    import numpy as _np

    frames = [
        encode_frame(_np.asarray(result.ybar_field, dtype=float), dtype),
        encode_frame(_np.asarray(result.fate_ids, dtype=float), dtype),
        encode_frame(_np.asarray(result.t_escape_years_field, dtype=float), dtype),
        encode_frame(_np.asarray(result.min_r_sel_km_field, dtype=float), dtype),
        encode_frame(_np.asarray(result.min_r_geo_km_field, dtype=float), dtype),
    ]
    data = {
        "status": result.status.value,
        "cause": result.cause.value,
        "message": result.message,
        "zone": result.zone,
        "model": result.model,
        "span_years": result.span_years,
        "a_over_a_moon": [float(v) for v in result.a_over_a_moon],
        "e_grid": [float(v) for v in result.e_grid],
        "ybar_field": None,
        "fate_ids": None,
        "t_escape_years_field": None,
        "min_r_sel_km_field": None,
        "min_r_geo_km_field": None,
        "diagnostic_focus": result.diagnostic_focus,
        "thresholds": dict(result.thresholds),
    }
    return data, frames


_BINARY_TOOLS: dict[str, Any] = {
    "orbit_family_generation": _family_binary_payload,
    "catalog_get": _catalog_binary_payload,
    "spatiography_dynamical_map": _map_binary_payload,
}


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

    payload_fn = _BINARY_TOOLS.get(tool)
    if payload_fn is not None:
        if dtype is None:
            # 协议约定（非参数校验）：响应含 ndarray/Orbit 对象，本协议要求
            # 走帧（MCP 传输层由 envelope 降级内联 JSON，帧仍是首选通道），
            # 缺省 dtype 视为协议用法错误，借用 INVALID_PARAMS 错误码。
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
            data, frames = payload_fn(result, dtype)
        except FrameError as exc:
            return [started, _line(envelope.error_envelope("INTERNAL_ERROR", f"帧编码失败：{exc}"))]
        response = envelope.ok_envelope(data)
        response["binary_frames"] = len(frames)
        return [*chunks, _line(response), *frames]

    return [*chunks, _line(envelope.invoke_tool(spec.method, arguments))]


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
