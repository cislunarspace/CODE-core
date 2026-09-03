"""传输中立执行核心：工具执行的单一入口（#601，ADR 0014 增补）。

工具面派生（mcp/tools.py）、执行策略（:data:`LONG_RUNNING_TOOLS`）、二进制
帧画布契约（本模块 ``_BINARY_TOOLS`` 一族）在此各只有一份定义；MCP server
与 sidecar 退成薄适配器——前者包 ``CallToolResult``，后者包 JSON 行加帧。
worker 子进程本体（mcp/worker.py）同样经 :func:`execute_tool` 执行。

本模块不 import mcp SDK、不 import anyio（worker 与 sidecar 同款约束，缺
``[mcp]`` extra 也能跑）；异步 spawn 的泵在 mcp/server.py。
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from .catalog_ingest import finite_or_none
from .config import Config
from .frames import FrameError, encode_frame
from .mcp import envelope, tools

if TYPE_CHECKING:
    from ..facade import Facade

__all__ = [
    "BINARY_DTYPES",
    "BINARY_FRAME_TOOLS",
    "LONG_RUNNING_TOOLS",
    "WORKER_ARGV",
    "execute_tool",
    "preflight",
    "worker_request_payload",
]

# 长任务工具（#588 / #576 Phase 2，子进程隔离架构）：分钟级计算改跑 worker
# 子进程，使取消（对端 cancel / 断连 EOF）可靠传播为进程 kill。执行策略是
# 两个传输层共同消费的关注点，单一清单在此（#601）；何时 spawn 归各适配器。
LONG_RUNNING_TOOLS = frozenset({"transfer_design", "orbit_family_generation"})

# 请求方可声明的二进制 dtype（ADR 0035 决策 1：渲染 f32，复算量 f64）。
BINARY_DTYPES = ("f32", "f64")

# worker 子进程命令（#607 起两个传输层共用；测试注入 fake worker 时
# monkeypatch 此常量）。
WORKER_ARGV = [sys.executable, "-m", "e2m2e.api.mcp.worker"]


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
        "mu": finite_or_none(getattr(result.system, "mu", None)),
        "requested_members": result.requested_members,
        "generated_members": result.generated_members,
        "family_id": result.family_id,
        "orbits": [
            {
                "states": None,
                "times": [float(t) for t in orbit.times],
                "period": finite_or_none(orbit.period),
            }
            for orbit in result.orbits
        ],
    }
    return data, frames


def _catalog_binary_payload(result: Any, dtype: str) -> tuple[dict[str, Any], list[bytes]]:
    """catalog_get 响应的画布契约：数组段 ndarray 出帧。

    帧序 = JSON 行 ``data.arrays`` 中 None 占位键的顺序；每帧是对应键的
    数组。非数组值原样留 JSON。元数据、标量段不受影响。
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
    frames = [
        encode_frame(np.asarray(result.ybar_field, dtype=float), dtype),
        encode_frame(np.asarray(result.fate_ids, dtype=float), dtype),
        encode_frame(np.asarray(result.t_escape_years_field, dtype=float), dtype),
        encode_frame(np.asarray(result.min_r_sel_km_field, dtype=float), dtype),
        encode_frame(np.asarray(result.min_r_geo_km_field, dtype=float), dtype),
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

#: 响应含大数组、走二进制帧的工具名（sidecar 协议的 requires-dtype 规则
#: 消费这份清单；ADR 0035）。
BINARY_FRAME_TOOLS = frozenset(_BINARY_TOOLS)


def preflight(facade: Facade, tool: str, binary_dtype: Any = None) -> envelope.Envelope | None:
    """执行前置校验：未知工具（含 placeholder）与非法 dtype。

    合法返回 None；非法返回错误信封（不执行）。调用方（sidecar）在产出
    进度行之前用它决定是否短路——规则单一来源在此，行数策略归适配器。
    """
    if tools.tool_spec(facade, tool) is None:
        return envelope.tool_not_found(tool)
    if binary_dtype is not None and binary_dtype not in BINARY_DTYPES:
        return envelope.error_envelope(
            "INVALID_PARAMS",
            f"binary_dtype 必须是 {BINARY_DTYPES} 之一或缺省，得到 {binary_dtype!r}",
        )
    return None


def execute_tool(
    facade: Facade,
    tool: str,
    arguments: dict[str, Any],
    *,
    progress_callback: Any = None,
    binary_dtype: Any = None,
) -> tuple[envelope.Envelope, list[bytes]]:
    """执行一个工具调用，返回 ``(统一信封, 二进制帧序列)``。

    唯一执行入口（#601）：规格查找、校验、错误翻译、画布帧抽取都只在
    此。``binary_dtype`` 缺省或工具不在帧清单内时走信封内联降级（MCP 纯
    文本通道）；声明 dtype 且工具在帧清单内时数组出帧，信封里的对应字段
    置 None 占位（``binary_frames`` 计数是 sidecar 协议字段，由适配器补）。
    ``progress_callback`` 按方法签名过滤（见 :func:`envelope.dispatch_tool`），
    未接受的工具零影响。
    """
    spec = tools.tool_spec(facade, tool)
    if spec is None:
        return envelope.tool_not_found(tool), []
    payload_fn = _BINARY_TOOLS.get(tool) if binary_dtype is not None else None
    extra = {"progress_callback": progress_callback} if progress_callback is not None else None
    if payload_fn is not None:
        result, err = envelope.dispatch_tool(spec.method, arguments, extra_kwargs=extra)
        if err is not None:
            return err, []
        try:
            data, frames = payload_fn(result, binary_dtype)
        except FrameError as exc:
            return envelope.error_envelope("INTERNAL_ERROR", f"帧编码失败：{exc}"), []
        return envelope.ok_envelope(data), frames
    return envelope.invoke_tool(spec.method, arguments, extra_kwargs=extra), []


def worker_request_payload(tool: str, arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    """worker 子进程请求行载荷（``{"tool", "arguments", "config"}``）。

    注入的 :class:`Config` 随请求下发（#601）：worker 用它重建 Facade，
    不从环境变量静默重建；载荷形状的唯一定义在此，server 构建与 worker
    解析共用。
    """
    return {"tool": tool, "arguments": arguments, "config": config.to_payload()}
