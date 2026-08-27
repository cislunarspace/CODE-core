"""统一信封：MCP 传输层输出形状（ADR 0014 §4）。

所有工具调用的返回都包成 ``{status, data, error, meta}``：成功时
``status="ok"``、``data`` 为 Response 模型的 JSON 序列化；失败时
``status="error"``、``error`` 为结构化错误（错误码 + message + details）。
异常在 api/ 边界翻译，不向 Agent 泄漏原始 traceback。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from e2m2e.data.types.orbit import Orbit

from ..catalog_ingest import _finite_or_none
from ..models import OrbitError

__all__ = ["Envelope", "ok_envelope", "error_envelope", "invoke_tool", "dispatch_tool"]

# 信封的 JSON 形状（不是 Pydantic 模型：传输层只做序列化，不做自校验）。
Envelope = dict[str, Any]


def _to_jsonable(value: Any) -> Any:
    """把 ``model_dump(mode="python")`` 输出里的不可 JSON 化值降级转换。

    MCP 是纯文本通道（sidecar 的大数组走二进制帧，ADR 0035）：Any/
    任意类型字段携带的 ndarray 转嵌套 list；Orbit 成员取画布契约字段
    （states/times/period/family_type，与 sidecar 帧契约同款）；System
    鸭子类型只透传 ``mu`` 标量；其余未知对象以 ``<类型名>`` 占位。总线
    保证结果可 JSON 序列化，不向传输层抛异常。
    """
    if isinstance(value, BaseModel):
        return _jsonify(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Orbit):
        return {
            "states": _to_jsonable(value.states),
            "times": _to_jsonable(value.times),
            "period": _finite_or_none(value.period),
            "family_type": value.family_type,
        }
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    mu = getattr(value, "mu", None)
    if mu is not None:
        return {"mu": _finite_or_none(mu)}
    return f"<{type(value).__name__}>"


def _jsonify(data: Any) -> Any:
    """把 Response 模型/普通对象转成可 JSON 序列化的结构。"""
    if isinstance(data, BaseModel):
        try:
            return data.model_dump(mode="json")
        except Exception:
            # Any/任意类型字段携带 ndarray/Orbit/System（族生成、
            # catalog_get/promote）时 mode="json" 序列化失败，回落为
            # python 模式转储 + 逐值转换。
            return _to_jsonable(data.model_dump(mode="python"))
    return data


def ok_envelope(data: Any, meta: dict[str, Any] | None = None) -> Envelope:
    """构造成功信封。"""
    return {"status": "ok", "data": _jsonify(data), "error": None, "meta": meta or {}}


def error_envelope(code: str, message: str, details: dict[str, Any] | None = None) -> Envelope:
    """构造失败信封。"""
    return {
        "status": "error",
        "data": None,
        "error": {"code": code, "message": message, "details": details or {}},
        "meta": {},
    }


def dispatch_tool(method: Any, arguments: dict[str, Any]) -> tuple[Any, Envelope | None]:
    """校验参数并执行工具方法，返回 ``(原始结果, None)`` 或 ``(None, 错误信封)``。

    与 :func:`invoke_tool` 同一套校验与错误翻译，但不做结果的 JSON 化——
    需要在转储前处理原始结果的传输层（如 sidecar 的二进制帧抽取，ADR 0035）
    用这个。失败时返回错误信封，不抛异常。
    """
    request_model = getattr(method, "request_model", None)
    try:
        if request_model is not None:
            request = request_model.model_validate(arguments)
            result = method(**request.model_dump(exclude_unset=True))
        else:
            result = method(**arguments)
    except OrbitError as exc:
        return None, error_envelope(exc.code, exc.message, exc.details)
    except ValidationError as exc:
        # exc.json() 走 Pydantic 自己的序列化，避免 details 里残留不可 JSON 化对象。
        return None, error_envelope(
            "INVALID_PARAMS",
            "输入参数校验失败",
            {"errors": json.loads(exc.json(include_url=False))},
        )
    except TypeError as exc:
        # 方法签名不匹配（如多余参数已由 forbid 校验拦下，此处兜底）。
        return None, error_envelope("INVALID_PARAMS", f"参数与工具签名不匹配：{exc}")
    except Exception as exc:
        return None, error_envelope("INTERNAL_ERROR", f"未预期的内部错误（{type(exc).__name__}）")
    return result, None


def invoke_tool(method: Any, arguments: dict[str, Any]) -> Envelope:
    """执行一个 Facade 工具方法并包成信封。

    入参经 ``request_model`` 校验（ValidationError → ``INVALID_PARAMS``），
    ``OrbitError`` 原样翻译，其余异常归为 ``INTERNAL_ERROR``（只保留异常类型
    名，不泄漏 traceback）。校验与错误翻译见 :func:`dispatch_tool`。
    """
    result, err = dispatch_tool(method, arguments)
    if err is not None:
        return err
    try:
        return ok_envelope(result)
    except Exception as exc:
        # 结果含不可 JSON 化对象（如 ndarray）：兑成结构化错误而非炸穿
        # 传输层（MCP 与 sidecar 共用此处）。
        return error_envelope("INTERNAL_ERROR", f"响应序列化失败（{type(exc).__name__}）")
