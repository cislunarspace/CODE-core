"""统一信封：MCP 传输层输出形状（ADR 0014 §4）。

所有工具调用的返回都包成 ``{status, data, error, meta}``：成功时
``status="ok"``、``data`` 为 Response 模型的 JSON 序列化；失败时
``status="error"``、``error`` 为结构化错误（错误码 + message + details）。
异常在 api/ 边界翻译，不向 Agent 泄漏原始 traceback。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import OrbitError

__all__ = ["Envelope", "ok_envelope", "error_envelope", "invoke_tool"]

# 信封的 JSON 形状（不是 Pydantic 模型：传输层只做序列化，不再校验自己）。
Envelope = dict[str, Any]


def _jsonify(data: Any) -> Any:
    """把 Response 模型/普通对象转成可 JSON 序列化的结构。"""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
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


def invoke_tool(method: Any, arguments: dict[str, Any]) -> Envelope:
    """执行一个 Facade 工具方法并包成信封。

    入参经 ``request_model`` 校验（ValidationError → ``INVALID_PARAMS``），
    ``OrbitError`` 原样翻译，其余异常归为 ``INTERNAL_ERROR``（只保留异常类型
    名，不泄漏 traceback）。
    """
    request_model = getattr(method, "request_model", None)
    try:
        if request_model is not None:
            request = request_model.model_validate(arguments)
            result = method(**request.model_dump())
        else:
            result = method(**arguments)
    except OrbitError as exc:
        return error_envelope(exc.code, exc.message, exc.details)
    except ValidationError as exc:
        # exc.json() 走 Pydantic 自己的序列化，避免 details 里残留不可 JSON 化对象。
        return error_envelope(
            "INVALID_PARAMS",
            "输入参数校验失败",
            {"errors": json.loads(exc.json(include_url=False))},
        )
    except TypeError as exc:
        # 方法签名不匹配（如多余参数已由 forbid 校验拦下，此处兜底）。
        return error_envelope("INVALID_PARAMS", f"参数与工具签名不匹配：{exc}")
    except Exception as exc:
        return error_envelope("INTERNAL_ERROR", f"未预期的内部错误（{type(exc).__name__}）")
    return ok_envelope(result)
