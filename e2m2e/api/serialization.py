"""api 层共享的异常状态翻译与 JSON 兼容序列化（#746）。

原先以私有名散落在 facade.py（``_exception_triplet``/``_serialize_value``）与
catalog_ingest.py（``finite_or_none``），被多个 api 模块跨模块导入；现集中为
公开名。只依赖 data 层模板与 numpy，不依赖 api 内其他模块。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from e2m2e.data.templates import ConvergenceState, FailureCause

__all__ = ["exception_triplet", "finite_or_none", "serialize_value"]


def exception_triplet(exc: Exception) -> tuple[ConvergenceState, FailureCause, str]:
    """读取算法异常携带的最终状态三元组。

    异常自身携带三元组时原样返回；否则按算法层普通失败处理（保留原始
    诊断信息），不让调用方吞掉 message 或误报契约错误。
    """
    status = getattr(exc, "status", None)
    cause = getattr(exc, "cause", None)
    message = getattr(exc, "message", None)
    if status is not None and cause is not None:
        return status, cause, message or str(exc)
    return (
        ConvergenceState.FAILED,
        FailureCause.UNKNOWN,
        str(exc),
    )


def serialize_value(value: Any) -> Any:
    """递归序列化 numpy 值为 JSON 兼容类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    return value


def finite_or_none(value: Any) -> float | None:
    """有限 float 或 None；None/NaN/Inf 一律折为 None（JSON 无非有限数记号）。"""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
