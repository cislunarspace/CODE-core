"""结构化日志：配置工厂。

标准 logging + 关键事件键值对、零新依赖（ADR 0011）：算法层保持
``logger.info``，打靶/延拓迭代等关键数值事件用键值对
（``logger.info("correction_iter", iter=3, error=1e-8)``）；本模块提供
配置工厂（Formatter 把键值对转 key=val）。api/config.py 控制级别和
handler。
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging"]


class KeyValueFormatter(logging.Formatter):
    """把日志的 extra 键值对附加到消息尾部（``key=val``）。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
                "message",
            }
        }
        if extras:
            return base + " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


def configure_logging(
    level: str = "WARNING",
    *,
    handler: logging.Handler | None = None,
    formatter: logging.Formatter | None = None,
) -> None:
    """配置根 logger（级别 + 键值对 Formatter）。

    Args:
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）。
        handler: 输出 handler（缺省 StreamHandler）。
        formatter: 自定义 Formatter（缺省 KeyValueFormatter）。
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.WARNING))
    if handler is None:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter or KeyValueFormatter("%(levelname)s %(name)s: %(message)s"))
    # 避免重复添加 handler（多次调用 configure_logging 幂等）
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
