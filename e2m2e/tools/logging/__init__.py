"""结构化日志：配置工厂。

标准 logging + 关键事件键值对、零新依赖（ADR 0011）：算法层保持 ``logger.info``，
打靶/延拓迭代等关键数值事件用键值对（``logger.info("correction_iter", iter=3,
error=1e-8)``）；本模块提供配置工厂（Formatter 把键值对转 key=val 或 JSON）。
api/config.py 控制级别和 handler。

实现状态：骨架。
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging"]


def configure_logging(level: str = "WARNING") -> None:
    """配置根 logger（级别 + 键值对 Formatter）。

    实现状态：待实现。

    Args:
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）。
    """
    logging.basicConfig(level=getattr(logging, level, logging.WARNING))
    raise NotImplementedError("tools/logging 配置工厂待实现")
