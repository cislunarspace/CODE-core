"""e2m2e 命令入口。

已实现子命令：mcp-serve（MCP 部署薄包装，ADR 0014 第 6 节）、
serve-stdio（GUI sidecar 入口，ADR 0035）。
完整 CLI 子命令（= Facade 方法，参数从同一份 Pydantic 模型生成）暂未提供。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e2m2e",
        description="e2m2e 命令行（地月空间转移轨道设计库）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "mcp-serve",
        help="启动 MCP 服务器（stdio 传输），把 Facade 工具暴露给 LLM Agent",
    )
    sub.add_parser(
        "serve-stdio",
        help="启动 GUI sidecar（stdio JSON 行 + 二进制帧，ADR 0035），供 Tauri 壳驱动",
    )
    return parser


def _run_mcp_serve() -> int:
    try:
        from e2m2e.api.config import Config
        from e2m2e.api.facade import Facade
        from e2m2e.api.mcp import create_server
    except ImportError as exc:
        print(
            f"MCP 协议层未安装（{exc}）。请安装 [mcp] extra："
            "uv sync --extra mcp / pip install 'e2m2e[mcp]'",
            file=sys.stderr,
        )
        return 1

    import anyio
    from mcp.server.stdio import stdio_server

    facade = Facade(config=Config())
    server = create_server(facade)

    async def _serve() -> None:
        options = server.create_initialization_options()
        async with stdio_server() as (read, write):
            await server.run(read, write, options, raise_exceptions=True)

    anyio.run(_serve)
    return 0


def _run_serve_stdio() -> int:
    from e2m2e.api.config import Config
    from e2m2e.api.facade import Facade
    from e2m2e.api.sidecar import run_loop

    run_loop(Facade(config=Config()), sys.stdin.buffer, sys.stdout.buffer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mcp-serve":
        return _run_mcp_serve()
    if args.command == "serve-stdio":
        return _run_serve_stdio()
    raise AssertionError(f"未知子命令 {args.command!r}（应由 subparser 拦截）")


if __name__ == "__main__":
    sys.exit(main())
