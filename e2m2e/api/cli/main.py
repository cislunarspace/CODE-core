"""e2m2e 命令入口。

工具子命令与 MCP 工具面对称（ADR 0014 决策 5，#602）：每个
`mcp_exposed` 且已实现的 Facade 方法都有一个同名子命令（下划线转连
字符），参数从同一份 Pydantic 请求模型生成，输出是同一个统一信封。
部署入口：mcp-serve（MCP 部署薄包装）、serve-stdio（GUI sidecar 入口，
ADR 0035）。

约定：信封 JSON 写 stdout；进度与 argparse 用法错误写 stderr；信封
`status=ok` 退出码 0，`error` 非 0。长任务工具改跑 worker 子进程
（路由清单在执行核心 LONG_RUNNING_TOOLS，#601），Ctrl-C kill 子进程。
"""

from __future__ import annotations

import argparse
import enum
import json
import subprocess
import sys
import typing
from collections.abc import Sequence
from typing import Any

from e2m2e.api import execution
from e2m2e.api.config import Config
from e2m2e.api.execution import LONG_RUNNING_TOOLS, execute_tool, worker_request_payload
from e2m2e.api.facade import Facade, ToolInfo, tool_inventory
from e2m2e.api.mcp import envelope

__all__ = ["main", "build_parser", "tool_subcommands"]


def tool_subcommands(facade: Facade) -> dict[str, ToolInfo]:
    """CLI 子命令清单：连字符命名 → 工具元数据（placeholder 不注册）。"""
    return {
        info.name.replace("_", "-"): info
        for info in tool_inventory(facade)
        if info.status == "implemented"
    }


def _unwrap_optional(annotation: Any) -> Any:
    """ "X | None" → X；其余原样。"""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _add_field_option(parser: argparse.ArgumentParser, name: str, field: Any) -> None:
    """把请求模型字段映射为命令行选项。

    标量按类型；bool 用 --x/--no-x；Enum 用 choices；容器/任意类型
    （epoch、target_ephemeris 等）无法自然扁平化，接受 JSON 字符串
    （--help 中注明），Pydantic 校验仍由模型负责。必填性与默认值、
    类型名写进帮助文本（--help 契约，#602 故事 2）。
    """
    flag = "--" + name.replace("_", "-")
    description = (field.description or "").strip()
    ann = _unwrap_optional(field.annotation)
    type_name = getattr(ann, "__name__", "") or ""
    if field.is_required():
        hint = f"（{type_name}，必填）" if type_name else "（必填）"
    elif field.default is not None:
        default_repr = repr(field.default)
        hint = f"（{type_name}，默认 {default_repr}）" if type_name else f"（默认 {default_repr}）"
    else:
        hint = f"（{type_name}）" if type_name else ""
    if ann is bool:
        parser.add_argument(
            flag,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"{description}{hint}",
        )
    elif ann in (str, int, float):
        parser.add_argument(
            flag,
            type=ann,
            default=None,
            help=f"{description}{hint}",
        )
    elif isinstance(ann, type) and issubclass(ann, enum.Enum):
        parser.add_argument(
            flag,
            choices=[member.value for member in ann],
            default=None,
            help=f"{description}{hint}",
        )
    else:
        parser.add_argument(
            flag,
            type=json.loads,
            default=None,
            metavar="JSON",
            help=f"{description}{hint}（JSON 值）",
        )


def _build_tool_subparser(
    sub: argparse._SubParsersAction, facade: Facade, command: str, info: ToolInfo
) -> None:
    method = getattr(facade, info.name)
    help_text = (method.__doc__ or info.name).strip().splitlines()[0]
    tool_parser = sub.add_parser(command, help=help_text, description=help_text)
    if info.request_model is not None:
        for field_name, field in info.request_model.model_fields.items():
            _add_field_option(tool_parser, field_name, field)


def build_parser(facade: Facade | None = None) -> argparse.ArgumentParser:
    """构建命令行解析器。

    Args:
        facade: 工具子命令从它的 `tool_inventory` 派生；None 时只注册
            部署子命令（向后兼容既有调用）。
    """
    parser = argparse.ArgumentParser(
        prog="e2m2e",
        description="e2m2e 命令行（地月空间转移轨道设计库）。工具子命令与 MCP 工具面同源对称",
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

    if facade is not None:
        for command, info in tool_subcommands(facade).items():
            _build_tool_subparser(sub, facade, command, info)
    return parser


def _emit_progress_to_stderr(fraction: float, message: str | None = None) -> None:
    """短任务进程内执行的进度回调：写 stderr，不影响 stdout 的信封。"""
    percent = round(fraction * 100)
    print(f"progress: {percent}%{(' ' + message) if message else ''}", file=sys.stderr)


def _run_via_worker(
    facade: Facade,
    tool_name: str,
    arguments: dict,
    progress_callback=None,
) -> envelope.Envelope:
    """长任务工具经 worker 子进程执行（#601 同款协议，同步泵）。

    进度行转发 stderr；KeyboardInterrupt（Ctrl-C）kill 子进程后重抛——
    进程 kill 是打断 GIL 释放下 Rust 长计算的唯一可靠手段。
    """
    proc = subprocess.Popen(
        execution.WORKER_ARGV, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True
    )
    assert proc.stdin is not None and proc.stdout is not None  # PIPE 已请求
    request = worker_request_payload(tool_name, arguments, facade.config)
    proc.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
    proc.stdin.close()
    env: envelope.Envelope | None = None
    try:
        for raw in proc.stdout:
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if message.get("type") == "progress":
                if progress_callback is not None:
                    progress_callback(message.get("fraction", 0.0), message.get("message"))
            elif message.get("type") == "result":
                env = message.get("envelope")
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    if env is None:
        env = envelope.error_envelope(
            "WORKER_CRASHED", f"worker 进程未返回结果（exit={proc.returncode}）"
        )
    return env


def _run_tool_command(facade: Facade, command: str, args: argparse.Namespace) -> int:
    """执行一个工具子命令：信封写 stdout，退出码按信封状态。"""
    info = tool_subcommands(facade)[command]
    arguments = {
        key: value for key, value in vars(args).items() if key != "command" and value is not None
    }
    if info.name in LONG_RUNNING_TOOLS:
        env = _run_via_worker(facade, info.name, arguments, _emit_progress_to_stderr)
    else:
        env, _frames = execute_tool(
            facade, info.name, arguments, progress_callback=_emit_progress_to_stderr
        )
    print(json.dumps(env, ensure_ascii=False))
    return 0 if env["status"] == "ok" else 1


def _run_mcp_serve(facade: Facade) -> int:
    try:
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

    server = create_server(facade)

    async def _serve() -> None:
        options = server.create_initialization_options()
        async with stdio_server() as (read, write):
            await server.run(read, write, options, raise_exceptions=True)

    anyio.run(_serve)
    return 0


def _run_serve_stdio(facade: Facade) -> int:
    from e2m2e.api.sidecar import run_loop

    run_loop(facade, sys.stdin.buffer, sys.stdout.buffer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # 工具子命令从 Facade 派生；Facade() 构造廉价（存配置 + 目录扫描），
    # --help 与部署子命令也顺带获得完整工具清单。
    facade = Facade(config=Config())
    args = build_parser(facade).parse_args(argv)
    if args.command == "mcp-serve":
        return _run_mcp_serve(facade)
    if args.command == "serve-stdio":
        return _run_serve_stdio(facade)
    try:
        return _run_tool_command(facade, args.command, args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
