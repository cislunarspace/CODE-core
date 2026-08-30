"""长任务工具 worker 子进程（#588 / #576 Phase 2，子进程隔离架构）。

``python -m e2m2e.api.mcp.worker`` 一次性执行一个工具调用，供 mcp-serve
把分钟级长计算（转移搜索、族生成）隔离到可 kill 的子进程：

- stdin 读一行 JSON 请求 ``{"tool": name, "arguments": {...}}``
- stdout 按行输出 ``{"type": "progress", "fraction": ..., "message": ...}``
  （节流，见 ``_PROGRESS_MIN_INTERVAL_SEC``）与最终一行
  ``{"type": "result", "envelope": {...}}``（统一信封，见 envelope.py）
- stderr 原样透传到父进程日志；正常路径退出码 0（工具失败以信封表达）

取消 = 父进程 kill 本进程：Rust 族生成在 GIL 释放下整段运行，协作式
检查点失效，Python 工作线程不可杀——进程 kill 是唯一可靠打断手段。
数据安全：catalog 落盘是工具方法末尾的一次性原子写（tmp + os.replace），
kill 只丢当前任务，不损坏已入库记录。

不 import mcp SDK：worker 只依赖 envelope/tools/facade/config（与 sidecar
同款约束，缺 ``[mcp]`` extra 也能跑）。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

__all__ = ["run_request", "main"]

# 进度行最小发送间隔（秒）：与 server._PROGRESS_MIN_INTERVAL_SEC 同值同义
# ——高粒度回调（WSB 网格搜索可达数千次）节流成管道可消化的节奏；
# fraction>=1.0 的收尾行永不节流。
_PROGRESS_MIN_INTERVAL_SEC = 0.1


def run_request(request: dict[str, Any], emit_line: Callable[[str], None]) -> None:
    """执行一个工具请求，进度行与结果行经 ``emit_line`` 输出。

    纯进程内逻辑（可测）：``emit_line`` 收到不含换行符的完整 JSON 行。
    进度回调可能从算法层线程触发（Rust drainer），输出以锁串行化；
    结果行始终在进度行之后。任何失败都翻译成错误信封，不向 stderr
    抛 traceback、不泄漏细节（api/ 边界契约）。
    """
    from ..config import Config
    from ..facade import Facade
    from . import envelope, tools

    lock = threading.Lock()
    last_sent = 0.0

    def emit_progress(fraction: float, message: str | None = None) -> None:
        nonlocal last_sent
        now = time.monotonic()
        if fraction < 1.0 and now - last_sent < _PROGRESS_MIN_INTERVAL_SEC:
            return
        last_sent = now
        line = json.dumps(
            {"type": "progress", "fraction": fraction, "message": message}, ensure_ascii=True
        )
        with lock:
            emit_line(line)

    tool = request.get("tool")
    arguments = request.get("arguments") or {}
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        env = envelope.error_envelope("INVALID_PARAMS", "worker 请求形状错误（tool/arguments）")
    else:
        facade = Facade(config=Config())
        spec = next((s for s in tools.tool_specs(facade) if s.name == tool), None)
        if spec is None:
            env = envelope.tool_not_found(tool)
        else:
            env = envelope.invoke_tool(
                spec.method, arguments, extra_kwargs={"progress_callback": emit_progress}
            )
    line = json.dumps({"type": "result", "envelope": env}, ensure_ascii=True)
    with lock:
        emit_line(line)


def main() -> int:
    """stdin 一行请求 → stdout 进度/结果行 → 退出。"""

    def emit_line(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    try:
        request: Any = json.loads(sys.stdin.readline())
    except ValueError:
        request = None
    if not isinstance(request, dict):
        request = {}
    run_request(request, emit_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
