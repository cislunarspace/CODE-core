r"""CLI 工具子命令测试（#602）：与 MCP 工具面对称。

从 main(argv) 入口进，断言 stdout 的信封与退出码，不断言 argparse 内部
结构。派生一致性对照 \`tool_inventory\`（先例：test_registered_tools_
match_inventory）。长任务走 worker 子进程（复用 #588 fake worker 模式）。
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.interface

from e2m2e.api import execution  # noqa: E402
from e2m2e.api.cli import main as cli_main  # noqa: E402
from e2m2e.api.cli.main import main, tool_subcommands  # noqa: E402
from e2m2e.api.config import Config  # noqa: E402
from e2m2e.api.facade import Facade, tool_inventory  # noqa: E402


@pytest.fixture
def facade() -> Facade:
    # tests/api/conftest.py 已把 E2M2E_CATALOG_DIR 重定向到独立临时目录
    return Facade(Config())


# ---------------------------------------------------------------------------
# 派生一致性（单一来源，不漂移）
# ---------------------------------------------------------------------------


def test_tool_subcommands_match_inventory(facade):
    """子命令集合 = tool_inventory 中 implemented 的连字符命名；placeholder 不出现。"""
    expected = {
        i.name.replace("_", "-") for i in tool_inventory(facade) if i.status == "implemented"
    }
    assert set(tool_subcommands(facade)) == expected
    placeholders = {i.name.replace("_", "-") for i in tool_inventory(facade)} - expected
    assert placeholders, "前提：Facade 至少有一个 placeholder 工具"
    assert not placeholders & set(tool_subcommands(facade))


def test_top_help_lists_tools_and_deployments(capsys):
    """`e2m2e --help` 同时列出工具子命令与两个部署入口。"""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "catalog-query" in out
    assert "design-orbit" in out
    assert "mcp-serve" in out
    assert "serve-stdio" in out


def test_tool_help_shows_options(capsys):
    """`e2m2e design-orbit --help` 展示模型字段的选项、类型与默认值。"""
    with pytest.raises(SystemExit) as exc_info:
        main(["design-orbit", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--orbit-type" in out
    assert "必填" in out  # orbit_type 无默认：标注必填
    assert "--output-step" in out
    assert "3600.0" in out  # 模型默认值进帮助文本
    # 复杂类型（epoch 是 Any）按 JSON 值选项暴露
    assert "--epoch" in out
    assert "JSON" in out


# ---------------------------------------------------------------------------
# 端到端：便宜工具真调用（ADR 0037 决策 2 的最小真实调用）
# ---------------------------------------------------------------------------


def test_catalog_query_end_to_end(capsys):
    """`e2m2e catalog-query` 对空库：退出码 0，stdout 是 ok 信封。"""
    rc = main(["catalog-query"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "ok"
    assert env["data"]["records"] == []


def test_catalog_get_error_exit_code(capsys):
    """查询不存在的记录：退出码 1，stdout 是结构化错误信封。"""
    rc = main(["catalog-get", "--record-id", "no-such-record"])
    assert rc == 1
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "error"
    assert env["error"]["code"] == "RECORD_NOT_FOUND"


def test_invalid_params_exit_code(capsys):
    """非法参数值：执行层校验 → INVALID_PARAMS 信封，退出码 1。"""
    rc = main(["catalog-query", "--libration-point", "99"])
    assert rc == 1
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "INVALID_PARAMS"


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as exc_info:
        main(["no-such-command"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 长任务：worker 子进程路由（fake worker 模式，同 test_mcp_worker 先例）
# ---------------------------------------------------------------------------

_FAKE_WORKER_SCRIPT = """
import json, os, sys, time
pidfile = os.environ.get("FAKE_WORKER_PIDFILE")
if pidfile:
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
mode = os.environ.get("FAKE_WORKER_MODE", "ok")
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
emit({"type": "progress", "fraction": 0.5, "message": "half"})
if mode == "crash":
    sys.exit(3)
if mode == "sleep":
    time.sleep(60)
emit({"type": "result", "envelope": {"status": "ok", "data": {"mode": mode},
        "error": None, "meta": {}}})
"""


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    """把 CLI 的 worker 命令换成 fake 脚本，可切模式（先例：test_mcp_worker）。"""
    pidfile = tmp_path / "fake-worker.pid"

    def use(mode: str) -> None:
        monkeypatch.setenv("FAKE_WORKER_MODE", mode)
        monkeypatch.setenv("FAKE_WORKER_PIDFILE", str(pidfile))
        monkeypatch.setattr(execution, "WORKER_ARGV", [sys.executable, "-c", _FAKE_WORKER_SCRIPT])

    use("ok")
    return SimpleNamespace(pidfile=pidfile, use=use)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    import errno
    import os

    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def test_long_tool_routes_via_worker(fake_worker, facade, capsys):
    """长任务工具改跑 worker 子进程：进度进 stderr，信封进 stdout，退出码 0。"""
    fake_worker.use("ok")
    rc = main(
        [
            "transfer-design",
            "--transfer-type",
            "HMN",
            "--tli-epoch",
            "2460800.5",
            "--target-orbit-radius-km",
            "384405.0",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    env = json.loads(captured.out.strip().splitlines()[-1])
    assert env["status"] == "ok"
    assert env["data"]["mode"] == "ok"
    assert "half" in captured.err, "worker 进度行应转发到 stderr"


def test_worker_crash_yields_error_exit(fake_worker, capsys):
    fake_worker.use("crash")
    rc = main(["transfer-design", "--transfer-type", "HMN", "--tli-epoch", "2460800.5"])
    assert rc == 1
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "WORKER_CRASHED"


def test_interrupt_kills_worker(fake_worker, facade, tmp_path):
    """Ctrl-C（KeyboardInterrupt）传播：worker 子进程被 kill，不残留。"""
    fake_worker.use("sleep")
    with open(tmp_path / "raise.pid", "w") as f:
        f.write("")  # 占位：pidfile 由 fake worker 写

    def raising_progress(fraction, message=None):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli_main._run_via_worker(facade, "transfer_design", {}, progress_callback=raising_progress)
    pid = int(fake_worker.pidfile.read_text(encoding="ascii").strip())
    import time

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    assert not _pid_alive(pid), "KeyboardInterrupt 后 worker 子进程应已被终止"
