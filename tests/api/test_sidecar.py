"""sidecar stdio 协议层测试。

进程内驱动 ``handle_request`` / ``run_loop``，不起真进程：输出字节块序列
（JSON 行 + 原始帧）即协议流，端到端断言帧流后 JSON 行正确恢复。
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.interface]

from kernel_helpers import requires_native_symbols  # noqa: E402

from e2m2e.api import execution  # noqa: E402
from e2m2e.api.config import Config  # noqa: E402
from e2m2e.api.facade import Facade, mcp_exposed  # noqa: E402
from e2m2e.api.frames import decode_frame  # noqa: E402
from e2m2e.api.mcp import envelope  # noqa: E402
from e2m2e.api.models import (  # noqa: E402
    CatalogGetRequest,
    CatalogQueryRequest,
    CatalogQueryResponse,
    CatalogRecordResponse,
    FamilyGenerationRequest,
    FamilyGenerationResponse,
)
from e2m2e.api.sidecar import handle_request, run_loop  # noqa: E402
from e2m2e.data.templates import ConvergenceState, FailureCause  # noqa: E402
from e2m2e.data.types.orbit import Orbit  # noqa: E402

_ARGS = {"orbit_type": "HALO", "libration_point": 1}  # 能过 FamilyGenerationRequest 校验的最小参数


def _orbit(seed: float, n: int = 3, period: float | None = None) -> Orbit:
    states = np.arange(n * 6, dtype=float).reshape(n, 6) + seed
    orbit = Orbit(states=states, times=np.linspace(0.0, 1.0, n))
    orbit.period = period
    return orbit


class _System:
    """鸭子类型 system：仅供 sidecar 映射读取 mu。"""

    mu = 1.215e-2


@pytest.fixture
def facade(tmp_path, monkeypatch) -> Facade:
    monkeypatch.setenv("E2M2E_CATALOG_DIR", str(tmp_path / "catalog"))

    class _StubFacade(Facade):
        @mcp_exposed(request_model=FamilyGenerationRequest)
        def orbit_family_generation(self, **params) -> FamilyGenerationResponse:
            return FamilyGenerationResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="ok",
                orbits=[_orbit(0.0, period=2.5), _orbit(10.0)],
                family_type="halo",
                system=_System(),
                requested_members=2,
                generated_members=2,
            )

        @mcp_exposed(request_model=CatalogQueryRequest)
        def catalog_query(self, **params) -> CatalogQueryResponse:
            return CatalogQueryResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="查询完成：0 条记录",
                records=[],
            )

        @mcp_exposed(request_model=CatalogGetRequest)
        def catalog_get(self, **params) -> CatalogRecordResponse:
            if getattr(self, "_broken_response", False):
                rec = _record()
                object.__setattr__(rec, "scalars", object())  # 不可 JSON 化的标量段
                return rec
            return _record()

    return _StubFacade(Config())


def _record() -> CatalogRecordResponse:
    """含 ndarray 数组段的族记录（catalog_get 响应形状）。"""
    return CatalogRecordResponse(
        record_id="r1",
        created_at="2026-01-01T00:00:00Z",
        source_tool="orbit_family_generation",
        source_record_id=None,
        orbit_family="halo_n",
        libration_point=1,
        jacobi=None,
        amplitude=None,
        has_cr3bp=True,
        has_ephemeris=False,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        member_count=2,
        tags=[],
        note="",
        scalars={"mu": 0.1},
        request={"orbit_type": "HALO"},
        members=[{"index": 0}, {"index": 1}],
        arrays={
            "cr3bp/members/0/states": np.arange(18.0).reshape(3, 6),
            "cr3bp/members/1/states": np.arange(18.0).reshape(3, 6) + 10.0,
        },
    )


def _lines_and_frames(chunks: list[bytes]):
    """把输出字节块切回 JSON 行与帧（按 binary_frames 声明消费）。"""
    stream = b"".join(chunks)
    pos = 0
    lines = []
    while pos < len(stream):
        nl = stream.index(b"\n", pos)
        line = json.loads(stream[pos:nl])
        frames = []
        pos = nl + 1
        for _ in range(line.get("binary_frames", 0)):
            _, _, consumed = decode_frame(memoryview(stream)[pos:])
            frames.append(stream[pos : pos + consumed])
            pos += consumed
        lines.append((line, frames))
    return lines


def test_family_generation_binary_roundtrip(facade):
    """画布契约：族状态数组走二进制帧，JSON 行占位 None + binary_frames 计数。"""
    chunks = handle_request(
        facade,
        {"tool": "orbit_family_generation", "arguments": _ARGS, "binary_dtype": "f32"},
    )
    (progress, _), (line, frames) = _lines_and_frames(chunks)
    assert progress["status"] == "progress"
    assert line["status"] == "ok"
    assert line["binary_frames"] == 2
    orbits = line["data"]["orbits"]
    assert [o["states"] for o in orbits] == [None, None]
    assert len(orbits[0]["times"]) == 3  # 小数组留 JSON
    # 周期原值透传（归一化单位）：成员缺失或非有限时为 null
    assert [o["period"] for o in orbits] == [2.5, None]
    # 重采样方程需要 mu：响应级原值透传
    assert line["data"]["mu"] == pytest.approx(1.215e-2)
    # 帧内容 = 成员状态数组，dtype 按请求方声明
    arr0, dtype0, _ = decode_frame(frames[0])
    assert dtype0 == "f32"
    np.testing.assert_allclose(arr0, np.arange(18, dtype=float).reshape(3, 6).astype(np.float32))


def test_family_generation_requires_binary_dtype(facade):
    """响应含 Orbit 对象不可 JSON 化，未声明 dtype 时给结构化错误而非崩溃。"""
    chunks = handle_request(facade, {"tool": "orbit_family_generation", "arguments": _ARGS})
    [line] = [json.loads(c) for c in chunks]
    assert line["status"] == "error"
    assert line["error"]["code"] == "INVALID_PARAMS"


def test_error_envelope_for_unknown_tool_and_bad_dtype(facade):
    # 未知工具与 TOOL_NOT_FOUND 统一编码（#601：两个传输层单一来源）
    for request, expected in (
        ({"tool": "no_such_tool", "arguments": {}}, "TOOL_NOT_FOUND"),
        (
            {"tool": "orbit_family_generation", "arguments": _ARGS, "binary_dtype": "f16"},
            "INVALID_PARAMS",
        ),
        (
            {"tool": "orbit_family_generation", "arguments": _ARGS, "binary_dtype": 0},
            "INVALID_PARAMS",
        ),
    ):
        chunks = handle_request(facade, request)
        assert len(chunks) == 1
        line = json.loads(chunks[0])
        assert line["status"] == "error"
        assert line["error"]["code"] == expected


def test_progress_line_then_response(facade):
    """进度行先于响应行：status=progress、meta 携带 job_id/percent/message。"""
    chunks = handle_request(
        facade,
        {
            "tool": "orbit_family_generation",
            "arguments": _ARGS,
            "binary_dtype": "f32",
            "job_id": "job-1",
        },
    )
    progress = json.loads(chunks[0])
    assert progress["status"] == "progress"
    assert progress["data"] is None and progress["error"] is None
    assert progress["meta"]["job_id"] == "job-1"
    assert progress["meta"]["percent"] == 0.0
    assert isinstance(progress["meta"]["message"], str)


def test_run_loop_end_to_end(facade):
    """run_loop：短工具请求行→(进度行+响应行+帧)，坏 JSON 行得错误信封不中断。

    长任务（族生成等）在 #607 起走 worker 子进程线程，由取消端到端
    测试覆盖；本测试用短工具保持字节顺序确定性。
    """
    requests = (
        b'{"tool": "catalog_get", "arguments": {"record_id": "r1"}, '
        b'"binary_dtype": "f64"}\n'
        b"this is not json\n"
    )
    stdout = io.BytesIO()
    run_loop(facade, io.BytesIO(requests), stdout)
    lines = _lines_and_frames([stdout.getvalue()])
    assert lines[0][0]["status"] == "progress"
    assert lines[1][0]["status"] == "ok" and lines[1][0]["binary_frames"] == 2
    assert len(lines[1][1]) == 2
    assert lines[2][0]["status"] == "error" and lines[2][0]["error"]["code"] == "INVALID_PARAMS"


def test_run_loop_empty_input(facade):
    stdout = io.BytesIO()
    run_loop(facade, io.BytesIO(b""), stdout)
    assert stdout.getvalue() == b""


def test_cli_serve_stdio_registered(capsys):
    """`e2m2e serve-stdio --help` 可用（ADR 0035 §结果）。"""
    import pytest as _pytest

    from e2m2e.api.cli.main import build_parser, main

    args = build_parser().parse_args(["serve-stdio"])
    assert args.command == "serve-stdio"
    with _pytest.raises(SystemExit) as exc_info:
        main(["serve-stdio", "--help"])
    assert exc_info.value.code == 0
    assert "serve-stdio" in capsys.readouterr().out


def test_catalog_get_binary_roundtrip(facade):
    """记录数组段的 ndarray 出帧，JSON 行 null 占位，帧序=占位序。"""
    chunks = handle_request(
        facade,
        {"tool": "catalog_get", "arguments": {"record_id": "r1"}, "binary_dtype": "f32"},
    )
    (progress, _), (line, frames) = _lines_and_frames(chunks)
    assert progress["status"] == "progress"
    assert line["status"] == "ok"
    assert line["binary_frames"] == 2
    arrays = line["data"]["arrays"]
    assert arrays == {"cr3bp/members/0/states": None, "cr3bp/members/1/states": None}
    assert line["data"]["scalars"] == {"mu": 0.1}  # 非数组段留 JSON
    arr1, dtype, _ = decode_frame(frames[1])
    assert dtype == "f32"
    np.testing.assert_allclose(arr1, (np.arange(18.0).reshape(3, 6) + 10.0).astype(np.float32))


def test_catalog_get_requires_binary_dtype(facade):
    """未声明 binary_dtype 时结构化错误而非 PydanticSerializationError 崩溃。"""
    chunks = handle_request(facade, {"tool": "catalog_get", "arguments": {"record_id": "r1"}})
    [line] = [json.loads(c) for c in chunks]
    assert line["status"] == "error"
    assert line["error"]["code"] == "INVALID_PARAMS"


def test_run_loop_survives_envelope_serialization_failure(facade):
    """信封化异常不得炸进程，返回 INTERNAL_ERROR 信封并继续循环。"""
    facade._broken_response = True
    try:
        requests = (
            b'{"tool": "catalog_get", "arguments": {"record_id": "r1"}, '
            b'"binary_dtype": "f32"}\n'
            b'{"tool": "catalog_query", "arguments": {}}\n'
        )
        stdout = io.BytesIO()
        run_loop(facade, io.BytesIO(requests), stdout)  # 不应抛异常
        lines = _lines_and_frames([stdout.getvalue()])
        # 坏请求兑成 INTERNAL_ERROR 信封（进度行随失败丢弃），循环存活
        assert lines[0][0]["status"] == "error"
        assert lines[0][0]["error"]["code"] == "INTERNAL_ERROR"
        # 第二个请求正常处理：进度行 + ok 响应
        assert lines[1][0]["status"] == "progress"
        assert lines[2][0]["status"] == "ok"
    finally:
        facade._broken_response = False


def test_invoke_tool_serializes_ndarray_inline():
    """结果含 ndarray 时信封层降级内联为嵌套 list 而非 INTERNAL_ERROR
    （sidecar 大数组仍首选二进制帧，见帧映射）。
    """
    from e2m2e.api.models import CatalogRecordResponse
    from e2m2e.data.templates import ConvergenceState, FailureCause

    record = CatalogRecordResponse(
        record_id="r",
        created_at="2026-01-01T00:00:00Z",
        source_tool="t",
        source_record_id=None,
        orbit_family=None,
        libration_point=1,
        jacobi=None,
        amplitude=None,
        has_cr3bp=True,
        has_ephemeris=False,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="",
        member_count=1,
        tags=[],
        note="",
        scalars={},
        request={},
        members=[],
        arrays={"cr3bp/states": np.zeros((3, 6))},
    )

    class _Method:
        request_model = None

        def __call__(self, **kw):
            return record

    env = envelope.invoke_tool(_Method(), {})
    assert env["status"] == "ok"
    assert env["data"]["arrays"]["cr3bp/states"] == [[0.0] * 6] * 3
    assert "Traceback" not in json.dumps(env)


def test_catalog_query_unchanged_by_binary_mapping(facade):
    """不进帧映射的工具走原信封化路径，无帧、无 binary_frames。"""
    chunks = handle_request(facade, {"tool": "catalog_query", "arguments": {}})
    lines = [json.loads(c) for c in chunks]
    assert len(lines) == 2  # 进度行 + 响应行，无帧
    assert lines[0]["status"] == "progress"
    assert lines[1]["status"] == "ok"
    assert "binary_frames" not in lines[1]
    assert lines[1]["data"]["message"] == "查询完成：0 条记录"


@requires_native_symbols("propagate_geocentric_fate_map_py")
def test_spatiography_dynamical_map_binary_roundtrip(facade):
    """画布契约：五个场走二进制帧（E2M2 帧），JSON 行留占位与两轴。"""
    chunks = handle_request(
        facade,
        {
            "tool": "spatiography_dynamical_map",
            "arguments": {"zone": "SC", "n_a": 3, "n_e": 2, "span_years": 0.3},
            "binary_dtype": "f64",
        },
    )
    (progress, _), (line, frames) = _lines_and_frames(chunks)
    assert progress["status"] == "progress"
    assert line["status"] == "ok"
    assert line["binary_frames"] == 5
    data = line["data"]
    assert data["ybar_field"] is None and data["fate_ids"] is None
    assert len(data["a_over_a_moon"]) == 3 and len(data["e_grid"]) == 2
    # 帧 1 = Ȳ 场 (n_a, n_e)；终端短路格 NaN 保留。
    ybar, dtype, _ = decode_frame(frames[0])
    assert dtype == "f64" and ybar.shape == (3, 2)
    assert np.any(np.isfinite(ybar))
    # 帧 2 = 命运类 id 场（f64 编码的整数 id）。
    fate, _, _ = decode_frame(frames[1])
    assert fate.shape == (3, 2)
    assert np.allclose(fate, np.round(fate), atol=1e-9)


def test_spatiography_dynamical_map_requires_binary_dtype(facade):
    chunks = handle_request(
        facade,
        {"tool": "spatiography_dynamical_map", "arguments": {"zone": "SC", "n_a": 2, "n_e": 2}},
    )
    [line] = [json.loads(c) for c in chunks]
    assert line["status"] == "error"
    assert line["error"]["code"] == "INVALID_PARAMS"


# ---------------------------------------------------------------------------
# 长任务 worker 路由与线级取消（#607）
# ---------------------------------------------------------------------------

_SIDECAKE_FAKE_WORKER = """
import json, os, struct, sys, time
pidfile = os.environ.get("SIDECAKE_PIDFILE")
if pidfile:
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
mode = os.environ.get("SIDECAKE_MODE", "progress-ok")
out = sys.stdout.buffer
def emit(obj):
    out.write((json.dumps(obj) + "\\n").encode()); out.flush()
emit({"type": "progress", "fraction": 0.5, "message": "half"})
if mode == "sleep":
    time.sleep(60)
if mode == "crash":
    sys.exit(3)
if mode == "frame":
    frame = struct.pack("<IBB", 0x324D3245, 0, 1) + struct.pack("<I", 2)
    frame += struct.pack("<2f", 1.0, 2.0)
    emit({"type": "result", "envelope": {"status": "ok", "data": {}, "error": None,
          "meta": {}}, "binary_frames": 1})
    out.write(frame); out.flush()
    sys.exit(0)
emit({"type": "result", "envelope": {"status": "ok", "data": {"mode": mode},
      "error": None, "meta": {}}})
"""


@pytest.fixture
def sidecar_worker(monkeypatch, tmp_path):
    """sidecar 的 worker 命令换成 fake 脚本，可切模式（#607）。"""
    pidfile = tmp_path / "sidecar-worker.pid"

    def use(mode: str) -> None:
        monkeypatch.setenv("SIDECAKE_MODE", mode)
        monkeypatch.setenv("SIDECAKE_PIDFILE", str(pidfile))
        monkeypatch.setattr(execution, "WORKER_ARGV", [sys.executable, "-c", _SIDECAKE_FAKE_WORKER])

    use("progress-ok")
    return SimpleNamespace(pidfile=pidfile, use=use)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    import errno
    import os

    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def test_long_tool_streams_progress_and_result(sidecar_worker, facade):
    """长任务经 worker：真实进度行（fraction）与结果行依次流出。"""
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    worker = threading.Thread(
        target=run_loop,
        args=(facade, os.fdopen(stdin_r, "rb"), os.fdopen(stdout_w, "wb")),
        daemon=True,
    )
    worker.start()
    os.write(
        stdin_w,
        b'{"tool": "orbit_family_generation", "arguments": {"orbit_type": "HALO", '
        b'"libration_point": 1}, "binary_dtype": "f32", "job_id": "j1"}\n',
    )
    import time as _time

    deadline = _time.monotonic() + 15
    collected = bytearray()
    collector = threading.Thread(
        target=lambda: [collected.extend(os.read(stdout_r, 4096)) for _ in iter(int, 1)],
        daemon=True,
    )
    collector.start()
    while _time.monotonic() < deadline and b'"mode"' not in bytes(collected):
        _time.sleep(0.05)
    os.close(stdin_w)
    worker.join(timeout=10)
    text = collected.decode("utf-8", errors="replace")
    assert '"status": "progress"' in text and "half" in text, "worker 进度应带真实 fraction 转发"
    assert '"status": "ok"' in text


def test_cancel_line_kills_worker_and_emits_cancelled(sidecar_worker, facade):
    """取消行 → 子进程被 kill + cancelled 状态行；循环存活可继续服务。"""
    sidecar_worker.use("sleep")
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    worker = threading.Thread(
        target=run_loop,
        args=(facade, os.fdopen(stdin_r, "rb"), os.fdopen(stdout_w, "wb")),
        daemon=True,
    )
    worker.start()
    os.write(
        stdin_w,
        b'{"tool": "orbit_family_generation", "arguments": {"orbit_type": "HALO", '
        b'"libration_point": 1}, "binary_dtype": "f32", "job_id": "j1"}\n',
    )
    import time as _time

    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and not sidecar_worker.pidfile.exists():
        _time.sleep(0.05)
    pid = int(sidecar_worker.pidfile.read_text(encoding="ascii").strip())
    os.write(stdin_w, b'{"cancel": "j1"}\n')
    collected = bytearray()
    collector = threading.Thread(
        target=lambda: [collected.extend(os.read(stdout_r, 4096)) for _ in iter(int, 1)],
        daemon=True,
    )
    collector.start()
    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and b'"cancelled"' not in bytes(collected):
        _time.sleep(0.05)
    assert b'"cancelled"' in bytes(collected), "取消后必须回 cancelled 状态行"
    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and _pid_alive(pid):
        _time.sleep(0.1)
    assert not _pid_alive(pid), "取消后 worker 子进程应已被终止"
    # 循环存活：后续请求照常处理
    os.write(stdin_w, b'{"tool": "catalog_query", "arguments": {}}\n')
    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and b'"records"' not in bytes(collected):
        _time.sleep(0.05)
    os.close(stdin_w)
    assert b'"cancelled"' in bytes(collected)


def test_cancel_unknown_job_is_idempotent(sidecar_worker, facade):
    """对未知 job 的取消：幂等回 cancelled 行，不崩溃。"""
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    worker = threading.Thread(
        target=run_loop,
        args=(facade, os.fdopen(stdin_r, "rb"), os.fdopen(stdout_w, "wb")),
        daemon=True,
    )
    worker.start()
    os.write(stdin_w, b'{"cancel": "no-such-job"}\n')
    import time as _time

    collected = bytearray()
    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and b'"cancelled"' not in bytes(collected):
        chunk = os.read(stdout_r, 4096)
        if chunk:
            collected.extend(chunk)
        else:
            _time.sleep(0.05)
    os.close(stdin_w)
    line = json.loads(bytes(collected).splitlines()[0])
    assert line["status"] == "cancelled"
    assert line["meta"]["job_id"] == "no-such-job"


def test_worker_frames_pass_through(sidecar_worker, facade):
    """worker 结果行声明的二进制帧原样透传（帧序 = 声明序）。"""
    sidecar_worker.use("frame")
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    worker = threading.Thread(
        target=run_loop,
        args=(facade, os.fdopen(stdin_r, "rb"), os.fdopen(stdout_w, "wb")),
        daemon=True,
    )
    worker.start()
    os.write(
        stdin_w,
        b'{"tool": "orbit_family_generation", "arguments": {"orbit_type": "HALO", '
        b'"libration_point": 1}, "binary_dtype": "f32", "job_id": "j2"}\n',
    )
    import time as _time

    collected = bytearray()
    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and len(bytes(collected)) < 120:
        chunk = os.read(stdout_r, 4096)
        if chunk:
            collected.extend(chunk)
        else:
            _time.sleep(0.05)
    os.close(stdin_w)
    worker.join(timeout=10)
    lines = _lines_and_frames([bytes(collected)])
    result_line, frames = next(line for line in lines if line[0].get("binary_frames"))
    assert result_line["status"] == "ok" and result_line["binary_frames"] == 1
    arr, dtype, _ = decode_frame(frames[0])
    assert dtype == "f32"
    np.testing.assert_allclose(arr, np.array([1.0, 2.0], dtype=np.float32))
