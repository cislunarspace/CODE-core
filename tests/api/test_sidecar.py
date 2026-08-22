"""sidecar stdio 协议层测试（issue #518 / ADR 0035）。

进程内驱动 ``handle_request`` / ``run_loop``，不起真进程：输出字节块序列
（JSON 行 + 原始帧）即协议流，端到端断言帧流后 JSON 行正确恢复。
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.interface]

from e2m2e.api.config import Config  # noqa: E402
from e2m2e.api.facade import Facade, mcp_exposed  # noqa: E402
from e2m2e.api.models import FamilyGenerationRequest, FamilyGenerationResponse  # noqa: E402
from e2m2e.api.sidecar import handle_request, run_loop  # noqa: E402
from e2m2e.api.sidecar.frames import decode_frame  # noqa: E402
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

    return _StubFacade(Config())


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
    # 周期原值透传（归一化单位，issue #525）：成员缺失或非有限时为 null
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
    for request in (
        {"tool": "no_such_tool", "arguments": {}},
        {"tool": "orbit_family_generation", "arguments": _ARGS, "binary_dtype": "f16"},
        {"tool": "orbit_family_generation", "arguments": _ARGS, "binary_dtype": 0},
    ):
        chunks = handle_request(facade, request)
        assert len(chunks) == 1
        line = json.loads(chunks[0])
        assert line["status"] == "error"
        assert line["error"]["code"] in ("UNKNOWN_TOOL", "INVALID_PARAMS")


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
    """run_loop：请求行→(进度行+响应行+帧)，坏 JSON 行得错误信封不中断。"""
    requests = (
        b'{"tool": "orbit_family_generation", "arguments": {"orbit_type": "HALO", '
        b'"libration_point": 1}, "binary_dtype": "f64", "job_id": "j"}\n'
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
