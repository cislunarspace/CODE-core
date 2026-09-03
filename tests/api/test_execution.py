"""执行核心测试（#601）：单一入口的信封/帧/配置语义。

只走接缝外部行为：给核心一个 facade 和一份参数，断言信封与帧内容。
本文件不依赖 ``[mcp]`` extra——执行核心是 sidecar 与 worker 的共同依赖。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from e2m2e.api import execution
from e2m2e.api.config import Config
from e2m2e.api.facade import Facade, mcp_exposed
from e2m2e.api.frames import decode_frame
from e2m2e.api.models import (
    CatalogGetRequest,
    CatalogQueryRequest,
    CatalogQueryResponse,
    CatalogRecordResponse,
    FamilyGenerationRequest,
    FamilyGenerationResponse,
)
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.interface

_ARGS = {"orbit_type": "HALO", "libration_point": 1}  # 能过 FamilyGenerationRequest 校验的最小参数


class _System:
    """鸭子类型 system：仅供画布契约读取 mu。"""

    mu = 1.215e-2


def _orbit(seed: float, n: int = 3, period: float | None = None) -> Orbit:
    states = np.arange(n * 6, dtype=float).reshape(n, 6) + seed
    orbit = Orbit(states=states, times=np.linspace(0.0, 1.0, n))
    orbit.period = period
    return orbit


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
                family_id="run-1",
                member_index=0,
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

    return _StubFacade(Config())


def test_unknown_tool_yields_tool_not_found(facade):
    """未知工具（含 placeholder）统一 TOOL_NOT_FOUND（#601 单一编码）。"""
    env, frames = execution.execute_tool(facade, "no_such_tool", {})
    assert env["status"] == "error"
    assert env["error"]["code"] == "TOOL_NOT_FOUND"
    assert frames == []


def test_preflight_unknown_tool_and_bad_dtype(facade):
    assert execution.preflight(facade, "no_such_tool") is not None
    assert execution.preflight(facade, "catalog_query", "f16") is not None
    assert execution.preflight(facade, "catalog_query", "f32") is None
    assert execution.preflight(facade, "catalog_query", None) is None


def test_binary_tool_frames_and_placeholder(facade):
    """声明 dtype 且工具在帧清单内：数组出帧，信封字段 None 占位。"""
    env, frames = execution.execute_tool(
        facade, "orbit_family_generation", _ARGS, binary_dtype="f32"
    )
    assert env["status"] == "ok"
    assert len(frames) == 2
    # binary_frames 计数是 sidecar 协议字段，核心不掺入
    assert "binary_frames" not in env
    orbits = env["data"]["orbits"]
    assert [o["states"] for o in orbits] == [None, None]
    assert [o["period"] for o in orbits] == [2.5, None]
    assert env["data"]["mu"] == pytest.approx(1.215e-2)
    arr0, dtype, _ = decode_frame(frames[0])
    assert dtype == "f32"
    np.testing.assert_allclose(arr0, np.arange(18, dtype=float).reshape(3, 6).astype(np.float32))


def test_dtype_on_non_frame_tool_is_ignored(facade):
    """dtype 声明对不在帧清单的工具无影响：照常内联信封，无帧。"""
    env, frames = execution.execute_tool(facade, "catalog_query", {}, binary_dtype="f32")
    assert env["status"] == "ok"
    assert frames == []
    assert env["data"]["message"] == "查询完成：0 条记录"


def test_invalid_params_yields_error_envelope(facade):
    env, frames = execution.execute_tool(facade, "orbit_family_generation", {"libration_point": 99})
    assert env["status"] == "error"
    assert env["error"]["code"] == "INVALID_PARAMS"
    assert frames == []


def test_progress_callback_forwarded_only_when_accepted(facade):
    """回调按方法签名过滤：未声明 progress_callback 形参的工具不注入。"""
    seen: list[float] = []

    env, _ = execution.execute_tool(
        facade,
        "catalog_query",
        {},
        progress_callback=lambda fraction, message=None: seen.append(fraction),
    )
    assert env["status"] == "ok"
    assert seen == [], "未声明形参的工具不得收到回调"


def test_progress_callback_reaches_accepting_method(tmp_path, monkeypatch):
    """声明了 progress_callback 形参的 Facade 方法确实收到核心注入的回调。

    族生成的回调是 Facade 阶段级使用（起止两端上报，不透传算法层），
    所以直接观测回调被触发（先例：tests/api/test_facade_progress.py）。
    """
    import e2m2e.algorithm.family as family_pkg

    class _FakeFamily:
        family_type = "dro"
        system = None
        metadata: dict = {}

        def __init__(self) -> None:
            self.orbits: list = []

        def __iter__(self):
            return iter(self.orbits)

        def __len__(self) -> int:
            return len(self.orbits)

    monkeypatch.setattr(family_pkg, "design_dro_family", lambda *a, **k: _FakeFamily())
    seen: list[float] = []
    facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))
    env, _ = execution.execute_tool(
        facade,
        "orbit_family_generation",
        {"orbit_type": "DRO", "n_orbits": 2},
        progress_callback=lambda fraction, message=None: seen.append(fraction),
    )
    assert env["status"] == "ok"
    assert seen == [0.0, 1.0], "声明的工具应收到核心注入的回调并触发它"


def test_config_payload_roundtrip():
    """Config 载荷往返保真（#601 跨进程契约）。"""
    config = Config(catalog_dir="somewhere", catalog_enabled=False)
    restored = Config.from_payload(config.to_payload())
    assert restored == config


def test_config_payload_rejects_unknown_field():
    """未知字段抛 ValueError：跨进程配置不静默降级（#601）。"""
    with pytest.raises(ValueError, match="bogus"):
        Config.from_payload({"bogus_field": 1})
    with pytest.raises(ValueError, match="必须是对象"):
        Config.from_payload([1, 2])


def test_worker_request_payload_carries_config():
    """worker 请求载荷形状：tool/arguments/config 三键（#601）。"""
    config = Config(catalog_dir="d")
    payload = execution.worker_request_payload("catalog_query", {"a": 1}, config)
    assert set(payload) == {"tool", "arguments", "config"}
    assert payload["tool"] == "catalog_query"
    assert payload["arguments"] == {"a": 1}
    assert Config.from_payload(payload["config"]) == config
    json.dumps(payload)  # 载荷必须可 JSON 行传输
