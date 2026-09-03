"""MCP 协议层测试：纯派生注册、schema、统一信封。

只测进程内逻辑（tool_specs / handle_call_tool / envelope），不需要真实
stdio 传输；handler 纯函数与 create_server 注册的是同一批。
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.interface]

pytest.importorskip("mcp")  # [mcp] extra 未装时整文件跳过（协议层依赖）

from e2m2e.api.config import Config  # noqa: E402
from e2m2e.api.facade import Facade, tool_inventory  # noqa: E402
from e2m2e.api.mcp import (  # noqa: E402
    envelope,
    handle_call_tool,
    handle_list_tools,
    tool_specs,
)
from e2m2e.api.models import DesignOrbitRequest  # noqa: E402


@pytest.fixture
def facade(tmp_path, monkeypatch) -> Facade:
    monkeypatch.setenv("E2M2E_CATALOG_DIR", str(tmp_path / "catalog"))
    return Facade(Config())


def test_registered_tools_match_inventory(facade):
    """注册清单 = tool_inventory() 中 implemented 的条目（单一来源，不漂移）。"""
    specs = tool_specs(facade)
    expected = [i.name for i in tool_inventory(facade) if i.status == "implemented"]
    assert [s.name for s in specs] == expected


def test_placeholder_mechanism_filters_stub_tools():
    """placeholder 元数据机制：标记占位的方法不注册（ADR 0014）。

    ADR 0043 决策 4 后 Facade 不再持有占位声明，用桩对象验证机制本身，
    待未来二档方法在领域类落地时仍走同一过滤。
    """
    from e2m2e.api.facade import mcp_exposed

    class _Stub:
        @mcp_exposed(request_model=DesignOrbitRequest)
        def real_tool(self, **params):
            return {}

        @mcp_exposed(status="placeholder")
        def placeholder_tool(self, **params):
            raise NotImplementedError

    assert [s.name for s in tool_specs(_Stub())] == ["real_tool"]


def test_input_schema_derived_from_request_model(facade):
    specs = {s.name: s for s in tool_specs(facade)}
    assert specs["design_orbit"].input_schema == DesignOrbitRequest.model_json_schema()
    # 独立验证关键字段确实进了 schema（不是仅对照同一来源的同义反复）
    props = specs["design_orbit"].input_schema["properties"]
    assert props["orbit_type"]["type"] == "string"
    assert props["output_step"]["default"] == 3600.0
    # 无 request_model 的方法以无参 schema 注册
    no_model = [
        i.name
        for i in tool_inventory(facade)
        if i.request_model is None and i.status == "implemented"
    ]
    for name in no_model:
        assert specs[name].input_schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }


def test_handle_list_tools(facade):
    listing = handle_list_tools(facade)
    assert [t.name for t in listing] == [s.name for s in tool_specs(facade)]
    assert all(t.description for t in listing)


def test_ok_envelope_wraps_response(facade):
    # 用便宜的 catalog_query（空库即成功）验证信封；design_orbit 会跑真算法，太慢。
    result = handle_call_tool(facade, "catalog_query", {})
    assert not result.is_error
    env = json.loads(result.content[0].text)
    assert set(env) == {"status", "data", "error", "meta"}
    assert env["status"] == "ok"
    assert env["error"] is None
    assert env["data"], "catalog_query 成功时 data 应为 Response 序列化"


def test_invalid_params_envelope(facade):
    result = handle_call_tool(facade, "catalog_query", {"libration_point": 99})
    assert result.is_error
    env = json.loads(result.content[0].text)
    assert env["status"] == "error"
    assert env["error"]["code"] == "INVALID_PARAMS"
    assert env["error"]["details"]["errors"]


def test_orbit_error_translated_without_traceback(facade):
    # catalog_get 对空库报 RECORD_NOT_FOUND（OrbitError 路径），无 traceback 泄漏
    result = handle_call_tool(facade, "catalog_get", {"record_id": "no-such-record"})
    env = json.loads(result.content[0].text)
    assert env["status"] == "error"
    assert env["error"]["code"] == "RECORD_NOT_FOUND"
    assert "Traceback" not in result.content[0].text


def test_missing_required_field_invalid_params(facade):
    # design_orbit 缺必填字段 → Pydantic ValidationError → INVALID_PARAMS
    result = handle_call_tool(facade, "design_orbit", {})
    env = json.loads(result.content[0].text)
    assert env["error"]["code"] == "INVALID_PARAMS"


def test_unknown_tool_envelope(facade):
    result = handle_call_tool(facade, "no_such_tool", {})
    assert result.is_error
    env = json.loads(result.content[0].text)
    assert env["error"]["code"] == "TOOL_NOT_FOUND"


def test_create_server_binds_facade(facade):
    import anyio
    from mcp.types import ListToolsRequest, PaginatedRequestParams

    from e2m2e.api.mcp import create_server

    server = create_server(facade)
    assert server.name == "e2m2e"

    async def run():
        entry = server.get_request_handler("tools/list")
        assert entry is not None
        result = await entry.handler(
            None, ListToolsRequest(method="tools/list", params=PaginatedRequestParams())
        )
        return result

    listing = anyio.run(run)
    listed = listing.tools if hasattr(listing, "tools") else listing
    assert [t.name for t in listed] == [t.name for t in handle_list_tools(facade)]


def test_dispatch_tool_omits_unset_fields():
    """未提供的字段不得变成显式 None 传给工具方法。"""

    class Req(BaseModel):
        a: int
        b: str | None = None

    seen: dict = {}

    class Tool:
        request_model = Req

        def __call__(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    result, err = envelope.dispatch_tool(Tool(), {"a": 1})
    assert err is None and result == "ok"
    assert seen == {"a": 1}


def test_invoke_tool_internal_error():
    class Boom:
        request_model = None

        def __call__(self, **kwargs):
            raise RuntimeError("secret detail")

    env = envelope.invoke_tool(Boom(), {})
    assert env["status"] == "error"
    assert env["error"]["code"] == "INTERNAL_ERROR"
    assert "secret detail" not in env["error"]["message"]
    assert "RuntimeError" in env["error"]["message"]


def test_family_response_serializes_orbit_members():
    """族生成响应（Orbit 成员）序列化为 JSON 数据，而非 INTERNAL_ERROR。

    降级契约与 sidecar 帧契约同款：成员留 states/times/period/family_type，
    System 鸭子类型透传 mu。
    """
    import numpy as np

    from e2m2e.api.models import FamilyGenerationResponse
    from e2m2e.data.templates import ConvergenceState, FailureCause
    from e2m2e.data.types.orbit import Orbit

    orbit = Orbit(states=np.eye(6, 6), times=np.linspace(0.0, 1.0, 6))

    class SystemStub:
        mu = 1.21506683e-2

    class FamTool:
        request_model = None

        def __call__(self, **kwargs):
            return FamilyGenerationResponse(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="族生成完成",
                orbits=[orbit],
                system=SystemStub(),
                requested_members=1,
                generated_members=1,
            )

    env = envelope.invoke_tool(FamTool(), {})
    assert env["status"] == "ok"
    json.dumps(env)  # MCP 传输层直接 dumps 信封（server._to_result）
    # 降级序列化路径（Orbit/ndarray 触发）枚举须取值而非 <ConvergenceState> 占位
    assert env["data"]["status"] == "converged"
    assert env["data"]["cause"] == "none"
    member = env["data"]["orbits"][0]
    assert member["states"][0] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert member["times"][0] == 0.0
    assert member["period"] is None  # eye(6) 非周期轨迹，x 零交叉检测不到周期
    assert env["data"]["system"] == {"mu": pytest.approx(1.21506683e-2)}


def test_catalog_record_response_serializes_arrays():
    """catalog_get/promote 响应的数组段（ndarray）内联为嵌套 list。"""
    import numpy as np

    from e2m2e.api.models import CatalogRecordResponse
    from e2m2e.data.templates import ConvergenceState, FailureCause

    record = CatalogRecordResponse(
        record_id="r1",
        created_at="2026-08-25T00:00:00+00:00",
        source_tool="design_orbit",
        source_record_id=None,
        orbit_family="dro",
        libration_point=None,
        jacobi=None,
        amplitude=None,
        has_cr3bp=True,
        has_ephemeris=False,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="ok",
        member_count=1,
        tags=[],
        note="",
        scalars={"mu": 1.21506683e-2},
        request={},
        members=[],
        arrays={"cr3bp/states": np.zeros((2, 6)), "cr3bp/times": np.array([0.0, 1.0])},
    )
    env = envelope.ok_envelope(record)
    assert env["status"] == "ok"
    json.dumps(env)
    assert env["data"]["arrays"]["cr3bp/states"] == [[0.0] * 6, [0.0] * 6]
    assert env["data"]["arrays"]["cr3bp/times"] == [0.0, 1.0]


def test_spacetime_times_description_states_dual_units():
    """times 字段描述须写明双单位契约（会合系是 t_syn 而非 JD_TDB）。"""
    from e2m2e.api.models import SpacetimeTransformRequest

    desc = SpacetimeTransformRequest.model_json_schema()["properties"]["times"]["description"]
    assert "JD_TDB" in desc
    assert "t_syn" in desc


def test_cli_parser_has_mcp_serve():
    from e2m2e.api.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["mcp-serve"])
    assert args.command == "mcp-serve"


# ---------------------------------------------------------------------------
# 长任务进度桥接（#576 Phase 1）
# ---------------------------------------------------------------------------


def _bridge_session_recorder(calls):
    """带 report_progress 录制的 Session 替身。"""

    class _Session:
        async def report_progress(self, progress, total=None, message=None):
            calls.append((progress, total, message))

    return _Session()


class TestProgressBridge:
    def test_make_reporter_none_without_token(self):
        """客户端未带 progressToken：返回 None（零开销直通）。"""
        from e2m2e.api.mcp import server as mcp_server

        class _Ctx:
            meta = None
            session = None

        assert mcp_server.make_progress_reporter(_Ctx()) is None

    def test_reporter_forwards_to_session_from_worker_thread(self):
        """worker 线程里的回调经捕获的事件循环切回发通知（工厂须在 loop 上调用）。"""
        import anyio

        from e2m2e.api.mcp import server as mcp_server

        calls: list = []

        class _Meta:
            progress_token = "tok"

        class _Ctx:
            meta = _Meta()
            session = _bridge_session_recorder(calls)

        async def scenario():
            reporter = mcp_server.make_progress_reporter(_Ctx())
            assert reporter is not None
            await anyio.to_thread.run_sync(lambda: reporter(0.5, "half"))
            await anyio.to_thread.run_sync(lambda: reporter(1.0))

        anyio.run(scenario)
        assert calls == [(0.5, 1.0, "half"), (1.0, 1.0, None)]

    def test_reporter_throttles_burst_but_not_final(self, monkeypatch):
        """节流：窗口内的中间通知丢弃；fraction=1.0 收尾永不节流。"""
        import anyio

        from e2m2e.api.mcp import server as mcp_server

        monkeypatch.setattr(mcp_server, "_PROGRESS_MIN_INTERVAL_SEC", 60.0)
        calls: list = []

        class _Meta:
            progress_token = "tok"

        class _Ctx:
            meta = _Meta()
            session = _bridge_session_recorder(calls)

        async def scenario():
            reporter = mcp_server.make_progress_reporter(_Ctx())
            await anyio.to_thread.run_sync(lambda: reporter(0.2))
            await anyio.to_thread.run_sync(lambda: reporter(0.4))
            await anyio.to_thread.run_sync(lambda: reporter(1.0))

        anyio.run(scenario)
        assert [progress for progress, _, _ in calls] == [0.2, 1.0]

    def test_tools_call_route_passes_extra_without_breaking_fast_tools(self, facade):
        """extra 注入路径对不接进度回调的工具零影响（catalog_query 真调用）。"""
        env = json.loads(
            handle_call_tool(
                facade,
                "catalog_query",
                {},
                extra_kwargs={"progress_callback": lambda f, m=None: None},
            )
            .content[0]
            .text
        )
        assert env["status"] == "ok"


def test_invoke_tool_injects_extra_kwargs_only_when_accepted():
    """dispatch/invoke 注入 extra 时按方法签名过滤，未接受形参静默丢弃。"""
    from e2m2e.api.mcp import envelope

    def method_with_cb(value, progress_callback=None):
        return [value, progress_callback]

    env = envelope.invoke_tool(
        method_with_cb, {"value": 1}, extra_kwargs={"progress_callback": "CB"}
    )
    assert env["status"] == "ok"
    assert env["data"] == [1, "CB"]

    def method_without_cb(value):
        return value

    env = envelope.invoke_tool(
        method_without_cb, {"value": 2}, extra_kwargs={"progress_callback": "CB"}
    )
    assert env["status"] == "ok"
    assert env["data"] == 2
