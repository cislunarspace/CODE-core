"""MCP 协议层测试（issue #510 / ADR 0014）：纯派生注册、schema、统一信封。

只测进程内逻辑（tool_specs / handle_call_tool / envelope），不需要真实
stdio 传输；handler 纯函数与 create_server 注册的是同一批。
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.interface]

pytest.importorskip("mcp")  # [mcp] extra 未装时整文件跳过（协议层依赖，ADR 0014）

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
    # placeholder 不注册（issue #510 决策）
    placeholders = {i.name for i in tool_inventory(facade) if i.status == "placeholder"}
    assert placeholders, "前提：Facade 至少有一个 placeholder 工具"
    assert not placeholders & {s.name for s in specs}


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


def test_placeholder_tool_not_registered(facade):
    placeholders = [i.name for i in tool_inventory(facade) if i.status == "placeholder"]
    for name in placeholders:
        result = handle_call_tool(facade, name, {})
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


def test_cli_parser_has_mcp_serve():
    from e2m2e.api.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["mcp-serve"])
    assert args.command == "mcp-serve"
