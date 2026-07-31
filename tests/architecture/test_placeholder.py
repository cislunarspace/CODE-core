"""架构骨架占位测试：未实现能力断言抛 NotImplementedError。

按模块一个占位测试文件（ADR 0011/设计共识）：只断言"调用抛 NotImplementedError
+ 错误信息含能力名"。实现完成后改成正常行为测试。
"""

from __future__ import annotations

import pytest


def test_design_orbit_placeholder():
    """design_orbit 占位：抛 NotImplementedError 且信息含能力名。"""
    from e2m2e.algorithm.design import design_orbit

    with pytest.raises(NotImplementedError, match="design_orbit"):
        design_orbit("DRO")


def test_control_orbit_placeholder():
    """control_orbit 占位。"""
    from e2m2e.algorithm.station_keeping import control_orbit

    with pytest.raises(NotImplementedError, match="control_orbit"):
        control_orbit(None)


def test_momentum_management_placeholder():
    """角动量管理（原 #261）占位：对外承诺能力。"""
    from e2m2e.algorithm.station_keeping import momentum_management

    with pytest.raises(NotImplementedError, match="角动量管理"):
        momentum_management(None)


def test_transfer_orbit_placeholder():
    """transfer_orbit 占位。"""
    from e2m2e.algorithm.transfer import transfer_orbit

    with pytest.raises(NotImplementedError, match="transfer_orbit"):
        transfer_orbit("HMN")


def test_propagate_orbit_placeholder():
    """propagate_orbit 占位。"""
    from e2m2e.algorithm.propagation import propagate_orbit

    with pytest.raises(NotImplementedError, match="propagate_orbit"):
        propagate_orbit(None, None, 1.0)


def test_family_design_placeholder():
    """六类初猜占位。"""
    from e2m2e.algorithm.family import design_dro, design_halo, design_nrho

    with pytest.raises(NotImplementedError, match="design_dro"):
        design_dro(10000.0)
    with pytest.raises(NotImplementedError, match="design_halo"):
        design_halo(2, 30000.0)
    with pytest.raises(NotImplementedError, match="design_nrho"):
        design_nrho(2, 1, 5000.0)


def test_ecom_srp_placeholder():
    """ECOM 光压（原 #253）占位：对外承诺能力。"""
    from e2m2e.algorithm.forces import ecom_solar_radiation_pressure

    with pytest.raises(NotImplementedError, match="ECOM"):
        ecom_solar_radiation_pressure()


def test_facade_placeholder():
    """Facade 方法占位。"""
    from e2m2e.api.facade import Facade

    facade = Facade()
    with pytest.raises(NotImplementedError, match="design_orbit"):
        facade.design_orbit(orbit_type="DRO")
    with pytest.raises(NotImplementedError, match="control_orbit"):
        facade.control_orbit()


def test_mcp_server_placeholder():
    """MCP create_server 占位。"""
    from e2m2e.api.mcp.server import create_server

    with pytest.raises(NotImplementedError, match="MCP"):
        create_server(None)


def test_tools_logging_placeholder():
    """tools/logging 配置工厂占位。"""
    from e2m2e.tools.logging import configure_logging

    with pytest.raises(NotImplementedError, match="tools/logging"):
        configure_logging()
