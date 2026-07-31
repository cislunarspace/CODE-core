"""架构骨架占位/行为测试。

未实现能力（角动量管理、ECOM 光压、LGA/WSB、transfer_orbit、MCP/tools/logging）
断言抛 NotImplementedError 且错误信息含能力名；已实现能力（design_orbit/
control_orbit/propagate_orbit/family 初猜、Facade 一档接入）改行为冒烟测试。
"""

from __future__ import annotations

import pytest


def test_design_orbit_implemented():
    """design_orbit 已实现：形状参数校验先于占位抛错。"""
    from e2m2e.algorithm.design import design_orbit

    with pytest.raises(ValueError, match="duration"):
        design_orbit("DRO", duration=0.0)


def test_control_orbit_implemented():
    """control_orbit 已实现：参数校验先于占位抛错。"""
    from e2m2e.algorithm.station_keeping import control_orbit

    with pytest.raises(ValueError, match="control_mode"):
        control_orbit(None, control_mode=9)


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


def test_propagate_orbit_implemented():
    """propagate_orbit 已实现：形状校验先于占位抛错。"""
    from e2m2e.algorithm.propagation import propagate_orbit

    with pytest.raises(ValueError, match="initial_state"):
        propagate_orbit(None, None, 1.0)


def test_family_design_implemented():
    """六类初猜已实现：真实数值族行走，非 NotImplementedError 占位。"""
    from e2m2e.algorithm.family import Cr3bpOrbitError, design_dro

    # 极小振幅在族行走内报收敛失败（Cr3bpOrbitError），证明已接入真实实现。
    with pytest.raises(Cr3bpOrbitError):
        design_dro(1.0)

    import inspect

    from e2m2e.algorithm import family

    for name in ("design_halo", "design_nrho", "design_lissajous", "design_triangular"):
        fn = getattr(family, name)
        assert callable(fn)
        assert "NotImplementedError" not in (inspect.getsource(fn) or "")


def test_ecom_srp_placeholder():
    """ECOM 光压（原 #253）占位：对外承诺能力。"""
    from e2m2e.algorithm.forces import ecom_solar_radiation_pressure

    with pytest.raises(NotImplementedError, match="ECOM"):
        ecom_solar_radiation_pressure()


def test_facade_placeholder():
    """Facade：已实现方法接入真实编排，未实现方法保持占位。"""
    from e2m2e.api.facade import Facade

    facade = Facade()
    # 已实现：参数校验先于占位抛错（OrbitError 包装 INVALID_PARAMS）
    from e2m2e.api.models import OrbitError

    with pytest.raises(OrbitError, match="INVALID_PARAMS"):
        facade.design_orbit(orbit_type="DRO", duration=0.0)
    with pytest.raises(OrbitError, match="INVALID_PARAMS"):
        facade.control_orbit(control_mode=9)
    # 未实现：保持占位
    with pytest.raises(NotImplementedError, match="transfer_design"):
        facade.transfer_design(transfer_type="HMN")
    with pytest.raises(NotImplementedError, match="orbit_propagation"):
        facade.orbit_propagation()


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
