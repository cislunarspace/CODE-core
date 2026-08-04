"""架构骨架占位/行为测试。

未实现能力（MCP/tools 服务器占位）
断言抛 NotImplementedError 且错误信息含能力名；已实现能力（design_orbit/
control_orbit/propagate_orbit/family 初猜、transfer_orbit、momentum_management、
Facade 一档接入）改行为冒烟测试。
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


def test_momentum_management_implemented():
    """角动量管理（原 #261）已实现：发动机布局校验先于占位抛错。"""
    import numpy as np

    from e2m2e.algorithm.station_keeping import EngineLayout

    # N=3 发动机布局不足 6 约束，EngineLayout 构造抛 ValueError（非占位）。
    with pytest.raises(ValueError, match="不足 6"):
        EngineLayout(
            positions_m=np.zeros((3, 3)),
            directions=np.tile([1.0, 0.0, 0.0], (3, 1)),
        )


def test_transfer_orbit_implemented():
    """transfer_orbit 已实现：HMN 缺参校验先于占位抛错。"""
    from e2m2e.algorithm.transfer import transfer_orbit

    with pytest.raises(ValueError, match="HMN 转移需要 tli_params"):
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


def test_ecom_srp_implemented():
    """ECOM 光压（原 #253）已实现：可正常构造并输出 to_rust_spec。"""
    from e2m2e.algorithm.forces import EcomSolarRadiationPressure

    dyb = [0.01] + [0.0] * 8
    ecom = EcomSolarRadiationPressure(dyb=dyb)
    spec = ecom.to_rust_spec()
    assert spec[0] == "ecom_srp"
    assert spec[1] == dyb


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
    with pytest.raises(OrbitError, match="INVALID_PARAMS"):
        facade.transfer_design(transfer_type="HMN")
    with pytest.raises(OrbitError, match="INVALID_PARAMS"):
        facade.orbit_propagation()
    with pytest.raises(OrbitError, match="INVALID_PARAMS"):
        facade.spacetime_transform()
    # 未实现：保持占位
    with pytest.raises(NotImplementedError, match="transfer_search"):
        facade.transfer_search()
    with pytest.raises(NotImplementedError, match="low_thrust_design"):
        facade.low_thrust_design()
    with pytest.raises(NotImplementedError, match="manifold_analysis"):
        facade.manifold_analysis()
    with pytest.raises(NotImplementedError, match="low_energy_transfer"):
        facade.low_energy_transfer()
    with pytest.raises(NotImplementedError, match="relative_motion"):
        facade.relative_motion()


def test_mcp_server_placeholder():
    """MCP create_server 占位。"""
    from e2m2e.api.mcp.server import create_server

    with pytest.raises(NotImplementedError, match="MCP"):
        create_server(None)


def test_tools_logging_implemented():
    """tools/logging 配置工厂已实现。"""
    from e2m2e.tools.logging import configure_logging

    configure_logging(level="ERROR")
