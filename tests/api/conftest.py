"""api 测试共享辅助。"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(autouse=True)
def _isolated_catalog_dir(monkeypatch, tmp_path):
    """默认 Facade 的自动入库重定向到每测试独立的临时目录（ADR 0031）。"""
    monkeypatch.setenv("E2M2E_CATALOG_DIR", str(tmp_path / "catalog"))


def control_orbit_business_parameters() -> dict[str, inspect.Parameter]:
    """返回 control_orbit 中应由 API 模型表达的业务参数。"""
    from e2m2e.algorithm.station_keeping import control_orbit

    runtime_parameters = {"spice", "kernel_dir", "n_workers", "seed"}
    return {
        name: parameter
        for name, parameter in inspect.signature(control_orbit).parameters.items()
        if name != "input_ephemeris" and name not in runtime_parameters
    }
