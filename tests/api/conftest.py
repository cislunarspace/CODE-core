"""api 测试共享辅助。"""

from __future__ import annotations

import inspect


def control_orbit_business_parameters() -> dict[str, inspect.Parameter]:
    """返回 control_orbit 中应由 API 模型表达的业务参数。"""
    from e2m2e.algorithm.station_keeping import control_orbit

    runtime_parameters = {"spice", "kernel_dir", "n_workers", "seed"}
    return {
        name: parameter
        for name, parameter in inspect.signature(control_orbit).parameters.items()
        if name != "input_ephemeris" and name not in runtime_parameters
    }
