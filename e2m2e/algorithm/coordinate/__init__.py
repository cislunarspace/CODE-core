"""坐标转换算法：IAU2006/synodic↔J2000/GCRS↔EBCRS。

转换**算法**归这里（ADR 0011 迁移，源：``core/coordinate/``），
``data/frames/`` 只留数据（EOP/闰秒/历表句柄，ADR 0015）。强化现有
Axes/Origin/CoordinateSystem 抽象（不新增 Frame 抽象）：所有坐标系表达
为 Axes + Origin + CoordinateSystem，时空间联合转换作为 CoordinateSystem
扩展方法。转换算法最终留 Python，Rust 下沉是后续性能优化。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .axes import Axes
from .coordinate_system import CoordinateSystem
from .dynamic_axes import DynamicAxes
from .gcrs_ebcrs import GCRSEBCRSSystem
from .gmat_itrf import GmatItrfReduction
from .gmat_time import TimeSystemConverter
from .iau_2006 import iau2000eq_matrix, precession_angles
from .origin import Origin
from .rho_bridge import eci_to_rho, rho_to_eci
from .standard_axes import (
    GMATITRFAxes,
    IAU2000EqAxes,
    ICRSAxes,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    standard_icrf,
    standard_itrf,
)
from .standard_dynamic_axes import LVLHAxes, VNBAxes
from .standard_origins import CelestialBodyOrigin, InertialOrigin
from .synodic_axes import SynodicAxes
from .synodic_j2000 import SynodicJ2000System
from .xys import ErfaXysProvider, XysProvider

__all__ = [
    "Axes",
    "Origin",
    "CoordinateSystem",
    "DynamicAxes",
    "ICRSAxes",
    "IAU2000EqAxes",
    "ITRFSpiceAxes",
    "ITRFAxes",
    "GMATITRFAxes",
    "ITRFApproxAxes",
    "standard_icrf",
    "standard_itrf",
    "LVLHAxes",
    "VNBAxes",
    "CelestialBodyOrigin",
    "InertialOrigin",
    "SynodicAxes",
    "SynodicJ2000System",
    "GCRSEBCRSSystem",
    "GmatItrfReduction",
    "TimeSystemConverter",
    "ErfaXysProvider",
    "XysProvider",
    "iau2000eq_matrix",
    "precession_angles",
    "rho_to_eci",
    "eci_to_rho",
    "spacetime_convert",
]

#: SPICE ET 秒 → JD_TDB 转换基准
_JD_TDB_AT_ET0 = 2451545.0
_SECONDS_PER_DAY = 86400.0


def spacetime_convert(
    transform_type: str,
    state: Any,
    epoch: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """时空坐标转换统一入口。

    按 transform_type 分发到 SynodicJ2000System 或 GCRSEBCRSSystem。
    不缓存转换器实例（每次调用新建）。

    Args:
        transform_type: ``synodic_to_j2000`` / ``j2000_to_synodic`` /
            ``gcrs_to_ebcrs`` / ``ebcrs_to_gcrs``。
        state: 单条状态向量。
        epoch: 时间值（JD_TDB，synodic 转换时为无量纲时间 t_syn）。
        kwargs: 额外参数（``et0_jd`` 参考历元 JD_TDB；``ephemeris_path``
            GCRS↔EBCRS 必需）。

    Returns:
        ``{"state": ndarray, "time": float, "transform_type": str, "details": dict}``
    """
    state_arr = np.asarray(state, dtype=float)
    if state_arr.ndim != 1:
        raise ValueError(f"state 应为一维数组，实际形状 {state_arr.shape}")

    et0_jd = float(kwargs.get("et0_jd", _JD_TDB_AT_ET0))
    et0 = (et0_jd - _JD_TDB_AT_ET0) * _SECONDS_PER_DAY

    if transform_type in ("synodic_to_j2000", "j2000_to_synodic"):
        from ...data.kernels.manager import SPICEManager
        from ..design.design_orbit import load_design_kernels
        from ..dynamics.cr3bp_system import CR3BP_System

        conv: Any
        spice = SPICEManager()
        load_design_kernels(spice, kwargs.get("kernel_dir"))

        MU_EM = 1.21506683e-2
        cr3bp_system = CR3BP_System(
            mu=MU_EM, primary="Earth", secondary="Moon"
        )._with_default_scales()
        conv = SynodicJ2000System(cr3bp_system=cr3bp_system, spice=spice)
    elif transform_type in ("gcrs_to_ebcrs", "ebcrs_to_gcrs"):
        ephemeris_path = kwargs.get("ephemeris_path")
        if not ephemeris_path:
            raise ValueError("GCRS↔EBCRS 转换需要 ephemeris_path（含时间星历的历表路径）")
        conv = GCRSEBCRSSystem(ephemeris_path)
    else:
        raise ValueError(f"未知 transform_type: {transform_type!r}")

    if transform_type == "synodic_to_j2000":
        t_syn = float(epoch)
        result = conv.synodic_to_j2000(state_arr, t_syn, et0)
        out_time = et0_jd + t_syn * conv._get_time_unit() / _SECONDS_PER_DAY
    elif transform_type == "j2000_to_synodic":
        t_syn = float(epoch)
        result = conv.j2000_to_synodic(state_arr, t_syn, et0)
        out_time = et0_jd + t_syn * conv._get_time_unit() / _SECONDS_PER_DAY
    elif transform_type == "gcrs_to_ebcrs":
        if state_arr.shape[0] != 3:
            raise ValueError(f"GCRS→EBCRS 输入应为 3 维位置，实际 {state_arr.shape[0]} 维")
        out_time, result = conv.gcrs_to_ebcrs(float(epoch), state_arr[:3])
    elif transform_type == "ebcrs_to_gcrs":
        if state_arr.shape[0] != 3:
            raise ValueError(f"EBCRS→GCRS 输入应为 3 维位置，实际 {state_arr.shape[0]} 维")
        out_time, result = conv.ebcrs_to_gcrs(float(epoch), state_arr[:3])
    else:
        raise ValueError(f"未知 transform_type: {transform_type!r}")

    return {
        "state": np.asarray(result, dtype=float),
        "time": float(out_time),
        "transform_type": transform_type,
        "details": {"epoch_in": float(epoch), "et0_jd": et0_jd},
    }
