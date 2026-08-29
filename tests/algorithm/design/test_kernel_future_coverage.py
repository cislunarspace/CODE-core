"""回归：body-fixed 内核清单须覆盖未来历元（预测 PCK 先于历史 PCK 加载）。

历史 PCK（``earth_latest_high_prec.bpc``）覆盖终点随仓库快照日期固定
（约 2026 年中）；设计/站保流水线只装历史 PCK 时，起始历元加默认年尺度
传播一旦越过终点，Rust 星历缓存构建即 SPICE FRAMEDATANOTFOUND
（``design_orbit`` 经 MCP/CLI 暴露后未来历元是真实使用路径）。修复：
``SPICEEarthPredictedKernel.bpc`` 先于历史 PCK 加载——SPICE 对重叠覆盖段
取后加载者，过去时段仍取历史高精度数据，未来时段由预测数据补齐。
"""

from __future__ import annotations

import pytest
from kernel_helpers import SPICE_KERNEL_DIR, requires_spice

from e2m2e.algorithm.design.design_orbit import (
    _BODY_FIXED_KERNELS,
    load_design_kernels,
)
from e2m2e.algorithm.station_keeping import monte_carlo
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = [pytest.mark.interface, pytest.mark.spice, requires_spice]

_PREDICT = "SPICEEarthPredictedKernel.bpc"
_HISTORICAL = "earth_latest_high_prec.bpc"
#: 历史 PCK 覆盖终点（随快照更新只会后移）与预测 PCK 终点（约 2037-04）之间。
_FUTURE_EPOCH_UTC = "2030-01-01"


def test_predict_pck_precedes_historical() -> None:
    """设计/站保两条内核清单都含预测 PCK，且加载顺序在历史 PCK 之前。"""
    assert _PREDICT in _BODY_FIXED_KERNELS
    assert _BODY_FIXED_KERNELS.index(_PREDICT) < _BODY_FIXED_KERNELS.index(_HISTORICAL)

    mc_names = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in monte_carlo._kernel_paths()]
    assert _PREDICT in mc_names
    assert mc_names.index(_PREDICT) < mc_names.index(_HISTORICAL)


def test_itrf93_orientation_available_at_future_epoch() -> None:
    """经生产加载路径后，超出历史 PCK 覆盖的历元 ITRF93 旋转可用。"""
    spice = SPICEManager()
    loaded = load_design_kernels(spice, kernel_dir=SPICE_KERNEL_DIR)
    assert any(p.endswith(_PREDICT) for p in loaded)
    matrix = spice.pxform("J2000", "ITRF93", spice.utc_to_et(_FUTURE_EPOCH_UTC))
    assert matrix.shape == (3, 3)
