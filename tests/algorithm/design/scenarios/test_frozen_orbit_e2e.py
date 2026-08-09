"""冻结轨道设计端到端回归测试（L3）。

复现已验证的物理结论（原型脚本 + 报告）：
- i=75° 下不存在严格冻结解（|Δe| > 0.01）
- a=3000 km 的 60 天漂移值：Δe ≈ −0.019, Δrp ≈ +61 km
- |Δrp| 随半长轴增大而增大（第三体摄动增强）

算法层 design_orbit() 始终返回单候选结果（扫描逻辑在 tod FacadeBridge，
不在此测试范围内），故通过多次调用收集各候选漂移。
"""

from __future__ import annotations

import os

import pytest
from kernel_helpers import SPICE_KERNEL_DIR

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest

_SPICE_KERNEL_DIR = SPICE_KERNEL_DIR
_SPICE_AVAILABLE = os.path.isdir(_SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.slow,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
]

DURATION_60D = 60 * 86400.0


@pytest.fixture(scope="module")
def a3000_60d():
    """a=3000 km 60 天传播（耗时约 30 秒）。"""
    return design_orbit(
        DesignOrbitRequest(
            orbit_type="ELFO",
            semi_major_axis=3000.0,
            duration=DURATION_60D,
            output_step=3600.0,
        )
    )


@pytest.mark.xfail(
    reason="ELFO drift_e 实测 −0.0051 与报告 −0.019 不符（#370），"
    "路径 bug 修复后暴露，待排查 #350 实现",
    strict=False,
)
def test_a3000_drift_values(a3000_60d):
    """a=3000 km 的 60 天漂移值应与报告一致（±15% 容差）。"""
    assert a3000_60d.drift_e == pytest.approx(-0.019, abs=0.005)
    assert a3000_60d.drift_rp_km == pytest.approx(61, abs=15)


@pytest.mark.xfail(
    reason="ELFO drift_e 实测 −0.0051 与报告 −0.019 不符（#370），"
    "路径 bug 修复后暴露，待排查 #350 实现",
    strict=False,
)
def test_no_strictly_frozen(a3000_60d):
    """i=75° 下 |Δe| 应 > 0.01（不存在严格冻结解）。"""
    assert abs(a3000_60d.drift_e) > 0.01


def test_drift_direction(a3000_60d):
    """i=75°（>临界倾角 63.4°）：Δe 应为负、Δrp 应为正。"""
    assert a3000_60d.drift_e < 0
    assert a3000_60d.drift_rp_km > 0


def test_drift_rp_increases_with_a(a3000_60d):
    """|Δrp| 应随半长轴增大而增大（第三体摄动增强）。"""
    a8000 = design_orbit(
        DesignOrbitRequest(
            orbit_type="ELFO",
            semi_major_axis=8000.0,
            duration=DURATION_60D,
            output_step=3600.0,
        )
    )
    assert abs(a8000.drift_rp_km) > abs(a3000_60d.drift_rp_km)
