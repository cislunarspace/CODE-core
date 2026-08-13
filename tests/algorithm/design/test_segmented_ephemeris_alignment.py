"""segmented 星历时间轴对齐回归测试。

回归 bug：``ForceModel._prepare_t_eval`` 在 t_eval 末尾自动追加段终点 tf，
``design_orbit`` 的 segmented 逐段积分把每段的 tf 端点状态一并拼进
``states_dense``，位置数组比时间网格 ``et_grid`` 多出段数个点（30 天 1
小时步长实测 746 点 vs 721 点，每段多 1 点）；``batch_j2000_to_synodic``
按索引把位置与旋转时刻配对，错位逐段累积，星历会合系曲线一圈一圈偏离
Halo 轨道（GUI 观感"慢慢发散"）。

用 GUI 默认参数的 30 天 Halo（L2、30000 km、phase 0）端到端回归：
长度一致判据直接对应根因（位置-时间错位），y 对称判据直接对应症状
（修复前实测错位把 y 拉到单侧 [-0.266, 0.010]，正常 ±0.107）。

另覆盖右开 mask 的尾部覆盖回归（duration 落入最后打靶节点与 n_rev 圈
终点之间的窗口时，et_grid 尾部点曾无段覆盖、长度断言报错）：
43 天与 30 天同为 3 圈（n_rev=3），但 43 天超出最后节点，修复前直接
ValueError。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest

_SPICE_KERNEL_DIR = os.environ.get(
    "SPICE_KERNEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels"),
)
_SPICE_AVAILABLE = os.path.isdir(_SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(_SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
]

DURATION_SEC = 30 * 86400.0
OUTPUT_STEP_SEC = 3600.0
# et_grid = et0 + arange(0, duration + 0.5·step, step)，含首末点
N_EXPECTED = int(DURATION_SEC / OUTPUT_STEP_SEC) + 1


@pytest.fixture(scope="module")
def halo_result():
    """GUI 默认参数的 30 天 Halo segmented 设计，模块内复用一次。"""
    return design_orbit(
        DesignOrbitRequest(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            duration=DURATION_SEC,
            output_step=OUTPUT_STEP_SEC,
        )
    )


def test_ephemeris_fields_aligned_with_time_grid(halo_result):
    """星历位置/速度/会合系点数必须与时间字段一致（位置-时间错位回归）。"""
    eph = halo_result.ephemeris
    assert len(eph) == N_EXPECTED
    assert len(eph.position_km) == N_EXPECTED
    assert len(eph.velocity_mps) == N_EXPECTED
    assert len(eph.synodic_position) == N_EXPECTED


def test_synodic_y_symmetric(halo_result):
    """会合系 y 振幅对称（错位回归：逐段累积的时间错位把 y 拉到单侧）。"""
    syn = np.asarray(halo_result.ephemeris.synodic_position)
    y = syn[:, 1]
    # 正常 Halo y ∈ ±0.107，|max+min| ≈ 0；错位时 y ∈ [-0.266, 0.010]，
    # |max+min| ≈ 0.256。阈值 0.05 远小于错位特征、远大于正常残差。
    assert abs(y.max() + y.min()) < 0.05, f"会合系 y 不对称: max={y.max():.4f} min={y.min():.4f}"


LONG_DURATION_SEC = 43 * 86400.0  # 3 圈（n_rev=3），但超出最后打靶节点


@pytest.fixture(scope="module")
def halo_result_long():
    """43 天 Halo segmented 设计：et_grid 尾部落入无节点覆盖窗口。"""
    return design_orbit(
        DesignOrbitRequest(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            duration=LONG_DURATION_SEC,
            output_step=OUTPUT_STEP_SEC,
        )
    )


def test_ephemeris_tail_beyond_last_patch_point(halo_result_long):
    """尾部窗口回归：et_grid 超出最后打靶节点时星历仍完整逐点对齐。

    修复前右开 mask 把尾部点排除在段外，此处曾报 ValueError（星历状态
    1022 点 vs 时间网格 1033 点，差 11 点）。
    """
    eph = halo_result_long.ephemeris
    n_expected = int(LONG_DURATION_SEC / OUTPUT_STEP_SEC) + 1
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected
