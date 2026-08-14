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

另覆盖 60 天合并层收敛回归（#400）：60 天 5 圈（Halo L2 amp=30000
周期 14.6 天）走 2 段 + 1 合并层。修复前合并层固定首末锚定，打靶停在
残差 7.533e-01 km（> 容差 2e-2）抛 DesignNotConvergedError；去掉首末
锚定后合并层收敛（实测 1.44e-03 km）。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.design import design_orbit
from tests.algorithm.design.conftest import make_design_request

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.spice,
    requires_spice,
]

DURATION_SEC = 30 * 86400.0
OUTPUT_STEP_SEC = 3600.0
# et_grid = et0 + arange(0, duration + 0.5·step, step)，含首末点
N_EXPECTED = int(DURATION_SEC / OUTPUT_STEP_SEC) + 1


@pytest.fixture(scope="module")
def halo_result():
    """GUI 默认参数的 30 天 Halo segmented 设计，模块内复用一次。"""
    return design_orbit(
        make_design_request(
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
        make_design_request(
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


MERGE_DURATION_SEC = 60 * 86400.0  # 5 圈（n_rev=5），2 段 + 1 合并层


@pytest.fixture(scope="module")
def halo_result_merge():
    """60 天 Halo segmented 设计：走合并层（#400 回归）。"""
    return design_orbit(
        make_design_request(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            duration=MERGE_DURATION_SEC,
            output_step=OUTPUT_STEP_SEC,
        )
    )


def test_merge_layer_converges(halo_result_merge):
    """合并层收敛回归（#400）：60 天设计成功且星历逐点对齐。

    修复前合并层固定首末两端（远月点锚点），打靶停在残差 7.533e-01 km
    （> 容差 2e-2）抛 DesignNotConvergedError；各段独立打靶均收敛
    （1.86e-02 / 4.04e-03 km），证明连续解存在、问题在合并层约束。
    """
    res = halo_result_merge
    assert res.correction is not None
    assert res.correction.max_residual < 2e-2
    n_expected = int(MERGE_DURATION_SEC / OUTPUT_STEP_SEC) + 1
    eph = res.ephemeris
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected


MULTILAYER_DURATION_SEC = 180 * 86400.0  # 13 圈（n_rev=13），5 段 + 3 合并层


@pytest.fixture(scope="module")
def halo_result_multilayer():
    """180 天 Halo segmented 设计：3 层合并（#400 多层回归）。

    修复前合并层锚定致 180 天停在第 1 层（残差 6.78e-01 km）；修复后
    3 层合并（5 段 → 3 → 2 → 1）全程收敛，覆盖合并层 2/3 的深层路径。
    """
    return design_orbit(
        make_design_request(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            duration=MULTILAYER_DURATION_SEC,
            output_step=OUTPUT_STEP_SEC,
        )
    )


def test_multilayer_merge_converges(halo_result_multilayer):
    """多层合并收敛回归（#400）：180 天设计成功、星历逐点对齐、会合系保形。

    保形判据：会合系 x ∈ [1.08, 1.22]（紧邻 L2 的 Halo 轨道管）且 y 对称
    （实测 180 天星历 x∈[1.111, 1.187]、y∈±0.107；圈间漂移是星历固有
    准周期特征，由站保处理）。
    """
    res = halo_result_multilayer
    assert res.correction is not None
    assert res.correction.max_residual < 2e-2
    n_expected = int(MULTILAYER_DURATION_SEC / OUTPUT_STEP_SEC) + 1
    eph = res.ephemeris
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    syn = np.asarray(eph.synodic_position)
    x, y = syn[:, 0], syn[:, 1]
    assert x.min() > 1.08 and x.max() < 1.22, f"会合系 x 超界: [{x.min():.4f}, {x.max():.4f}]"
    assert abs(y.max() + y.min()) < 0.05, f"会合系 y 不对称: max={y.max():.4f} min={y.min():.4f}"
