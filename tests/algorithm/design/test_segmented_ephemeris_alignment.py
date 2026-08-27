"""segmented 星历时间轴对齐回归测试。

锁定的失效模式：``ForceModel._prepare_t_eval`` 在 t_eval 末尾自动追加段
终点 tf 时，``design_orbit`` 的 segmented 逐段积分会把每段的 tf 端点状态
一并拼进 ``states_dense``，位置数组比时间网格 ``et_grid`` 多出段数个点；
``batch_j2000_to_synodic`` 按索引把位置与旋转时刻配对，错位逐段累积，
星历会合系曲线一圈一圈偏离 Halo 轨道。

用 GUI 默认参数的 30 天 Halo（L2、30000 km、phase 0）端到端回归：
长度一致判据直接对应根因（位置-时间错位），y 对称判据直接对应症状
（错位把 y 拉到单侧，正常应为对称 ±0.107）。

另覆盖右开 mask 的尾部覆盖（duration 落入最后打靶节点与 n_rev 圈
终点之间的窗口时，et_grid 尾部点必须有段覆盖）。

另覆盖 60 天合并层收敛：60 天 5 圈（Halo L2 amp=30000 周期 14.6 天）
走 2 段 + 1 合并层；合并层不得固定首末锚定，否则打靶收不进容差。
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

    右开 mask 若把尾部点排除在段外，星历点数就会少于时间网格。
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
    """60 天 Halo segmented 设计：走合并层。"""
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
    """合并层收敛回归：60 天设计成功且星历逐点对齐。

    合并层不得固定首末两端锚定；各段独立打靶均收敛即可证明连续解存在、
    问题只会在合并层约束上。
    """
    res = halo_result_merge
    assert res.correction is not None
    assert res.correction.max_residual < 2e-2
    assert res.correction_method == "segmented"
    n_expected = int(MERGE_DURATION_SEC / OUTPUT_STEP_SEC) + 1
    eph = res.ephemeris
    assert len(eph) == n_expected
    assert len(eph.position_km) == n_expected
    assert len(eph.velocity_mps) == n_expected
    assert len(eph.synodic_position) == n_expected


MULTILAYER_DURATION_SEC = 180 * 86400.0  # 13 圈（n_rev=13），5 段 + 3 合并层


@pytest.fixture(scope="module")
def halo_result_multilayer():
    """180 天 Halo segmented 设计：3 层合并多层回归。

    3 层合并（5 段 → 3 → 2 → 1）须全程收敛，覆盖合并层 2/3 的深层路径；
    锚定约束会让设计停在第 1 层。
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
    """多层合并收敛回归：180 天设计成功、星历逐点对齐、会合系保形。

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
