"""segmented 分段打靶拼接的星历连续性测试。

回归防护：``correction_method="segmented"`` 逐段积分填 et_grid 时，每段的
``seg_t0``（patch point 时刻，非整数小时）与 ``t_eval_seg[0]``（et_grid 整数
小时点）不严格相等。``propagate_compiled`` 曾假设 ``t_eval[0]==t0``、初始化
``eval_idx=1`` 跳过首点，导致 ``t_eval_seg[0] > seg_t0`` 时首个输出点状态错置
为初值，相邻点 J2000 位置差塌缩到 ~1e-3 km（速度正常，位置却停滞），被 synodic
坐标旋转放大成可见跳变。本测试独立校验星历 J2000 相邻点无此类停滞。

属 tests/orbit_design 三层分层中的 L3（scenarios，端到端）：覆盖 design_orbit
的 segmented 分段打靶全链路集成行为。
"""

import numpy as np
import pytest

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest

pytestmark = [pytest.mark.slow, pytest.mark.spice, pytest.mark.l3]

# 30 天 Halo：main_design 默认参数，复现逐段积分场景
DURATION_SEC = 30 * 86400.0
AMPLITUDE_KM = 30000.0

PERTURBATION = {
    "sun_body": 1,
    "planets": 0,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 1,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 0,
    "coupling": 0,
}

# 相邻 et_grid 点（1 小时间隔）的 J2000 位置差下限：Halo 远月点速度最低约
# 0.1 km/s，1 小时移动 ≥ 300 km。停滞 bug 表现为差值塌缩到 ~1e-3 km，
# 远低于此下限。
MIN_HOURLY_DRIFT_KM = 10.0


@pytest.fixture(scope="module")
def segmented_result():
    """跑一次 30 天 Halo segmented 设计，模块内复用（耗时 ~4 分钟）。"""
    return design_orbit(
        DesignOrbitRequest(
            orbit_type="HALO",
            collinear_point=2,
            amplitude=AMPLITUDE_KM,
            phase=0.0,
            duration=DURATION_SEC,
            output_step=3600.0,
            perturbation=PERTURBATION,
            correction_method="segmented",
        )
    )


def test_no_j2000_stall_points(segmented_result):
    """相邻 et_grid 点的 J2000 位置差不得出现停滞。

    回归 bug：``propagate_compiled`` 的 ``eval_idx`` 初始化假设
    ``t_eval[0]==t0``，逐段积分 ``t_eval_seg[0] > seg_t0`` 时首个输出点
    位置错置为初值，与下一个点几乎相同（差 ~1e-3 km），但速度正常。
    """
    eph = segmented_result.ephemeris
    pos = np.asarray(eph.position_km)
    drift = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    n_stalled = int(np.sum(drift < MIN_HOURLY_DRIFT_KM))
    assert n_stalled == 0, (
        f"发现 {n_stalled} 个 J2000 停滞点（1 小时位移 < {MIN_HOURLY_DRIFT_KM} km），"
        f"最小漂移 {drift.min():.4f} km。疑似 propagate_compiled 的 t_eval[0]≠t0 回归。"
    )


def test_patch_point_spacing_sane(segmented_result):
    """烟雾测试：相邻 patch point 间距落在 Halo 振幅量级，防塌缩/爆炸。

    本测试只校验打靶修正后相邻 patch point 的位置差落在合理量级（数千到
    数十万 km，对应一个 Halo 振幅量级），作为退化/发散烟雾测试；它**不**
    做重积分、也**不**校验"段末与重积分亚百米级自洽"。真正的重积分自洽
    验证由 ``test_no_j2000_stall_points``（星历相邻点 J2000 漂移）承担。
    """
    conv = segmented_result.correction
    state_patch = np.asarray(conv.state_patch)
    # 检查前若干段（避免全段重积分耗时过长）
    max_diff = 0.0
    for i in range(min(6, len(state_patch) - 1)):
        seg = state_patch[i + 1] - state_patch[i]
        max_diff = max(max_diff, float(np.linalg.norm(seg[:3])))
    # 段间位置差应在一个 Halo 振幅量级（~1e3-1e6 km），非异常塌缩或爆炸
    assert 1e3 < max_diff < 1e6, f"段间位置差异常: {max_diff:.1f} km"
