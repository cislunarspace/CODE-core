"""``to_normalized`` / ``from_normalized`` 单测。

覆盖：
- SI ↔ 归一化的 round-trip 在浮点精度内闭合；
- 位置与速度分别按 LU / VU 缩放；
- 支持 ``(6,)`` 单状态与 ``(n, 6)`` 批量状态；
- 输入形状错误抛 ``ValueError``。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.normal_form import NormalFormContext
from e2m2e.algorithm.normal_form.constants import LU_KM, MU, TU_S, VU_KMS
from e2m2e.algorithm.normal_form.units import from_normalized, to_normalized


@pytest.fixture
def ctx(earth_moon_system) -> NormalFormContext:
    """L1 上下文，使用 qiao 默认 LU/TU。"""
    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=2451545.0,
        order=2,
    )


# ---------------------------------------------------------------------------
# 缩放关系
# ---------------------------------------------------------------------------


def test_to_normalized_divides_position_and_velocity_by_LU_VU(ctx):
    """``to_normalized`` 位置除以 LU、速度除以 VU。"""
    state_si = np.array([384747.981, 0.0, 0.0, 1.02416, 0.0, 0.0])
    state_norm = to_normalized(state_si, ctx)

    assert state_norm[0] == pytest.approx(1.0)  # x_km / LU = 1
    assert state_norm[3] == pytest.approx(1.02416 / VU_KMS)


def test_from_normalized_multiplies_by_LU_VU(ctx):
    """``from_normalized`` 位置乘以 LU、速度乘以 VU。"""
    state_norm = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    state_si = from_normalized(state_norm, ctx)

    assert state_si[0] == pytest.approx(LU_KM)
    assert state_si[3] == pytest.approx(VU_KMS)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_si_to_norm_to_si_round_trip(ctx):
    """SI → 归一化 → SI 应回到原值。"""
    state_si = np.array([12345.6, -9876.5, 100.0, 0.123, -0.456, 0.789])
    recovered = from_normalized(to_normalized(state_si, ctx), ctx)
    np.testing.assert_allclose(recovered, state_si, rtol=0.0, atol=1e-9)


def test_norm_to_si_to_norm_round_trip(ctx):
    """归一化 → SI → 归一化 应回到原值。"""
    state_norm = np.array([0.987, -0.123, 0.456, -0.321, 0.654, -0.987])
    recovered = to_normalized(from_normalized(state_norm, ctx), ctx)
    np.testing.assert_allclose(recovered, state_norm, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    "state_si",
    [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 原点
        np.array([LU_KM, 0.0, 0.0, 0.0, 0.0, 0.0]),  # x = 1 LU
        np.array([-LU_KM, LU_KM, LU_KM, VU_KMS, -VU_KMS, VU_KMS]),  # 全分量
        np.array([384400.0, 0.0, 0.0, 0.0, 1.022, 0.0]),  # 类 LEO/月球轨道量级
    ],
)
def test_round_trip_various_states(ctx, state_si):
    """多组代表性 SI 状态 round-trip 闭合。"""
    recovered = from_normalized(to_normalized(state_si, ctx), ctx)
    np.testing.assert_allclose(recovered, state_si, rtol=0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 批量接口
# ---------------------------------------------------------------------------


def test_batch_state_round_trip(ctx):
    """``(n, 6)`` 批量状态同样 round-trip。"""
    rng = np.random.default_rng(20260107)
    batch_si = rng.normal(scale=[1e5, 1e5, 1e5, 1.0, 1.0, 1.0], size=(17, 6))
    recovered = from_normalized(to_normalized(batch_si, ctx), ctx)
    np.testing.assert_allclose(recovered, batch_si, rtol=0.0, atol=1e-9)


def test_batch_norm_round_trip(ctx):
    """``(n, 6)`` 归一化批量状态同样 round-trip。"""
    rng = np.random.default_rng(20260107)
    batch_norm = rng.normal(scale=0.5, size=(13, 6))
    recovered = to_normalized(from_normalized(batch_norm, ctx), ctx)
    np.testing.assert_allclose(recovered, batch_norm, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def test_wrong_last_dim_raises_to_normalized(ctx):
    """最后一维不是 6 时抛 ``ValueError``。"""
    with pytest.raises(ValueError, match="6 个分量"):
        to_normalized(np.array([1.0, 2.0, 3.0]), ctx)
    with pytest.raises(ValueError, match="6 个分量"):
        to_normalized(np.zeros((3, 5)), ctx)


def test_wrong_last_dim_raises_from_normalized(ctx):
    """``from_normalized`` 同样校验最后一维。"""
    with pytest.raises(ValueError, match="6 个分量"):
        from_normalized(np.array([1.0, 2.0, 3.0]), ctx)


# ---------------------------------------------------------------------------
# 上下文驱动的 LU/TU/VU
# ---------------------------------------------------------------------------


def test_custom_LU_TU_change_scaling(earth_moon_system):
    """覆盖 LU/TU 后缩放比例同步更新。"""
    ctx_default = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=2451545.0,
        order=2,
    )
    ctx_custom = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=2451545.0,
        order=2,
        LU=1000.0,
        TU=100.0,
    )
    state_si = np.array([12345.6, 0.0, 0.0, 0.5, 0.0, 0.0])

    norm_default = to_normalized(state_si, ctx_default)
    norm_custom = to_normalized(state_si, ctx_custom)

    assert norm_custom[0] == pytest.approx(state_si[0] / 1000.0)
    assert norm_custom[3] == pytest.approx(state_si[3] / (1000.0 / 100.0))
    # 自定义 LU (1000) 远小于默认 LU (~384748)，归一化后的位置分量应更大
    assert abs(norm_custom[0]) > abs(norm_default[0])


def test_qiao_constants_used_in_round_trip():
    """不依赖 ``earth_moon_system``，直接验证 qiao LU/TU/VU 与 MU 数值。"""
    # 这些断言把 qiao 常量与 NormalFormContext 默认值钉住，
    # 防止后续误改常量而 round-trip 测试蒙混过关。
    assert pytest.approx(384747.981) == LU_KM
    assert pytest.approx(375699.843898365) == TU_S
    assert pytest.approx(LU_KM / TU_S) == VU_KMS
    assert pytest.approx(1.215058560962404e-2) == MU
