"""``NormalFormContext`` 构造与字段正确性测试。

覆盖：L1–L5 均能成功构造；关键字段（LU/TU/VU、mu/mu_e/mu_m/mu_s、JD0、
平动点位置、γ、基础频率、中心流形频率、特征指数、历元、阶数）符合
qiao 约定与物理直觉。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithms.normal_form import NormalFormContext, NormalFormResult
from e2m2e.algorithms.normal_form.constants import (
    BASE_FREQUENCIES,
    JD0_J2000,
    LU_KM,
    MU,
    MU_E,
    MU_M,
    MU_S,
    TU_S,
)
from e2m2e.core import LibrationPoint

# ---------------------------------------------------------------------------
# 构造烟测：L1–L5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point",
    [LibrationPoint.L1, LibrationPoint.L2, LibrationPoint.L3, LibrationPoint.L4, LibrationPoint.L5],
)
def test_context_constructs_for_all_libration_points(earth_moon_system, point):
    """L1–L5 都能成功构造 ``NormalFormContext``。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=point,
        epoch=JD0_J2000,
        order=4,
    )
    assert ctx.libration_point is point
    assert ctx.order == 4
    assert ctx.epoch == pytest.approx(JD0_J2000)


# ---------------------------------------------------------------------------
# 字段正确性
# ---------------------------------------------------------------------------


def test_qiao_constants_are_carried(earth_moon_system):
    """默认构造时 LU/TU/mu/mu_e/mu_m/mu_s/JD0 与 qiao 一致。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
    )
    assert pytest.approx(LU_KM) == ctx.LU
    assert pytest.approx(TU_S) == ctx.TU
    assert pytest.approx(LU_KM / TU_S) == ctx.VU
    assert ctx.mu == pytest.approx(MU)
    assert ctx.mu_e == pytest.approx(MU_E)
    assert ctx.mu_m == pytest.approx(MU_M)
    assert ctx.mu_s == pytest.approx(MU_S)
    assert ctx.jd0 == pytest.approx(JD0_J2000)


def test_base_frequencies_match_qiao(earth_moon_system):
    """基础频率四个分量与 qiao Global_File.py 完全一致。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=JD0_J2000,
        order=3,
    )
    np.testing.assert_allclose(ctx.base_frequencies, BASE_FREQUENCIES, rtol=1e-12)


def test_central_frequencies_vary_with_point(earth_moon_system):
    """L1/L2/L3 的中心流形频率各不相同；L4/L5 也给出数值。"""
    ctx_l1 = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
    )
    ctx_l2 = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=JD0_J2000,
        order=2,
    )
    ctx_l4 = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L4,
        epoch=JD0_J2000,
        order=2,
    )

    assert ctx_l1.central_frequencies == pytest.approx((2.33774371420711, 2.27427342163957))
    assert ctx_l2.central_frequencies == pytest.approx((1.86464967793235, 1.79093984309149))
    assert ctx_l4.central_frequencies == pytest.approx((0.30259440630339, 1.00408270193080))
    # 各点中心流形频率应不同（互不相等）
    assert ctx_l1.central_frequencies != ctx_l2.central_frequencies


def test_characteristic_exponent_per_point(earth_moon_system):
    """特征指数 λ 与 qiao 表格一致；L1 最大、L3 最小。"""
    lambdas = {}
    for point in LibrationPoint:
        ctx = NormalFormContext(
            system=earth_moon_system,
            libration_point=point,
            epoch=JD0_J2000,
            order=2,
        )
        lambdas[point] = ctx.characteristic_exponent

    assert lambdas[LibrationPoint.L1] == pytest.approx(2.93924602471)
    assert lambdas[LibrationPoint.L2] == pytest.approx(2.16475967850)
    assert lambdas[LibrationPoint.L3] == pytest.approx(0.17970712561)
    assert lambdas[LibrationPoint.L4] == pytest.approx(0.01416193941)
    assert lambdas[LibrationPoint.L5] == pytest.approx(0.01381362875)
    # L1 是最不稳定共线点
    assert lambdas[LibrationPoint.L1] > lambdas[LibrationPoint.L2] > lambdas[LibrationPoint.L3]


# ---------------------------------------------------------------------------
# 平动点几何
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected_x"),
    [
        (LibrationPoint.L1, 1.0 - 0.150934288618019),
        (LibrationPoint.L2, 1.0 + 0.167832751054508),
        (LibrationPoint.L3, -0.992912060200654),
    ],
)
def test_collinear_libration_positions(earth_moon_system, point, expected_x):
    """共线点位置用 qiao γ 值给出，符合 ``(1 ± γ, 0, 0)`` / ``(-γ, 0, 0)``。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=point,
        epoch=JD0_J2000,
        order=2,
    )
    np.testing.assert_allclose(ctx.libration_position, [expected_x, 0.0, 0.0], atol=1e-12)
    assert ctx.gamma is not None


@pytest.mark.parametrize(
    ("point", "expected_y"),
    [
        (LibrationPoint.L4, math.sqrt(3.0) / 2.0),
        (LibrationPoint.L5, -math.sqrt(3.0) / 2.0),
    ],
)
def test_triangular_libration_positions(earth_moon_system, point, expected_y):
    """三角点位置为等边三角形顶点 ``(1/2 - mu, ±√3/2, 0)``，γ 为 None。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=point,
        epoch=JD0_J2000,
        order=2,
    )
    expected_x = 0.5 - MU
    np.testing.assert_allclose(ctx.libration_position, [expected_x, expected_y, 0.0], atol=1e-12)
    assert ctx.gamma is None


# ---------------------------------------------------------------------------
# 历元与构造变体
# ---------------------------------------------------------------------------


def test_epoch_accepts_datetime(earth_moon_system):
    """``epoch`` 接受 ``datetime``，自动转儒略日。"""
    from datetime import datetime, timezone

    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=dt,
        order=2,
    )
    assert ctx.epoch == pytest.approx(JD0_J2000)


def test_invalid_order_raises(earth_moon_system):
    """``order <= 0`` 或非整数应抛 ``ValueError``。"""
    with pytest.raises(ValueError, match="order"):
        NormalFormContext(
            system=earth_moon_system,
            libration_point=LibrationPoint.L1,
            epoch=JD0_J2000,
            order=0,
        )
    with pytest.raises(ValueError, match="order"):
        NormalFormContext(
            system=earth_moon_system,
            libration_point=LibrationPoint.L1,
            epoch=JD0_J2000,
            order=-3,
        )
    with pytest.raises(ValueError, match="order"):
        NormalFormContext(
            system=earth_moon_system,
            libration_point=LibrationPoint.L1,
            epoch=JD0_J2000,
            order=2.5,  # type: ignore[arg-type]
        )


def test_explicit_overrides_take_effect(earth_moon_system):
    """显式传入 ``LU``/``TU``/``mu`` 时覆盖默认值。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
        LU=1.0,
        TU=1.0,
        mu=0.0123,
        mu_s=999.0,
    )
    assert ctx.LU == 1.0
    assert ctx.TU == 1.0
    assert ctx.VU == 1.0  # LU/TU = 1
    assert ctx.mu == pytest.approx(0.0123)
    assert ctx.mu_s == 999.0


def test_context_system_mu_preferred_over_default(earth_moon_system):
    """``System.mu``（若存在）应覆盖 qiao 默认 mu。"""
    # earth_moon_system 默认 mu=1.215058560962404e-2，与 qiao MU 一致；
    # 显式重设 system.mu 后构造 context，应拿到新 mu。
    earth_moon_system.mu = 0.0123
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
    )
    assert ctx.mu == pytest.approx(0.0123)


# ---------------------------------------------------------------------------
# 时间转换
# ---------------------------------------------------------------------------


def test_seconds_to_tu_round_trip(earth_moon_system):
    """秒 ↔ TU 互为逆运算。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
    )
    t_s = 12345.6789
    assert ctx.tu_to_seconds(ctx.seconds_to_tu(t_s)) == pytest.approx(t_s)
    t_tu = 5.678
    assert ctx.seconds_to_tu(ctx.tu_to_seconds(t_tu)) == pytest.approx(t_tu)


# ---------------------------------------------------------------------------
# NormalFormResult 占位
# ---------------------------------------------------------------------------


def test_normal_form_result_is_constructible(earth_moon_system):
    """``NormalFormResult`` 至少能以默认参数构造（具体填充后续切片实现）。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=2,
    )
    res = NormalFormResult(context=ctx, order=2)
    assert res.context is ctx
    assert res.order == 2
    assert res.success is False
    assert res.metadata == {}


def test_repr_and_str_are_informative(earth_moon_system):
    """``__repr__`` / ``__str__`` 不抛异常且包含平动点名称。"""
    ctx = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=JD0_J2000,
        order=3,
    )
    assert "L2" in repr(ctx)
    assert "L2" in str(ctx)
    assert "LU=" in repr(ctx)
