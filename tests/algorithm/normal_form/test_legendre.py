"""``normal_form.legendre`` 测试。"""

from __future__ import annotations

import math

import pytest

# sympy 是 normal-form optional dep；未安装时整个文件 skip（不 error）。
pytest.importorskip("sympy")

from e2m2e.algorithm.normal_form.legendre import (
    DEFAULT_COLLINEAR_ORDER,
    DEFAULT_TRIANGULAR_ORDER,
    LegendreExpansionResult,
    expand_legendre_1_over_r,
    expand_legendre_for_body,
)

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 基本烟测
# ---------------------------------------------------------------------------


def test_expand_legendre_returns_dataclass_with_required_fields():
    """结果类型为 :class:`LegendreExpansionResult`，含必要字段。"""
    res = expand_legendre_1_over_r(3)
    assert isinstance(res, LegendreExpansionResult)
    assert res.max_degree == 3
    assert res.term_count > 0
    assert isinstance(res.polynomial, dict)
    assert res.max_degree == 3


def test_expand_legendre_rejects_invalid_order():
    """``max_degree < 1`` 报 :class:`ValueError`。"""
    with pytest.raises(ValueError, match="max_degree"):
        expand_legendre_1_over_r(0)
    with pytest.raises(ValueError, match="max_degree"):
        expand_legendre_1_over_r(-2)


def test_expand_legendre_order_1_is_simple():
    """一阶展开只有线性 q 项 + 1/r0 常数项。"""
    res = expand_legendre_1_over_r(1)
    # 一阶 Legendre 中 6 元 dict 仅出现 (0,0,0,0,0,0)、(1,0,0,...)、(0,1,0,...)、(0,0,1,...)
    keys = set(res.polynomial.keys())
    assert (0, 0, 0, 0, 0, 0) in keys
    assert len(keys) >= 4


def test_expand_legendre_term_count_grows_monotonically():
    """更高阶展开会引入更多项（物理直觉：每个角度余弦多贡献一份齐次多项式）。"""
    n1 = expand_legendre_1_over_r(3).term_count
    n2 = expand_legendre_1_over_r(4).term_count
    n3 = expand_legendre_1_over_r(5).term_count
    assert n1 < n2 < n3


def test_default_orders_have_recommended_values():
    """默认阶数与 qiao/Legendre 切片约定一致。"""
    assert DEFAULT_COLLINEAR_ORDER == 10
    assert DEFAULT_TRIANGULAR_ORDER == 8


# ---------------------------------------------------------------------------
# 符号正确性
# ---------------------------------------------------------------------------


def test_expand_legendre_does_not_depend_on_p():
    """1/r 是标量场；``p1, p2, p3`` 系数恒为 0。"""
    res = expand_legendre_1_over_r(4)
    for pow_tuple, coef in res.polynomial.items():
        p_exponents = pow_tuple[3:]
        if any(p > 0 for p in p_exponents):
            assert coef == 0, f"1/r 标量场不应含 p 依赖；幂次 {pow_tuple} 系数 = {coef}"


def test_expand_legendre_constant_term_is_inverse_r0():
    """常数项应为 ``1/r0``（无量纲距离 ``r0 = |r|`` 的 L₀ 项）。"""
    import sympy as sp

    res = expand_legendre_1_over_r(3)
    const_coef = res.polynomial[(0, 0, 0, 0, 0, 0)]
    # 把所有其他符号替换为 1 后，常数项应该 = 1 (即 sum(1/r0 = 1/1 = 1))
    sub_map = {}
    for s in const_coef.free_symbols:
        sub_map[s] = sp.Integer(1)
    value = float(const_coef.subs(sub_map))
    assert value == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 数值正确性
# ---------------------------------------------------------------------------


def _build_subs(coef, rx, ry, rz, r0_val) -> dict:
    """按 :func:`sympy` 表达式的 ``free_symbols`` 名字构造替换表。"""

    vals = {"rx": rx, "ry": ry, "rz": rz, "r0": r0_val}
    sub: dict[object, float] = {}
    for s in coef.free_symbols:
        name = getattr(s, "name", None)
        if name in vals:
            sub[s] = float(vals[name])
    return sub


@pytest.mark.parametrize("max_degree", [3, 4])
def test_legendre_reproduces_1_over_r_numerically(max_degree):
    """对一系列参考点，多项式重建必须逼近闭式 ``1/|r − q|``。

    q 在 L1 附近的 ``~0.05 LU`` 量级小偏移下，3 阶 Legendre 截断即可
    提供 ``|recon − exact| < 1e-4`` 的精度；4 阶进一步收敛。
    """
    res = expand_legendre_1_over_r(max_degree)
    rx, ry, rz = 1.0, 0.0, 0.0
    r0_val = math.sqrt(rx * rx + ry * ry + rz * rz)

    test_qs = [
        (0.05, 0.03, 0.02),
        (-0.04, 0.02, -0.01),
        (0.01, 0.0, 0.0),
    ]
    tolerance = 1e-4 if max_degree == 3 else 1e-5
    for q1, q2, q3 in test_qs:
        exact = 1.0 / math.sqrt((rx - q1) ** 2 + (ry - q2) ** 2 + (rz - q3) ** 2)
        recon = 0.0
        for pow_tuple, coef in res.polynomial.items():
            n1, n2, n3 = pow_tuple[:3]
            sub = _build_subs(coef, rx, ry, rz, r0_val)
            recon += float(coef.subs(sub)) * (q1**n1) * (q2**n2) * (q3**n3)
        assert recon == pytest.approx(exact, abs=tolerance), (
            f"order={max_degree} q=({q1},{q2},{q3}) exact={exact} recon={recon}"
        )


# ---------------------------------------------------------------------------
# expand_legendre_for_body
# ---------------------------------------------------------------------------


def test_expand_legendre_for_body_negates_and_scales_coefficients():
    """``expand_legendre_for_body`` 把每个系数乘 ``-mu``，但不改幂次。"""
    res = expand_legendre_1_over_r(3)
    mu = 0.012
    body = expand_legendre_for_body(res, mu=mu)
    assert set(body.keys()) == set(res.polynomial.keys())
    for pow_tuple in res.polynomial:
        if res.polynomial[pow_tuple] == 0:
            continue
        # 注意：Legendre 系数可能仍有 (rx, ry, rz, r0) 自由符号；
        # 此处对比两个 dict 在同一符号替换下的差值。
        sample = res.polynomial[pow_tuple]
        subs = _build_subs(sample, 1.0, 0.0, 0.0, 1.0)
        original_val = float(sample.subs(subs))
        body_val = float(body[pow_tuple].subs(subs))
        assert body_val == pytest.approx(-mu * original_val, abs=1e-12)
