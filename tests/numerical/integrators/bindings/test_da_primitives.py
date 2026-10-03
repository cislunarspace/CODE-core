"""dace-rs 微分代数原语的行为测试（issue #784）。

覆盖四类事实：sin(1+xy) 展开系数与解析值一致；二变量映射求逆往返残差；
bound 包围覆盖随机采样点；CompiledDa 与直接求值一致。DA 上下文是进程级
全局状态，每个用例开头各自 da_init_py（同时重置线程截断阶），不跨用例持对象。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.integrators import (
    CompiledDa,
    Da,
    da_init_py,
    da_initialized_py,
    da_set_truncation_order_py,
    da_truncation_order_py,
    da_vector_invert_py,
)

pytestmark = [
    pytest.mark.theory,
    requires_native_symbols("da_init_py", "Da", "CompiledDa", "da_vector_invert_py"),
]


def test_context_init_and_truncation_order():
    """初始化后可查询状态；截断阶设置返回旧值并可还原。"""
    da_init_py(6, 2)
    assert da_initialized_py() is True
    assert da_truncation_order_py() == 6
    assert da_set_truncation_order_py(4) == 6
    assert da_truncation_order_py() == 4
    assert da_set_truncation_order_py(6) == 4
    assert da_truncation_order_py() == 6


def test_sin_expansion_coefficients():
    """sin(1+xy) 的截断 Taylor 系数与 sin/cos 在 1 处的解析值一致。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    y = Da.variable(2)
    f = (Da.constant(1.0) + x * y).sin()
    np.testing.assert_allclose(
        f.coefficient([0, 0]),
        math.sin(1),
        atol=1e-14,
        err_msg="常数项应为 sin(1)",
    )
    np.testing.assert_allclose(
        f.coefficient([1, 1]),
        math.cos(1),
        atol=1e-14,
        err_msg="xy 项系数应为 cos(1)",
    )
    np.testing.assert_allclose(
        f.coefficient([2, 2]),
        -math.sin(1) / 2,
        atol=1e-14,
        err_msg="(xy)^2 项系数应为 -sin(1)/2",
    )
    np.testing.assert_allclose(
        f.coefficient([3, 3]),
        -math.cos(1) / 6,
        atol=1e-14,
        err_msg="(xy)^3 项系数应为 -cos(1)/6",
    )
    # [1,0] 不在展开里；(xy)^4 是 8 阶项，被 6 阶截断丢弃。
    assert abs(f.coefficient([1, 0])) < 1e-15
    assert abs(f.coefficient([4, 4])) < 1e-15


def test_scalar_and_da_arithmetic():
    """标量与 Da 双向四则运算的系数：q = 2(1+x)^2 - x/2/... 逐项核对。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    p = 1.0 + x
    q = 2.0 * (p * p) - x / Da.constant(2.0)
    np.testing.assert_allclose(q.coefficient([0, 0]), 2.0, atol=1e-14)
    np.testing.assert_allclose(q.coefficient([1, 0]), 3.5, atol=1e-14)
    np.testing.assert_allclose(q.coefficient([2, 0]), 2.0, atol=1e-14)


def test_invert_roundtrip():
    """二变量多项式映射求逆：先求值再逆映射应回到原采样点。"""
    da_init_py(8, 2)
    x = Da.variable(1)
    y = Da.variable(2)
    f1 = x + 0.3 * (x * x) + 0.2 * (x * y)
    f2 = y + 0.15 * x + 0.1 * (y * y)
    g = da_vector_invert_py([f1, f2])
    rng = np.random.default_rng(20261003)
    for pt in rng.uniform(-0.12, 0.12, size=(20, 2)):
        fv = [fi.eval(list(pt)) for fi in (f1, f2)]
        gv = [gi.eval(fv) for gi in g]
        np.testing.assert_allclose(
            gv,
            pt,
            atol=1e-7,
            err_msg=f"求逆往返在 {pt} 处未回到原点",
        )


def test_bound_contains_samples():
    """bound 给出的包围覆盖 [-1,1]^2 上的随机采样值。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    y = Da.variable(2)
    p = Da.constant(0.5) - 0.3 * x + 0.2 * y + 0.1 * (x * y) + 0.05 * (y * y)
    lo, hi = p.bound()
    rng = np.random.default_rng(784)
    vals = np.array([p.eval(list(pt)) for pt in rng.uniform(-1, 1, size=(200, 2))])
    assert vals.min() >= lo - 1e-12
    assert vals.max() <= hi + 1e-12


def test_compiled_matches_direct():
    """CompiledDa 批量求值与逐个 Da.eval 一致。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    y = Da.variable(2)
    p1 = x * x + 0.5 * y
    p2 = 0.3 * x + y * y * y
    cd = CompiledDa.from_das([p1, p2])
    single = p1.compile()
    # ord 是求值树实际存储的最大阶（这里 y^3 为 3 阶），不是上下文截断阶。
    assert (cd.dim, cd.ord, cd.vars) == (2, 3, 2)
    rng = np.random.default_rng(7841)
    for pt in rng.uniform(-0.5, 0.5, size=(50, 2)):
        args = list(pt)
        np.testing.assert_allclose(
            cd.eval(args),
            [p1.eval(args), p2.eval(args)],
            atol=1e-13,
            err_msg=f"编译求值与直接求值在 {pt} 处不一致",
        )
        np.testing.assert_allclose(
            single.eval(args),
            p1.eval(args),
            atol=1e-13,
        )


def test_smoke_chain():
    """issue 验收链路：初始化 → 构造变量 → 求逆 → 包围 → 往返求值。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    y = Da.variable(2)
    g = da_vector_invert_py([x + 0.25 * (y * y), y + 0.5 * (x * x)])
    for gi in g:
        lo, hi = gi.bound()
        assert lo <= hi
    pt = [0.05, 0.05]
    fv = [
        (x + 0.25 * (y * y)).eval(pt),
        (y + 0.5 * (x * x)).eval(pt),
    ]
    gv = [gi.eval(fv) for gi in g]
    np.testing.assert_allclose(gv, pt, atol=1e-6)


def test_error_paths():
    """错误映射：除零 641、log 非正 647、奇异求逆 642、init 911、p<2。"""
    da_init_py(6, 2)
    x = Da.variable(1)
    with pytest.raises(RuntimeError, match="DA 运算失败.*641"):
        Da.constant(1.0) / Da.constant(0.0)
    with pytest.raises(RuntimeError, match="647"):
        Da.constant(-1.0).log()
    with pytest.raises(RuntimeError, match="642"):
        da_vector_invert_py([x, x])
    with pytest.raises(ValueError, match="DA 初始化失败"):
        da_init_py(64, 64)
    with pytest.raises(ValueError, match="p >= 2"):
        Da.constant(0.5).norm_power(1)
