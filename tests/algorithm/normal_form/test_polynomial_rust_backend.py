"""数值多项式核 Rust 后端与 Python 参照路径的等价性对照（#464）。

测公开行为：同输入下幂次集合、系数值（容差内）、零元与阈值剔除结果一致。
不测 Rust 内部 HashMap 布局。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.normal_form.polynomial import (
    keys_by_order,
    poly_poisson,
    poly_simplify,
    polylist_simplify,
    trim_degree,
)
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = pytest.mark.theory


def _assert_poly_close(
    actual: dict[tuple[int, ...], object],
    expected: dict[tuple[int, ...], object],
    *,
    atol: float = 1e-12,
) -> None:
    """幂次集合一致，系数按类型逐项对照。"""
    assert set(actual.keys()) == set(expected.keys()), (
        f"幂次集合不一致：actual={set(actual.keys())} expected={set(expected.keys())}"
    )
    for key in expected:
        a = actual[key]
        e = expected[key]
        if isinstance(e, np.ndarray) or isinstance(a, np.ndarray):
            assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=atol)
        else:
            assert_allclose(complex(a), complex(e), rtol=0.0, atol=atol)


# ---------------------------------------------------------------------------
# poly_poisson
# ---------------------------------------------------------------------------


class TestPolyPoissonRust:
    """标量 / 时间序列 Poisson 括号：Rust 与 Python 参照一致。"""

    def test_scalar_real_matches_python(self) -> None:
        """标量实系数：``{q1 p1, q2 p2}`` 等基本项。"""
        f = {(1, 0, 0, 1, 0, 0): 2.0}  # 2·q1·p1
        g = {(0, 1, 0, 0, 1, 0): 3.0}  # 3·q2·p2
        # 不同共轭对上无交叉 → 泊松括号为 0
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        default = poly_poisson(f, g)
        _assert_poly_close(rs, py)
        _assert_poly_close(default, py)

    def test_scalar_same_pair_nonzero(self) -> None:
        """同对 (q_k, p_k) 上的非零泊松括号。"""
        # f = q1^2, g = p1^2 → {f,g} = 4 q1 p1
        f = {(2, 0, 0, 0, 0, 0): 1.0}
        g = {(0, 0, 0, 2, 0, 0): 1.0}
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        _assert_poly_close(rs, py)
        assert (1, 0, 0, 1, 0, 0) in rs
        assert_allclose(float(rs[(1, 0, 0, 1, 0, 0)]), 4.0)  # type: ignore[arg-type]

    def test_scalar_complex_matches_python(self) -> None:
        """标量复系数（中心流形复坐标路径）。"""
        f = {(1, 0, 0, 1, 0, 0): 1.0 + 2.0j}
        g = {(0, 1, 0, 0, 0, 0): 0.5 - 0.25j}
        # f = c1·q1·p1, g = c2·q2 → 无同对交叉，零
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        _assert_poly_close(rs, py)

        # 非零：f = λ·q1·p1, g = q1^2 → {f,g} = λ·(-2)·q1^2? 见公式
        # {q1 p1, q1^2} = Σ_k (a_k b_{k+3} - b_k a_{k+3}) with a=(1,0,0,1,0,0), b=(2,0,0,0,0,0)
        # k=0: a0*b3 - b0*a3 = 1*0 - 2*1 = -2 → new_pow = (1+2-1, 0,0, 1+0-1,0,0)=(2,0,0,0,0,0)
        f2 = {(1, 0, 0, 1, 0, 0): 1j}
        g2 = {(2, 0, 0, 0, 0, 0): 1.0}
        py2 = poly_poisson(f2, g2, backend="python")
        rs2 = poly_poisson(f2, g2, backend="rust")
        _assert_poly_close(rs2, py2)

    def test_series_real_matches_python(self) -> None:
        """实值时间序列系数。"""
        t = np.linspace(0.0, 1.0, 8)
        f = {(1, 0, 0, 1, 0, 0): 1.5 * np.ones_like(t)}
        g = {(2, 0, 0, 0, 0, 0): np.sin(t) + 0.1}
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        _assert_poly_close(rs, py, atol=1e-14)

    def test_series_complex_matches_python(self) -> None:
        """复值时间序列系数（同调方程路径）。"""
        t = np.linspace(0.0, 2.0, 16)
        ones = np.ones_like(t)
        h2c = {
            (1, 0, 0, 1, 0, 0): 0.5 * ones,
            (0, 1, 0, 0, 1, 0): 1j * 1.2 * ones,
        }
        w = {(1, 2, 0, 0, 0, 0): (0.1 + 0.05j) * ones}
        py = poly_poisson(h2c, w, backend="python")
        rs = poly_poisson(h2c, w, backend="rust")
        _assert_poly_close(rs, py, atol=1e-14)

    def test_empty_and_zero_convention(self) -> None:
        """零多项式约定：结果空时退回 ``(0,)*6`` 零系数。"""
        f = {(0, 0, 0, 0, 0, 0): 0.0}
        g = {(1, 0, 0, 0, 0, 0): 1.0}
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        _assert_poly_close(rs, py)
        assert (0, 0, 0, 0, 0, 0) in rs

    def test_multi_term_merge(self) -> None:
        """多项输入时同幂次系数正确合并。"""
        f = {
            (1, 0, 0, 0, 0, 0): 1.0,  # q1
            (0, 1, 0, 0, 0, 0): 2.0,  # q2
        }
        g = {
            (0, 0, 0, 1, 0, 0): 3.0,  # p1
            (0, 0, 0, 0, 1, 0): 4.0,  # p2
        }
        # {q1, p1} = 1 → const 3; {q2, p2} = 1 → const 8; total const 11
        py = poly_poisson(f, g, backend="python")
        rs = poly_poisson(f, g, backend="rust")
        _assert_poly_close(rs, py)
        assert_allclose(float(rs[(0, 0, 0, 0, 0, 0)]), 11.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# poly_simplify / polylist_simplify
# ---------------------------------------------------------------------------


class TestPolySimplifyRust:
    """阈值与合并语义与 Python 参照一致。"""

    def test_scalar_merge_and_threshold(self) -> None:
        # 近零项剔除 + 非零项保留
        poly = {
            (1, 0, 0, 0, 0, 0): 3.0,
            (0, 1, 0, 0, 0, 0): 1e-15,
            (0, 0, 1, 0, 0, 0): -2.5,
        }
        py = poly_simplify(poly, eps=1e-12, backend="python")
        rs = poly_simplify(poly, eps=1e-12, backend="rust")
        _assert_poly_close(rs, py)
        assert (0, 1, 0, 0, 0, 0) not in rs

    def test_scalar_complex_keeps_nonzero(self) -> None:
        """复标量非近零项保留；合并语义与 Python 一致。"""
        poly = {
            (1, 0, 0, 0, 0, 0): 0.5 + 0.5j,
            (0, 1, 0, 0, 0, 0): 1.0 - 2.0j,
        }
        py = poly_simplify(poly, eps=1e-12, backend="python")
        rs = poly_simplify(poly, eps=1e-12, backend="rust")
        _assert_poly_close(rs, py)

    def test_scalar_complex_threshold(self) -> None:
        """复标量按模长阈值：近零剔除，两侧一致。"""
        poly = {
            (1, 0, 0, 0, 0, 0): 1e-14 + 1e-14j,  # |z|≈1.4e-14 ≤ 1e-12
            (0, 1, 0, 0, 0, 0): 0.5 + 0.5j,
        }
        py = poly_simplify(poly, eps=1e-12, backend="python")
        rs = poly_simplify(poly, eps=1e-12, backend="rust")
        _assert_poly_close(rs, py)
        assert (1, 0, 0, 0, 0, 0) not in rs
        assert (0, 1, 0, 0, 0, 0) in rs

    def test_all_near_zero_returns_zero_poly(self) -> None:
        poly = {(1, 0, 0, 0, 0, 0): 1e-20, (0, 1, 0, 0, 0, 0): 0.0}
        py = poly_simplify(poly, eps=1e-12, backend="python")
        rs = poly_simplify(poly, eps=1e-12, backend="rust")
        _assert_poly_close(rs, py)
        assert set(rs.keys()) == {(0, 0, 0, 0, 0, 0)}

    def test_empty_input(self) -> None:
        py = poly_simplify({}, backend="python")
        rs = poly_simplify({}, backend="rust")
        _assert_poly_close(rs, py)


class TestPolylistSimplifyRust:
    """时间序列 mean-abs 阈值语义。"""

    def test_mean_abs_threshold_boundary(self) -> None:
        # mean abs = 1e-15 恰在默认 eps 边界（<= eps 剔除）
        poly = {
            (1, 0, 0, 0, 0, 0): np.array([1e-15, 1e-15, 1e-15]),
            (0, 1, 0, 0, 0, 0): np.array([1.0, 2.0, 3.0]),
            (0, 0, 0, 0, 0, 0): np.array([0.0, 0.0, 0.0]),
        }
        py = polylist_simplify(poly, eps=1e-15, backend="python")
        rs = polylist_simplify(poly, eps=1e-15, backend="rust")
        _assert_poly_close(rs, py)
        assert (1, 0, 0, 0, 0, 0) not in rs
        assert (0, 1, 0, 0, 0, 0) in rs

    def test_zero_epsilon_keeps_nonzero(self) -> None:
        powers = {
            (0, 0, 0, 0, 0, 0): np.array([1.0]),
            (0, 1, 0, 0, 0, 0): np.array([0.0]),
            (1, 0, 0, 0, 0, 0): np.array([2.0]),
        }
        py = polylist_simplify(powers, eps=0.0, backend="python")
        rs = polylist_simplify(powers, eps=0.0, backend="rust")
        _assert_poly_close(rs, py)
        assert set(rs.keys()) == {(0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)}

    def test_complex_series(self) -> None:
        poly = {
            (1, 0, 0, 1, 0, 0): np.array([1.0 + 1.0j, 2.0 + 0.5j]),
            (0, 1, 0, 0, 0, 0): np.array([1e-20 + 0j, 1e-20 + 0j]),
        }
        py = polylist_simplify(poly, eps=1e-15, backend="python")
        rs = polylist_simplify(poly, eps=1e-15, backend="rust")
        _assert_poly_close(rs, py)

    def test_merge_same_key(self) -> None:
        """同幂次在阈值通过后累加（构造时 dict 键唯一，用两次调用合并语义
        无法直接注入；此处验证保留项的数值本身）。"""
        poly = {
            (1, 0, 0, 0, 0, 0): np.array([1.0, 2.0]),
            (0, 1, 0, 0, 0, 0): np.array([3.0, 4.0]),
        }
        py = polylist_simplify(poly, backend="python")
        rs = polylist_simplify(poly, backend="rust")
        _assert_poly_close(rs, py)

    def test_all_dropped_returns_zero_series(self) -> None:
        poly = {(1, 0, 0, 0, 0, 0): np.array([0.0, 0.0])}
        py = polylist_simplify(poly, backend="python")
        rs = polylist_simplify(poly, backend="rust")
        _assert_poly_close(rs, py)
        assert set(rs.keys()) == {(0, 0, 0, 0, 0, 0)}


# ---------------------------------------------------------------------------
# 幂次工具
# ---------------------------------------------------------------------------


class TestPowerUtilsRust:
    def test_keys_by_order(self) -> None:
        poly = {
            (1, 0, 0, 0, 0, 0): 1.0,
            (2, 1, 0, 0, 0, 0): 2.0,
            (0, 0, 0, 1, 0, 0): 3.0,
            (1, 1, 1, 0, 0, 0): 4.0,
        }
        py = keys_by_order(poly, backend="python")
        rs = keys_by_order(poly, backend="rust")
        assert py == rs
        assert py[1] == [(0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0)]
        assert py[3] == [(1, 1, 1, 0, 0, 0), (2, 1, 0, 0, 0, 0)]

    def test_trim_degree(self) -> None:
        poly = {
            (1, 0, 0, 0, 0, 0): 1.0,
            (2, 1, 0, 0, 0, 0): 2.0,
            (0, 0, 0, 0, 0, 0): 0.5,
        }
        py = trim_degree(poly, 2, backend="python")
        rs = trim_degree(poly, 2, backend="rust")
        _assert_poly_close(rs, py)
        assert (2, 1, 0, 0, 0, 0) not in rs

    def test_trim_degree_empty_becomes_zero(self) -> None:
        poly = {(3, 0, 0, 0, 0, 0): 1.0}
        py = trim_degree(poly, 1, backend="python")
        rs = trim_degree(poly, 1, backend="rust")
        _assert_poly_close(rs, py)
        assert set(rs.keys()) == {(0, 0, 0, 0, 0, 0)}

    def test_trim_degree_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="max_degree"):
            trim_degree({(0, 0, 0, 0, 0, 0): 1.0}, -1, backend="rust")
        with pytest.raises(ValueError, match="max_degree"):
            trim_degree({(0, 0, 0, 0, 0, 0): 1.0}, -1, backend="python")


# ---------------------------------------------------------------------------
# 失败策略：无静默回退
# ---------------------------------------------------------------------------


def test_default_rust_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 Rust 路径扩展错误直接上抛，绝不偷偷改跑 Python。"""
    import e2m2e.integrators as integrators

    def unavailable(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RustExtensionUnavailableError("模拟 Rust 扩展缺失")

    monkeypatch.setattr(integrators, "poly_poisson_py", unavailable)
    with pytest.raises(RustExtensionUnavailableError, match="模拟 Rust 扩展缺失"):
        poly_poisson(
            {(1, 0, 0, 0, 0, 0): 1.0},
            {(0, 0, 0, 1, 0, 0): 1.0},
            backend="rust",
        )


def test_invalid_backend_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        poly_poisson(
            {(1, 0, 0, 0, 0, 0): 1.0},
            {(0, 0, 0, 1, 0, 0): 1.0},
            backend="auto",  # type: ignore[arg-type]
        )
