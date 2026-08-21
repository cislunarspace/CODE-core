"""值函数网格高阶梯度查询接口测试（#499）。

契约见 ADR 0032 决策 4：张量积三次插值、梯度为插值函数解析导数、
时间插值必选（至少线性）、维度无关、越界抛明确异常。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.levelset import (
    ValueFunctionQueryError,
    value_function_gradient,
)

pytestmark = [pytest.mark.theory]


def _grid(
    bounds: tuple[tuple[float, float], ...], shape: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    return tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, shape, strict=True))


class TestPolynomialExactness:
    """三次插值对不超过三次的多项式应精确（机器精度量级）。"""

    def test_quadratic_2d_gradient_exact(self) -> None:
        axes = _grid(((-1.0, 1.0), (-2.0, 2.0)), (16, 24))

        def f(x: float, y: float) -> float:
            return x**2 + 2.0 * x * y + 3.0 * y**2 - x + 0.5 * y + 1.0

        mesh = np.meshgrid(*axes, indexing="ij")
        values = f(*mesh)[None, :, :]
        times = np.array([0.0])

        state = np.array([0.3, -0.7])
        value, gradient = value_function_gradient(axes, values, times, state, time=0.0)

        expected_value = f(0.3, -0.7)
        expected_gradient = np.array([2 * 0.3 + 2 * (-0.7) - 1.0, 2 * 0.3 + 6 * (-0.7) + 0.5])
        assert value == pytest.approx(expected_value, abs=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)

    def test_cubic_4d_gradient_exact(self) -> None:
        axes = _grid(((-1.0, 1.0),) * 4, (8, 9, 10, 11))

        def f(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
            return x**3 - 2.0 * y**2 * z + 0.5 * z**3 + w**2 + x * w

        mesh = np.meshgrid(*axes, indexing="ij")
        values = f(*mesh)[None, ...]
        times = np.array([1.0])

        state = np.array([0.31, -0.42, 0.53, -0.64])
        x, y, z, w = state
        expected_gradient = np.array(
            [3 * x**2 + w, -4.0 * y * z, -2.0 * y**2 + 1.5 * z**2, 2 * w + x]
        )

        _, gradient = value_function_gradient(axes, values, times, state, time=1.0)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)


class TestSmoothFunctionAccuracy:
    """非多项式光滑函数：接口梯度误差应低于中心差分至少一个量级。"""

    def test_gradient_beats_central_difference(self) -> None:
        shape = (41, 43, 39, 37)
        axes = _grid(((-1.0, 1.0),) * 4, shape)
        mesh = np.meshgrid(*axes, indexing="ij")

        def f(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
            return np.sin(x0) * np.cos(x1) + 0.3 * x2**2 * np.sin(x3) + 0.1 * np.exp(x0 * x2)

        values = f(*mesh)[None, ...]
        times = np.array([0.0])

        rng = np.random.default_rng(42)
        errors_spline: list[float] = []
        errors_central: list[float] = []
        spacing = [float(a[1] - a[0]) for a in axes]
        for _ in range(20):
            state = rng.uniform(-0.8, 0.8, size=4)
            x0, x1, x2, x3 = state
            exact = np.array(
                [
                    np.cos(x0) * np.cos(x1) + 0.1 * x2 * np.exp(x0 * x2),
                    -np.sin(x0) * np.sin(x1),
                    0.6 * x2 * np.sin(x3) + 0.1 * x0 * np.exp(x0 * x2),
                    0.3 * x2**2 * np.cos(x3),
                ]
            )

            _, gradient = value_function_gradient(axes, values, times, state, time=0.0)
            errors_spline.append(float(np.linalg.norm(gradient - exact)))

            # 对照组：geo-nrho 现状——np.gradient 中心差分 + 多重线性插值
            node_grads = np.gradient(values[0], *spacing, edge_order=2)
            central = np.array([_multilinear(axes, g, state) for g in node_grads])
            errors_central.append(float(np.linalg.norm(central - exact)))

        assert np.median(errors_spline) < np.median(errors_central) / 10.0


def _multilinear(axes: tuple[np.ndarray, ...], values: np.ndarray, point: np.ndarray) -> float:
    indices = []
    weights = []
    for axis, coordinate in zip(axes, point, strict=True):
        hi = int(np.searchsorted(axis, coordinate, side="right"))
        lo = max(0, min(hi - 1, len(axis) - 2))
        hi = lo + 1
        w = (coordinate - axis[lo]) / (axis[hi] - axis[lo])
        indices.append((lo, hi))
        weights.append(float(np.clip(w, 0.0, 1.0)))
    result = 0.0
    for mask in np.ndindex(*(2,) * len(axes)):
        index = tuple(indices[d][mask[d]] for d in range(len(axes)))
        weight = np.prod([weights[d] if mask[d] else 1.0 - weights[d] for d in range(len(axes))])
        result += float(weight) * float(values[index])
    return result


class TestTimeInterpolation:
    """时间维必选线性插值；单快照视为定常场。"""

    def test_midpoint_linear_in_time(self) -> None:
        axes = _grid(((-1.0, 1.0), (-1.0, 1.0)), (16, 16))
        mesh = np.meshgrid(*axes, indexing="ij")
        q = mesh[0] ** 2 + mesh[1] ** 2  # V(x, t) = t * q(x)
        times = np.array([0.0, 2.0])
        values = np.stack([0.0 * q, 2.0 * q])

        state = np.array([0.4, -0.6])
        value, gradient = value_function_gradient(axes, values, times, state, time=1.0)

        assert value == pytest.approx(1.0 * (0.4**2 + 0.6**2), abs=1e-12)
        np.testing.assert_allclose(gradient, 1.0 * np.array([0.8, -1.2]), atol=1e-10)

    def test_single_snapshot_treated_as_time_invariant(self) -> None:
        axes = _grid(((-1.0, 1.0),), (8,))
        values = (axes[0] ** 2)[None, :]
        times = np.array([3.0])

        value, gradient = value_function_gradient(
            axes, values, times, np.array([0.25]), time=-100.0
        )
        assert value == pytest.approx(0.0625, abs=1e-12)
        np.testing.assert_allclose(gradient, [0.5], atol=1e-10)


class TestOutOfBounds:
    @pytest.fixture
    def grid_2d(self) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        axes = _grid(((-1.0, 1.0), (-1.0, 1.0)), (8, 8))
        mesh = np.meshgrid(*axes, indexing="ij")
        snapshot = mesh[0] ** 2 + mesh[1] ** 2
        values = np.stack([snapshot, snapshot])
        return axes, values, np.array([0.0, 1.0])

    def test_state_outside_grid_raises(self, grid_2d: tuple) -> None:
        axes, values, times = grid_2d
        with pytest.raises(ValueFunctionQueryError, match="状态"):
            value_function_gradient(axes, values, times, np.array([1.5, 0.0]), time=0.5)

    def test_time_outside_range_raises(self, grid_2d: tuple) -> None:
        axes, values, times = grid_2d
        with pytest.raises(ValueFunctionQueryError, match="时间"):
            value_function_gradient(axes, values, times, np.array([0.0, 0.0]), time=2.0)

    def test_boundary_points_accepted(self, grid_2d: tuple) -> None:
        axes, values, times = grid_2d
        value_function_gradient(axes, values, times, np.array([-1.0, 1.0]), time=0.0)
        value_function_gradient(axes, values, times, np.array([0.0, 0.0]), time=1.0)


class TestInputValidation:
    def test_values_shape_mismatch_raises(self) -> None:
        axes = _grid(((-1.0, 1.0),), (8,))
        with pytest.raises(ValueError, match="values"):
            value_function_gradient(axes, np.zeros((2, 8)), np.array([0.0]), np.array([0.0]), 0.0)

    def test_non_increasing_axis_raises(self) -> None:
        axes = (np.array([0.0, 0.5, 0.4, 1.0]),)
        with pytest.raises(ValueError, match="严格递增"):
            value_function_gradient(axes, np.zeros((1, 4)), np.array([0.0]), np.array([0.2]), 0.0)

    def test_error_is_value_error_compatible(self) -> None:
        axes = _grid(((-1.0, 1.0),), (8,))
        values = (axes[0] ** 2)[None, :]
        with pytest.raises(ValueError):
            value_function_gradient(axes, values, np.array([0.0]), np.array([5.0]), 0.0)
