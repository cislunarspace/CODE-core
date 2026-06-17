"""Axes + Origin + CoordinateSystem 框架单元测试。

验证抽象基类、固定/旋转轴状态变换、
原点偏移、正交性与向量变换。
"""


import numpy as np
import pytest

from e2m2e.core.axes import Axes
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.origin import Origin


class FixedAxes(Axes):
    """Axes that rotates vectors by a fixed angle around the z-axis."""

    def __init__(self, angle: float) -> None:
        self._angle = angle
        self._cos = np.cos(angle)
        self._sin = np.sin(angle)

    def rotation_matrix(self, et: float) -> np.ndarray:
        return np.array(
            [
                [self._cos, -self._sin, 0.0],
                [self._sin, self._cos, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def angular_velocity(self, et: float) -> np.ndarray:
        return np.zeros(3)


class FixedOrigin(Origin):
    """Origin with a constant offset state."""

    def __init__(self, state: np.ndarray) -> None:
        self._state = np.asarray(state, dtype=float)

    def state(self, et: float) -> np.ndarray:
        return self._state.copy()


class TestAxesABC:
    """Tests for the Axes abstract base class."""

    def test_axes_is_abstract(self):
        """Axes cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Axes()

    def test_fixed_axes_is_subclass(self):
        """A concrete Axes subclass can be instantiated."""
        axes = FixedAxes(angle=np.pi / 4)
        assert isinstance(axes, Axes)


class TestOriginABC:
    """Tests for the Origin abstract base class."""

    def test_origin_is_abstract(self):
        """Origin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Origin()

    def test_fixed_origin_returns_state(self):
        """A concrete Origin returns its state."""
        origin = FixedOrigin(state=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(
            origin.state(et=0.0), np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        )


class RotatingAxes(Axes):
    """Axes rotating at a constant angular velocity around the z-axis."""

    def __init__(self, omega: float) -> None:
        self._omega = omega

    def rotation_matrix(self, et: float) -> np.ndarray:
        cos = np.cos(self._omega * et)
        sin = np.sin(self._omega * et)
        return np.array(
            [
                [cos, -sin, 0.0],
                [sin, cos, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def angular_velocity(self, et: float) -> np.ndarray:
        return np.array([0.0, 0.0, self._omega])


class TestCoordinateSystemState:
    """Tests for state transformation through CoordinateSystem."""

    def test_transform_state_identity(self):
        """Transforming between identical coordinate systems leaves state unchanged."""
        axes = FixedAxes(angle=0.0)
        origin = FixedOrigin(state=np.zeros(6))
        cs = CoordinateSystem(axes=axes, origin=origin)

        state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        result = cs.transform_state(state, from_cs=cs, to_cs=cs, et=0.0)

        np.testing.assert_allclose(result, state, atol=1e-14)

    def test_transform_state_origin_offset(self):
        """Transforming between coordinate systems with different origins applies offset."""
        axes = FixedAxes(angle=0.0)
        origin_a = FixedOrigin(state=np.zeros(6))
        origin_b = FixedOrigin(state=np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0]))
        cs_a = CoordinateSystem(axes=axes, origin=origin_a)
        cs_b = CoordinateSystem(axes=axes, origin=origin_b)

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cs_a.transform_state(state, from_cs=cs_a, to_cs=cs_b, et=0.0)

        expected = np.array([-1.0, 0.0, 0.0, -0.5, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_transform_state_rotating_relative_velocity(self):
        """Transforming from inertial to rotating axes adds coriolis term."""
        axes_inertial = FixedAxes(angle=0.0)
        axes_rotating = RotatingAxes(omega=1.0)
        origin = FixedOrigin(state=np.zeros(6))
        cs_inertial = CoordinateSystem(axes=axes_inertial, origin=origin)
        cs_rotating = CoordinateSystem(axes=axes_rotating, origin=origin)

        et = 0.0
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cs_inertial.transform_state(
            state, from_cs=cs_inertial, to_cs=cs_rotating, et=et
        )

        # At et=0, rotating frame coincides with inertial frame, so r is unchanged.
        # v_rotating = v_inertial - omega x r = [0, 0, 0] - [0, 0, 1] x [1, 0, 0]
        #            = -[0, 1, 0] = [0, -1, 0]
        np.testing.assert_allclose(result[:3], np.array([1.0, 0.0, 0.0]), atol=1e-14)
        np.testing.assert_allclose(result[3:], np.array([0.0, -1.0, 0.0]), atol=1e-14)

    def test_transform_state_rotating_target_at_nonzero_epoch(self):
        """Rotating target axes subtract Rdot @ r_axes before projecting velocity."""
        axes_inertial = FixedAxes(angle=0.0)
        axes_rotating = RotatingAxes(omega=1.0)
        origin = FixedOrigin(state=np.zeros(6))
        cs_inertial = CoordinateSystem(axes=axes_inertial, origin=origin)
        cs_rotating = CoordinateSystem(axes=axes_rotating, origin=origin)

        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cs_inertial.transform_state(
            state, from_cs=cs_inertial, to_cs=cs_rotating, et=np.pi / 2
        )

        np.testing.assert_allclose(result[:3], np.array([0.0, -1.0, 0.0]), atol=1e-14)
        np.testing.assert_allclose(result[3:], np.array([-1.0, 0.0, 0.0]), atol=1e-14)

    def test_transform_state_round_trip(self):
        """Transforming forward and backward returns the original state."""
        axes_a = FixedAxes(angle=np.pi / 6)
        axes_b = RotatingAxes(omega=0.5)
        origin_a = FixedOrigin(state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        origin_b = FixedOrigin(state=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
        cs_a = CoordinateSystem(axes=axes_a, origin=origin_a)
        cs_b = CoordinateSystem(axes=axes_b, origin=origin_b)

        state = np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6])
        et = 1.0
        intermediate = cs_a.transform_state(state, from_cs=cs_a, to_cs=cs_b, et=et)
        result = cs_b.transform_state(intermediate, from_cs=cs_b, to_cs=cs_a, et=et)

        np.testing.assert_allclose(result, state, atol=1e-14)

        """Transforming between identical axes leaves the vector unchanged."""
        axes = FixedAxes(angle=0.0)
        origin = FixedOrigin(state=np.zeros(6))
        cs = CoordinateSystem(axes=axes, origin=origin)

        vec = np.array([1.0, 0.0, 0.0])
        result = cs.transform_vector(vec, from_cs=cs, to_cs=cs, et=0.0)

        np.testing.assert_allclose(result, vec, atol=1e-14)

    def test_transform_vector_rotation(self):
        """Transforming between rotated axes applies the rotation."""
        axes_a = FixedAxes(angle=0.0)
        axes_b = FixedAxes(angle=np.pi / 2)
        origin = FixedOrigin(state=np.zeros(6))
        cs_a = CoordinateSystem(axes=axes_a, origin=origin)
        cs_b = CoordinateSystem(axes=axes_b, origin=origin)

        vec = np.array([1.0, 0.0, 0.0])
        result = cs_a.transform_vector(vec, from_cs=cs_a, to_cs=cs_b, et=0.0)

        np.testing.assert_allclose(result, np.array([0.0, -1.0, 0.0]), atol=1e-14)


class TestCoordinateSystemOrthogonality:
    """CoordinateSystem 中使用的 Axes 旋转矩阵满足正交性。

    issue #76 验收第 4 条"所有转换矩阵满足正交性(R @ R.T = I 误差 < 1e-14)"
    的字面落实——在新框架的 test_coordinate_system.py 里显式覆盖,因
    test_standard_axes.py 覆盖的是具体类,本文件覆盖"框架内的任何 Axes
    子类"契约。
    """

    @pytest.mark.parametrize("et", [0.0, 1.0, 100.0, 86400.0, -86400.0])
    def test_fixed_axes_rotation_matrix_orthogonal(self, et):
        """FixedAxes 在多个 et 上 R @ R.T = I,Frobenius 误差 < 1e-14。"""
        axes = FixedAxes(angle=np.pi / 3)
        R = axes.rotation_matrix(et)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)

    @pytest.mark.parametrize("et", [0.0, 1.0, 100.0, 86400.0])
    def test_rotating_axes_rotation_matrix_orthogonal(self, et):
        """RotatingAxes 在多个 et 上正交。"""
        axes = RotatingAxes(omega=0.5)
        R = axes.rotation_matrix(et)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)


class TestCoordinateSystemTransformVector:
    """transform_vector 边界用例:零向量经任意坐标变换后仍为零。"""

    def test_transform_vector_zero_vector_is_zero(self):
        """零向量经过任意坐标变换后仍是零向量(无信息被捏造)。"""
        axes_a = FixedAxes(angle=0.0)
        axes_b = FixedAxes(angle=np.pi / 4)
        origin = FixedOrigin(state=np.zeros(6))
        cs_a = CoordinateSystem(axes=axes_a, origin=origin)
        cs_b = CoordinateSystem(axes=axes_b, origin=origin)

        zero = np.zeros(3)
        result = cs_a.transform_vector(zero, from_cs=cs_a, to_cs=cs_b, et=0.0)
        np.testing.assert_array_equal(result, zero)
