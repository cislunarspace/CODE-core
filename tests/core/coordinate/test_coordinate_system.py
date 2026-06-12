"""
Unit tests for the new Axes + Origin + CoordinateSystem framework.
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
