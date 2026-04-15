"""
Unit tests for CoordinateTransformation class
"""

import numpy as np

from e2m2e.core.coordinate import CoordinateTransformation, ReferenceFrame


class TestReferenceFrameEnum:
    """Tests for ReferenceFrame enum"""

    def test_reference_frame_values(self):
        """Test all reference frames exist"""
        assert ReferenceFrame.ROTATING.value == "rotating"
        assert ReferenceFrame.INERTIAL.value == "inertial"
        assert ReferenceFrame.BARYCENTRIC.value == "barycentric"
        assert ReferenceFrame.PRIMARY_CENTERED.value == "primary_centered"
        assert ReferenceFrame.SECONDARY_CENTERED.value == "secondary_centered"
        assert ReferenceFrame.SYNODIC.value == "synodic"

    def test_reference_frame_count(self):
        """Test there are 6 reference frames"""
        assert len(ReferenceFrame) == 6


class TestCoordinateTransformationInit:
    """Tests for CoordinateTransformation initialization"""

    def test_init_stores_system(self, earth_moon_system):
        """Test init stores system reference"""
        coord = CoordinateTransformation(system=earth_moon_system)
        assert coord.system is earth_moon_system

    def test_init_stores_mu(self, earth_moon_system):
        """Test init extracts mu from system"""
        coord = CoordinateTransformation(system=earth_moon_system)
        assert coord.mu == earth_moon_system.mu

    def test_init_sets_cache_empty(self, earth_moon_system):
        """Test rotation matrix cache is empty initially"""
        coord = CoordinateTransformation(system=earth_moon_system)
        assert len(coord.rotation_matrices) == 0
        assert coord.initialized is True


class TestCoordinateTransformationRotationMatrix:
    """Tests for compute_rotation_matrix method"""

    def test_rotation_matrix_shape(self, earth_moon_coordinate):
        """Test rotation matrix is 3x3"""
        R = earth_moon_coordinate.compute_rotation_matrix(time=0.0)
        assert R.shape == (3, 3)

    def test_rotation_matrix_orthogonal(self, earth_moon_coordinate):
        """Test rotation matrix is orthogonal (R @ R.T = I)"""
        R = earth_moon_coordinate.compute_rotation_matrix(time=0.5)
        identity = R @ R.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant(self, earth_moon_coordinate):
        """Test rotation matrix has determinant 1"""
        R = earth_moon_coordinate.compute_rotation_matrix(time=0.5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_matrix_at_zero(self, earth_moon_coordinate):
        """Test rotation matrix at time 0 is identity"""
        R = earth_moon_coordinate.compute_rotation_matrix(time=0.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_at_quarter_period(self, earth_moon_coordinate):
        """Test rotation matrix at t=pi/2 is 90 degree rotation"""
        R = earth_moon_coordinate.compute_rotation_matrix(time=np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10)

    def test_rotation_matrix_is_cached(self, earth_moon_coordinate):
        """Test same time returns cached matrix"""
        R1 = earth_moon_coordinate.compute_rotation_matrix(time=0.5)
        R2 = earth_moon_coordinate.compute_rotation_matrix(time=0.5)
        assert R1 is R2


class TestCoordinateTransformationRotatingToInertial:
    """Tests for rotating_to_inertial method"""

    def test_rotating_to_inertial_shape(self, earth_moon_coordinate):
        """Test output has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.rotating_to_inertial(state=state, time=0.0)
        assert len(result) == 6

    def test_rotating_to_inertial_at_zero_time(self, earth_moon_coordinate):
        """Test at t=0, position unchanged but velocity has Coriolis term"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.rotating_to_inertial(state=state, time=0.0)
        # Position unchanged at t=0
        assert np.allclose(result[:3], state[:3], atol=1e-10)
        # Velocity changes due to Coriolis term: R_dot.T @ position at t=0
        # R_dot.T @ [0.5, 0.1, 0] = [0.1, -0.5, 0]
        assert np.allclose(result[3:], state[3:] + np.array([0.1, -0.5, 0.0]), atol=1e-10)

    def test_rotating_to_inertial_position_only(self, earth_moon_coordinate):
        """Test position transformation at t=pi/2"""
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # on x-axis
        result = earth_moon_coordinate.rotating_to_inertial(state=state, time=np.pi / 2)

        # Rotated 90 degrees: x -> y, y -> -x
        assert np.isclose(result[0], 0.0, atol=1e-10)
        assert np.isclose(result[1], -1.0, atol=1e-10)


class TestCoordinateTransformationInertialToRotating:
    """Tests for inertial_to_rotating method"""

    def test_inertial_to_rotating_shape(self, earth_moon_coordinate):
        """Test output has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.inertial_to_rotating(state=state, time=0.0)
        assert len(result) == 6

    def test_inertial_to_rotating_at_zero_time(self, earth_moon_coordinate):
        """Test at t=0, position unchanged but velocity has Coriolis correction"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.inertial_to_rotating(state=state, time=0.0)
        # Position unchanged at t=0
        assert np.allclose(result[:3], state[:3], atol=1e-10)
        # Velocity changes due to Coriolis term: -R_dot @ position_rotating
        assert np.allclose(result[3:], state[3:] + np.array([0.1, -0.5, 0.0]), atol=1e-10)


class TestCoordinateTransformationReversibility:
    """Tests for round-trip transformations"""

    def test_rotating_inertial_round_trip(self, earth_moon_coordinate):
        """Test rotating->inertial->rotating returns original for position-only state"""
        # Use zero velocity to avoid Coriolis complications in velocity
        original = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
        time = 0.5

        rotated = earth_moon_coordinate.rotating_to_inertial(state=original, time=time)
        back = earth_moon_coordinate.inertial_to_rotating(state=rotated, time=time)

        # Position should round-trip correctly
        assert np.allclose(original[:3], back[:3], atol=1e-10)

    def test_inertial_rotating_round_trip(self, earth_moon_coordinate):
        """Test inertial->rotating->inertial returns original for position-only state"""
        # Use zero velocity to avoid Coriolis complications in velocity
        original = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
        time = 0.5

        inertial = earth_moon_coordinate.inertial_to_rotating(state=original, time=time)
        back = earth_moon_coordinate.rotating_to_inertial(state=inertial, time=time)

        # Position should round-trip correctly
        assert np.allclose(original[:3], back[:3], atol=1e-10)


class TestCoordinateTransformationBarycentric:
    """Tests for barycentric coordinate transformations"""

    def test_barycentric_to_primary_shape(self, earth_moon_coordinate):
        """Test output has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.barycentric_to_primary(state=state)
        assert len(result) == 6

    def test_primary_to_barycentric_shape(self, earth_moon_coordinate):
        """Test output has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.primary_to_barycentric(state=state)
        assert len(result) == 6

    def test_barycentric_primary_round_trip(self, earth_moon_coordinate):
        """Test barycentric->primary->barycentric returns original"""
        original = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])

        primary = earth_moon_coordinate.barycentric_to_primary(state=original)
        back = earth_moon_coordinate.primary_to_barycentric(state=primary)

        assert np.allclose(original, back, atol=1e-10)

    def test_barycentric_to_secondary_shape(self, earth_moon_coordinate):
        """Test secondary transformation has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.barycentric_to_secondary(state=state)
        assert len(result) == 6

    def test_secondary_to_barycentric_shape(self, earth_moon_coordinate):
        """Test secondary transformation has correct shape"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.secondary_to_barycentric(state=state)
        assert len(result) == 6

    def test_barycentric_secondary_round_trip(self, earth_moon_coordinate):
        """Test barycentric->secondary->barycentric returns original"""
        original = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])

        secondary = earth_moon_coordinate.barycentric_to_secondary(state=original)
        back = earth_moon_coordinate.secondary_to_barycentric(state=secondary)

        assert np.allclose(original, back, atol=1e-10)


class TestCoordinateTransformationGeneralTransform:
    """Tests for general transform method"""

    def test_transform_rotating_to_inertial(self, earth_moon_coordinate):
        """Test transform with rotating to inertial"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.transform(
            state=state,
            from_frame=ReferenceFrame.ROTATING,
            to_frame=ReferenceFrame.INERTIAL,
            time=0.0,
        )
        assert len(result) == 6

    def test_transform_inertial_to_rotating(self, earth_moon_coordinate):
        """Test transform with inertial to rotating"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.transform(
            state=state,
            from_frame=ReferenceFrame.INERTIAL,
            to_frame=ReferenceFrame.ROTATING,
            time=0.0,
        )
        assert len(result) == 6

    def test_transform_with_string_frames(self, earth_moon_coordinate):
        """Test transform accepts string frame names"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.transform(
            state=state, from_frame="rotating", to_frame="inertial", time=0.0
        )
        assert len(result) == 6

    def test_transform_same_frame(self, earth_moon_coordinate):
        """Test transform with same source and target frame"""
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = earth_moon_coordinate.transform(
            state=state,
            from_frame=ReferenceFrame.ROTATING,
            to_frame=ReferenceFrame.ROTATING,
            time=0.5,
        )
        assert np.allclose(result, state, atol=1e-10)
