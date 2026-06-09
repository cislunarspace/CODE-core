"""Cover uncovered branches in CoordinateTransformation."""

import numpy as np
import pytest

from e2m2e.core.coordinate import CoordinateTransformation, ReferenceFrame
from e2m2e.core.cr3bp_system import CR3BP_System


@pytest.fixture
def system():
    return CR3BP_System(mu=0.0121506683, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture
def coord(system):
    return CoordinateTransformation(system)


class TestLRUCacheEviction:
    def test_cache_evicts_oldest_when_full(self, coord):
        coord.CACHE_ROTATION_MATRICES = True
        coord.MAX_CACHE_SIZE = 3
        coord.rotation_matrices.clear()
        coord.rotation_matrix_derivatives.clear()

        coord.compute_rotation_matrix(0.1)
        coord.compute_rotation_matrix(0.2)
        coord.compute_rotation_matrix(0.3)
        assert len(coord.rotation_matrices) == 3

        coord.compute_rotation_matrix(0.4)
        assert len(coord.rotation_matrices) == 3
        assert 0.1 not in coord.rotation_matrices
        assert 0.4 in coord.rotation_matrices


class TestNoCoriolisBranch:
    def test_rotating_to_inertial_no_coriolis(self, coord):
        original = coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS
        coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = False
        try:
            state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
            result = coord.rotating_to_inertial(state, time=0.5)
            R = coord.compute_rotation_matrix(0.5)
            expected_vel = R.T @ state[3:]
            assert np.allclose(result[3:], expected_vel, atol=1e-10)
        finally:
            coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = original

    def test_inertial_to_rotating_no_coriolis(self, coord):
        original = coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS
        coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = False
        try:
            state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
            result = coord.inertial_to_rotating(state, time=0.5)
            R = coord.compute_rotation_matrix(0.5)
            expected_vel = R @ state[3:]
            assert np.allclose(result[3:], expected_vel, atol=1e-10)
        finally:
            coord.VELOCITY_TRANSFORM_INCLUDE_CORIOLIS = original


class TestMuNoneValueError:
    def test_barycentric_to_primary_mu_none(self):
        coord = CoordinateTransformation.__new__(CoordinateTransformation)
        coord.mu = None
        coord.rotation_matrices = {}
        coord.rotation_matrix_derivatives = {}
        with pytest.raises(ValueError, match="系统未初始化"):
            coord.barycentric_to_primary(np.zeros(6))

    def test_primary_to_barycentric_mu_none(self):
        coord = CoordinateTransformation.__new__(CoordinateTransformation)
        coord.mu = None
        coord.rotation_matrices = {}
        coord.rotation_matrix_derivatives = {}
        with pytest.raises(ValueError, match="系统未初始化"):
            coord.primary_to_barycentric(np.zeros(6))

    def test_barycentric_to_secondary_mu_none(self):
        coord = CoordinateTransformation.__new__(CoordinateTransformation)
        coord.mu = None
        coord.rotation_matrices = {}
        coord.rotation_matrix_derivatives = {}
        with pytest.raises(ValueError, match="系统未初始化"):
            coord.barycentric_to_secondary(np.zeros(6))

    def test_secondary_to_barycentric_mu_none(self):
        coord = CoordinateTransformation.__new__(CoordinateTransformation)
        coord.mu = None
        coord.rotation_matrices = {}
        coord.rotation_matrix_derivatives = {}
        with pytest.raises(ValueError, match="系统未初始化"):
            coord.secondary_to_barycentric(np.zeros(6))


class TestTransformNotImplementedError:
    def test_unsupported_transform_pair(self, coord):
        with pytest.raises(NotImplementedError):
            coord.transform(
                np.zeros(6),
                from_frame=ReferenceFrame.ROTATING,
                to_frame=ReferenceFrame.PRIMARY_CENTERED,
            )

    def test_transform_inertial_to_primary_centered(self, coord):
        with pytest.raises(NotImplementedError):
            coord.transform(
                np.zeros(6),
                from_frame=ReferenceFrame.INERTIAL,
                to_frame=ReferenceFrame.PRIMARY_CENTERED,
            )


class TestStringRepr:
    def test_str(self, coord):
        s = str(coord)
        assert "CoordinateTransformation" in s
        assert "system=" in s

    def test_repr(self, coord):
        r = repr(coord)
        assert "CoordinateTransformation" in r
        assert "cache_size=" in r


class TestTransformSecondaryFrames:
    def test_barycentric_to_secondary_via_transform(self, coord):
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = coord.transform(state, from_frame="barycentric", to_frame="secondary_centered")
        direct = coord.barycentric_to_secondary(state)
        assert np.allclose(result, direct)

    def test_secondary_to_barycentric_via_transform(self, coord):
        state = np.array([0.5, 0.1, 0.0, 0.01, 0.02, 0.0])
        result = coord.transform(state, from_frame="secondary_centered", to_frame="barycentric")
        direct = coord.secondary_to_barycentric(state)
        assert np.allclose(result, direct)
