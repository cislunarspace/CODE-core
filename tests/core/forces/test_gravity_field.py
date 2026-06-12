"""GravityField 测试。"""

import numpy as np
import pytest

from e2m2e.core.forces import GravityField


@pytest.fixture
def minimal_gravity_file(tmp_path):
    """生成一个最小 .gfc 文件 fixture。"""
    content = """modelname EGM96_TEST
earth_gravity_constant 398600.441500000
radius 6378.136300000
max_degree 2
norm fully_normalized
gfc 0 0 1.000000000000e+00 0.000000000000e+00
gfc 2 0 -4.841651437908e-04 0.000000000000e+00
END"""
    path = tmp_path / "test.gfc"
    path.write_text(content)
    return path


def j2_analytical_acceleration(r, mu, radius, j2):
    """解析 J2 加速度公式。"""
    x, y, z = r
    rr = np.dot(r, r)
    r_norm = np.sqrt(rr)
    factor = -1.5 * j2 * mu * radius**2 / (rr**2 * r_norm)
    common = 1.0 - 5.0 * z * z / rr
    ax = factor * x * common
    ay = factor * y * common
    az = factor * z * (3.0 - 5.0 * z * z / rr)
    return np.array([ax, ay, az])


def test_gravity_field_degree_zero_is_point_mass(minimal_gravity_file):
    """degree=0 时退化为点质量加速度。"""
    gf = GravityField(
        body="EARTH",
        degree=0,
        gravity_file=minimal_gravity_file,
    )

    r = np.array([7000.0, 0.0, 0.0])
    state = np.concatenate([r, np.zeros(3)])
    acc = gf.compute_acceleration(0.0, state, None)

    expected = -gf.gravitational_parameter / np.linalg.norm(r) ** 3 * r
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


def test_gravity_field_degree_two_matches_j2_formula(minimal_gravity_file):
    """degree=2, order=0 时中心引力 + J2 与解析公式一致。"""
    gf = GravityField(
        body="EARTH",
        degree=2,
        order=0,
        gravity_file=minimal_gravity_file,
    )

    # 从正规化 C20 反推 J2
    j2 = -gf.coefficients["C"][2, 0] * np.sqrt(5.0)

    r = np.array([7000.0, 800.0, 900.0])
    state = np.concatenate([r, np.zeros(3)])
    acc = gf.compute_acceleration(0.0, state, None)

    point_mass = -gf.gravitational_parameter / np.linalg.norm(r) ** 3 * r
    j2_acc = j2_analytical_acceleration(
        r, gf.gravitational_parameter, gf.reference_radius, j2
    )
    expected = point_mass + j2_acc
    np.testing.assert_allclose(acc, expected, rtol=1e-12)
