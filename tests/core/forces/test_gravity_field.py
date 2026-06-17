"""GravityField 测试。

覆盖 degree=0 点质量退化、degree=2 J2 解析一致性、
潮汐集成与 dot 历元外推。
"""

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


# ----------------------------------------------------------------------------
# 潮汐集成测试(Slice 7 / AC1-AC4)
# ----------------------------------------------------------------------------


class _StubSpice:
    """SPICE 桩:返回固定的 Sun/Moon 位置与 GM,供潮汐集成测试隔离。"""

    _GM = {"SUN": 1.32712440018e11, "MOON": 4902.8001, "EARTH": 398600.4415}

    def get_body_position(self, target, et, frame, observer):
        if target == "SUN":
            return np.array([1.495978707e8, 0.0, 0.0])
        if target == "MOON":
            return np.array([384400.0, 0.0, 0.0])
        raise ValueError(f"unexpected target {target}")

    def get_gm(self, body):
        return self._GM[body]


class _StubSystem:
    """最小系统:只暴露 spice,供 _effective_coefficients 使用。"""

    def __init__(self):
        self.spice = _StubSpice()


class TestGravityFieldTideIntegration:
    """GravityField 潮汐集成(Slice 7):_effective_coefficients 在不同档位下
    返回不同有效系数。
    """

    def test_none_mode_returns_base_coefficients(self, minimal_gravity_file):
        """tide_mode='none' 时有效系数等于文件原始系数。"""
        gf = GravityField("EARTH", degree=2, gravity_file=minimal_gravity_file)
        C_eff, S_eff = gf._effective_coefficients(t=0.0, system=None)

        np.testing.assert_array_equal(C_eff, gf._data.C)
        np.testing.assert_array_equal(S_eff, gf._data.S)

    def test_solid_mode_changes_c20(self, minimal_gravity_file):
        """tide_mode='solid' 时 C20 含固体潮修正(≠ 原始)。"""
        gf_none = GravityField("EARTH", degree=2, gravity_file=minimal_gravity_file)
        gf_solid = GravityField(
            "EARTH", degree=2, gravity_file=minimal_gravity_file, tide_mode="solid"
        )
        sys_stub = _StubSystem()

        C_none, _ = gf_none._effective_coefficients(t=0.0, system=None)
        C_solid, _ = gf_solid._effective_coefficients(t=0.0, system=sys_stub)

        assert not np.allclose(C_none[2, 0], C_solid[2, 0], atol=1e-15)

    def test_solid_and_pole_adds_pole_tide(self, minimal_gravity_file):
        """tide_mode='solid_and_pole' 比 'solid' 多极潮修正(ΔC21 不同)。"""
        gf_solid = GravityField(
            "EARTH", degree=2, gravity_file=minimal_gravity_file, tide_mode="solid"
        )
        gf_pole = GravityField(
            "EARTH",
            degree=2,
            gravity_file=minimal_gravity_file,
            tide_mode="solid_and_pole",
            polar_motion_provider=lambda et: (0.1, 0.3),
        )
        sys_stub = _StubSystem()

        _, S_solid = gf_solid._effective_coefficients(t=0.0, system=sys_stub)
        _, S_pole = gf_pole._effective_coefficients(t=0.0, system=sys_stub)

        # 极潮影响 (2,1):S21 应不同
        assert not np.allclose(S_solid[2, 1], S_pole[2, 1], atol=1e-15)

    def test_solid_and_pole_without_provider_raises(self, minimal_gravity_file):
        """solid_and_pole 档缺 polar_motion_provider 抛 ValueError。"""
        gf = GravityField(
            "EARTH",
            degree=2,
            gravity_file=minimal_gravity_file,
            tide_mode="solid_and_pole",
        )

        with pytest.raises(ValueError, match="polar_motion_provider"):
            gf._effective_coefficients(t=0.0, system=_StubSystem())

    def test_zero_tide_convention_differs_from_tide_free(self, minimal_gravity_file):
        """zero_tide 减去永久潮汐,C20 与 tide_free 不同。"""
        gf_free = GravityField(
            "EARTH",
            degree=2,
            gravity_file=minimal_gravity_file,
            tide_mode="solid",
            tide_convention="tide_free",
        )
        gf_zero = GravityField(
            "EARTH",
            degree=2,
            gravity_file=minimal_gravity_file,
            tide_mode="solid",
            tide_convention="zero_tide",
        )
        sys_stub = _StubSystem()

        C_free, _ = gf_free._effective_coefficients(t=0.0, system=sys_stub)
        C_zero, _ = gf_zero._effective_coefficients(t=0.0, system=sys_stub)

        # 永久潮汐 C20 量级 ~1e-9,差异可测
        assert abs(C_free[2, 0] - C_zero[2, 0]) > 1e-10

    def test_dot_extrapolation_with_epoch(self, tmp_path):
        """epoch + dot 行 → C20 按历元外推(Slice 2 集成)。"""
        content = """modelname DOT_TEST
earth_gravity_constant 398600.441500000
radius 6378.136300000
max_degree 2
norm fully_normalized
gfc 0 0 1.000000000000e+00 0.000000000000e+00
gfc 2 0 -4.841651437908e-04 0.000000000000e+00
gfc 2 1 -1.869876401566e-10 1.195280120309e-09
gfc 2 2 2.439143523982e-06 -1.400166836544e-06
dot 2 0 1.0e-11 0.000000000000e+00
END"""
        path = tmp_path / "dot.gfc"
        path.write_text(content)

        seconds_per_year = 365.25 * 86400.0
        gf = GravityField("EARTH", degree=2, gravity_file=path, epoch=0.0)

        C0, _ = gf._effective_coefficients(t=0.0, system=None)
        C1, _ = gf._effective_coefficients(t=seconds_per_year, system=None)

        # 1 年后 C20 增加 dotC*1 年
        np.testing.assert_allclose(C1[2, 0], C0[2, 0] + 1.0e-11, rtol=1e-12)

    def test_invalid_tide_mode_raises(self):
        with pytest.raises(ValueError, match="tide_mode"):
            GravityField("EARTH", tide_mode="weird")
