"""GravityField 测试。

覆盖 degree=0 点质量退化、degree=2 J2 解析一致性、
潮汐集成与 dot 历元外推,以及 issue #187 的天体无关改造:
按 ``body`` 切换 body-fixed frame(地球 ITRF93、月球 MOON_PA)与默认重力文件。
"""

import os

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
    j2_acc = j2_analytical_acceleration(r, gf.gravitational_parameter, gf.reference_radius, j2)
    expected = point_mass + j2_acc
    np.testing.assert_allclose(acc, expected, rtol=1e-12)


# ----------------------------------------------------------------------------
# 潮汐集成测试(Slice 7 / AC1-AC4)
# ----------------------------------------------------------------------------


class _StubSpice:
    """SPICE 桩:返回固定的 Sun/Moon/Earth 位置与 GM,供潮汐集成测试隔离。"""

    _GM = {"SUN": 1.32712440018e11, "MOON": 4902.8001, "EARTH": 398600.4415}

    def get_body_position(self, target, et, frame, observer):
        if target == "SUN":
            return np.array([1.495978707e8, 0.0, 0.0])
        if target == "MOON":
            return np.array([384400.0, 0.0, 0.0])
        if target == "EARTH":
            # 地球相对观察者(月球)的位置:沿 +x,月地距离 384400 km。
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


# ----------------------------------------------------------------------------
# 月球固体潮集成(issue #188):body='MOON' + tide_mode='solid'
# 扰动体=地球,Love 数从 grgm900c.tide 读(k₂=0.024116),只做 Step1。
# ----------------------------------------------------------------------------


class TestMoonTideIntegration:
    """body='MOON' 的固体潮集成:扰动体=地球,只走 Step1(无 Step2/极潮)。"""

    def test_moon_none_mode_returns_base_coefficients(self):
        """tide_mode='none'(默认)时月球系数不变(安全默认)。"""
        gf_none = GravityField("MOON", degree=4)
        C_eff, S_eff = gf_none._effective_coefficients(t=0.0, system=None)
        np.testing.assert_array_equal(C_eff, gf_none._data.C)
        np.testing.assert_array_equal(S_eff, gf_none._data.S)

    def test_moon_solid_mode_changes_c20(self):
        """tide_mode='solid' 时月球 C20 含固体潮修正(≠ 原始)。"""
        gf_none = GravityField("MOON", degree=4)
        gf_solid = GravityField("MOON", degree=4, tide_mode="solid")
        sys_stub = _StubSystem()

        C_none, _ = gf_none._effective_coefficients(t=0.0, system=None)
        C_solid, _ = gf_solid._effective_coefficients(t=0.0, system=sys_stub)

        assert not np.allclose(C_none[2, 0], C_solid[2, 0], atol=1e-15)

    def test_moon_solid_delta_matches_direct_step1(self):
        """body='MOON' 走 _effective_coefficients 的 ΔC/ΔS 与直接调
        solid_tide_step1(扰动体=地球,Love=月球 k₂)逐字一致。"""
        from e2m2e.core.forces.earth_tide import solid_tide_step1

        gf_solid = GravityField("MOON", degree=4, tide_mode="solid")
        gf_none = GravityField("MOON", degree=4)
        sys_stub = _StubSystem()

        C_solid, S_solid = gf_solid._effective_coefficients(t=0.0, system=sys_stub)
        C_none, S_none = gf_none._effective_coefficients(t=0.0, system=None)
        dC = C_solid - C_none
        dS = S_solid - S_none

        # 手算:扰动体=地球(相对月球,沿 +x 384400 km),Love=k₂=0.024116
        k_moon = np.zeros((5, 5), dtype=float)
        k_moon[2, 0] = k_moon[2, 1] = k_moon[2, 2] = 0.024116
        earth_pos = np.array([384400.0, 0.0, 0.0])
        expected_dC, expected_dS = solid_tide_step1(
            [(earth_pos, 398600.4415)],
            k_love=k_moon,
            k_plus=None,
            mu_central=gf_solid.gravitational_parameter,
            r_central=gf_solid.reference_radius,
        )
        n = 5
        np.testing.assert_allclose(dC[:n, :n], expected_dC[:n, :n], atol=1e-15)
        np.testing.assert_allclose(dS[:n, :n], expected_dS[:n, :n], atol=1e-15)

    def test_moon_solid_delta_c20_reasonable_magnitude(self):
        """月球固体潮 ΔC20 量级 ~1e-8(地球作为扰动体,massratio μ_E/μ_M≈81)。"""
        gf_solid = GravityField("MOON", degree=4, tide_mode="solid")
        gf_none = GravityField("MOON", degree=4)
        sys_stub = _StubSystem()

        C_solid, _ = gf_solid._effective_coefficients(t=0.0, system=sys_stub)
        C_none, _ = gf_none._effective_coefficients(t=0.0, system=None)
        dC20 = C_solid[2, 0] - C_none[2, 0]

        # 量级 1e-9 到 1e-7
        assert 1e-9 < abs(dC20) < 1e-7
        # P20(0) < 0 → ΔC20 < 0
        assert dC20 < 0.0

    def test_moon_solid_no_step2_or_pole(self):
        """月球固体潮不追加 Step2/极潮:不同时刻 ΔC[2,1] 的频率相关项不变
        (Step2 才有时变;月球只走 Step1,扰动体位置固定时 Δ 恒定)。"""
        gf_solid = GravityField("MOON", degree=4, tide_mode="solid")
        sys_stub = _StubSystem()

        C_t0, S_t0 = gf_solid._effective_coefficients(t=0.0, system=sys_stub)
        C_t1, S_t1 = gf_solid._effective_coefficients(
            t=86400.0 * 30, system=sys_stub  # 30 天后
        )
        # Stub 返回固定扰动体位置 → Δ 恒定(无 Step2 时变性)
        np.testing.assert_allclose(C_t0, C_t1, atol=1e-15)
        np.testing.assert_allclose(S_t0, S_t1, atol=1e-15)


# ----------------------------------------------------------------------------
# 天体无关改造(issue #187):按 body 切换 body-fixed frame 与默认重力文件
# ----------------------------------------------------------------------------

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernels"
)


class _SystemStub:
    """暴露 ``spice`` 与 ``coordinate_system`` 的最小系统桩,供坐标轴构造测试。

    ``coordinate_system`` 在 GravityField 转换路径里被读取,这里置 None 即可,
    因为新测试只调用 ``_get_input_coordinate_system`` 验证轴选择。
    """

    def __init__(self, spice):
        self.spice = spice
        self.coordinate_system = None


@pytest.fixture
def body_frame_spice():
    """加载地月 SPICE 内核并返回 SPICEManager,测试后自动卸载。

    需要 ``kernels/`` 下有 de430/de440s BSP、naif0012.tls、pck00010.tpc、
    earth_latest_high_prec.bpc(定义 ITRF93)、SPICELunaFrameKernel.tf +
    SPICELunaCurrentKernel.bpc(定义 MOON_PA);缺失则跳过。
    """
    import spiceypy

    from e2m2e.core.spice import SPICEManager

    if not os.path.isdir(_KERNELS_DIR):
        pytest.skip("kernels directory not found")
    # 先用 spiceypy 探测 ITRF93 / MOON_PA 是否可用,避免逐文件判断。
    needed = [
        "de430.bsp",
        "de440s.bsp",
        "naif0012.tls",
        "pck00010.tpc",
        "earth_latest_high_prec.bpc",
        "SPICELunaCurrentKernel.bpc",
        "SPICELunaFrameKernel.tf",
    ]
    loaded = []
    manager = SPICEManager()
    try:
        # BSP / TLS: 取第一个存在的
        for name in ("de430.bsp", "de440s.bsp"):
            path = os.path.join(_KERNELS_DIR, name)
            if os.path.exists(path):
                manager.load_kernel(path)
                loaded.append(path)
                break
        for name in needed[2:]:
            path = os.path.join(_KERNELS_DIR, name)
            if os.path.exists(path):
                manager.load_kernel(path)
                loaded.append(path)
        et = manager.utc_to_et("2026-01-01T00:00:00")
        try:
            spiceypy.pxform("ITRF93", "J2000", et)
            spiceypy.pxform("MOON_PA", "J2000", et)
        except Exception:
            pytest.skip("ITRF93/MOON_PA frames not available; missing kernels")
        yield manager
    finally:
        for path in reversed(loaded):
            manager.unload_kernel(path)


class TestBodyAgnosticGravityField:
    """issue #187:GravityField 按 body 切换 body-fixed frame 与默认文件。"""

    def test_earth_default_input_frame_is_itrf93(self):
        """``body='EARTH'`` 默认 input_frame 为 ITRF93。"""
        gf = GravityField("EARTH", degree=2)
        assert gf.input_frame == "ITRF93"

    def test_moon_default_input_frame_is_moon_pa(self):
        """``body='MOON'`` 默认 input_frame 为 MOON_PA。"""
        gf = GravityField("MOON", degree=2)
        assert gf.input_frame == "MOON_PA"

    def test_explicit_input_frame_overrides_body_default(self):
        """显式传入 input_frame 时覆盖按 body 推导的默认值。"""
        gf = GravityField("EARTH", degree=2, input_frame="IAU_EARTH")
        assert gf.input_frame == "IAU_EARTH"

    def test_unknown_body_without_input_frame_raises(self):
        """未知 body 且不传 input_frame 时报错。"""
        with pytest.raises(ValueError, match="default body-fixed frame"):
            GravityField("MARS", degree=2)

    def test_unknown_body_without_gravity_file_raises(self):
        """未知 body 且不传 gravity_file 时报错(input_frame 给定后才会触达)。"""
        with pytest.raises(ValueError, match="default gravity file"):
            GravityField("MARS", degree=2, input_frame="IAU_MARS")

    # -- 默认重力文件守卫 ----------------------------------------------------

    def test_earth_default_file_is_egm96(self):
        """``body='EARTH'`` 不给 gravity_file 时加载包内 EGM96-to10。"""
        gf = GravityField("EARTH", degree=2)
        assert gf.gravitational_parameter == pytest.approx(398600.4415, rel=1e-9)
        assert gf.reference_radius == pytest.approx(6378.1363, rel=1e-9)

    def test_moon_default_file_is_grgm900c(self):
        """``body='MOON'`` 不给 gravity_file 时加载包内 GRGM900C。"""
        gf = GravityField("MOON", degree=2)
        assert gf.gravitational_parameter == pytest.approx(4902.800, rel=1e-4)
        assert gf.reference_radius == pytest.approx(1738.0, rel=1e-9)

    def test_moon_construct_with_explicit_degree(self):
        """``GravityField('MOON', degree=10, gravity_file=...)`` 可构造。"""
        from importlib import resources

        ref = resources.files("e2m2e.core.forces.data").joinpath("grgm900c.cof")
        with ref.open("rb") as f:
            content = f.read()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cof", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            gf = GravityField("MOON", degree=10, gravity_file=tmp_path)
            assert gf.degree == 10
            # C[1,0] 在月球模型中为 0(质心位于月心)
            assert gf.coefficients["C"][1, 0] == pytest.approx(0.0, abs=1e-15)
        finally:
            os.unlink(tmp_path)

    # -- body-fixed 轴切换(SPACE-backed) ---------------------------------

    def test_earth_axes_use_itrf93(self, body_frame_spice):
        """EARTH 的 _get_input_coordinate_system 用 ITRFSpiceAxes(frame=ITRF93)。"""
        from e2m2e.core.standard_axes import ITRFSpiceAxes

        gf = GravityField("EARTH", degree=2)
        cs = gf._get_input_coordinate_system(_SystemStub(body_frame_spice))
        assert isinstance(cs.axes, ITRFSpiceAxes)
        assert cs.axes.frame == "ITRF93"

    def test_moon_axes_use_moon_pa(self, body_frame_spice):
        """MOON 的 _get_input_coordinate_system 用 ITRFSpiceAxes(frame=MOON_PA)。"""
        from e2m2e.core.standard_axes import ITRFSpiceAxes

        gf = GravityField("MOON", degree=2)
        cs = gf._get_input_coordinate_system(_SystemStub(body_frame_spice))
        assert isinstance(cs.axes, ITRFSpiceAxes)
        assert cs.axes.frame == "MOON_PA"

    def test_earth_and_moon_axes_differ(self, body_frame_spice):
        """地球与月球的 body-fixed 旋转矩阵不同(不同 frame)。"""
        gf_earth = GravityField("EARTH", degree=2)
        gf_moon = GravityField("MOON", degree=2)
        et = body_frame_spice.utc_to_et("2026-01-01T00:00:00")
        cs_e = gf_earth._get_input_coordinate_system(_SystemStub(body_frame_spice))
        cs_m = gf_moon._get_input_coordinate_system(_SystemStub(body_frame_spice))
        Re = cs_e.axes.rotation_matrix(et)
        Rm = cs_m.axes.rotation_matrix(et)
        assert not np.allclose(Re, Rm)
        # 二者均为正交矩阵(行列式 = 1)
        assert np.linalg.det(Re) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.det(Rm) == pytest.approx(1.0, abs=1e-9)

    # -- 月球引力单点加速度 --------------------------------------------------

    def test_moon_gravity_single_point_magnitude_and_direction(self):
        """月心轨道上一点的月球引力加速度量级 ~1.6e-3 km/s^2,方向指向月心。

        使用 system=None 直接在 MOON_PA 原生系中计算,绕开坐标转换;
        球谐递推本身天体无关。
        """
        gf = GravityField("MOON", degree=10)
        # 距月心 1738 + 100 = 1838 km,沿 MOON_PA +x 轴
        r = np.array([1738.0 + 100.0, 0.0, 0.0])
        state = np.concatenate([r, np.zeros(3)])
        acc = gf.compute_acceleration(0.0, state, None)

        # 量级:月球表面重力 ~1.62e-3 km/s^2,100km 高度略小
        acc_norm = np.linalg.norm(acc)
        assert 1.0e-3 < acc_norm < 2.0e-3

        # 方向:吸引,acc 与 r 反向
        assert np.dot(acc, r) < 0.0

        # 与点质量加速度接近(高阶项在 100km 高度贡献 < 1%)
        acc_pm = -gf.gravitational_parameter / np.linalg.norm(r) ** 3 * r
        rel_diff = np.linalg.norm(acc - acc_pm) / np.linalg.norm(acc_pm)
        assert rel_diff < 0.01

    def test_moon_gravity_degree_zero_is_point_mass(self):
        """月球 degree=0 退化为月心点质量引力。"""
        gf = GravityField("MOON", degree=0)
        r = np.array([2000.0, 500.0, -300.0])
        state = np.concatenate([r, np.zeros(3)])
        acc = gf.compute_acceleration(0.0, state, None)
        expected = -gf.gravitational_parameter / np.linalg.norm(r) ** 3 * r
        np.testing.assert_allclose(acc, expected, rtol=1e-12)

