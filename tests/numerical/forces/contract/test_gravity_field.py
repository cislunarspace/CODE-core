"""GravityField 测试。

Python 单点 ``compute_acceleration`` 与 ``_effective_coefficients`` 已删除；
加速度与潮汐由 Rust ``propagate_compiled`` /
``gravity_field_acceleration`` 承载。本文件保留构造校验、配置属性与
``to_rust_spec`` 序列化契约。
"""

import os

import numpy as np
import pytest

from e2m2e.algorithm.forces import GravityField
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


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


def test_gravity_field_degree_zero_rust_spec_matches_point_mass(minimal_gravity_file):
    """degree=0 的 GravityField spec 与 PointMassGravity 物理等价（同 mu）。"""
    from e2m2e.algorithm.forces import PointMassGravity

    system = FakeSystem()
    gf = GravityField(
        body="EARTH",
        degree=0,
        gravity_file=minimal_gravity_file,
    )
    pm = PointMassGravity(body="EARTH", mu=gf.gravitational_parameter)

    gf_spec = gf.to_rust_spec(system)
    pm_spec = pm.to_rust_spec(system)

    # GravityField degree=0 的物理内容是中心点质量；与 PointMassGravity 同 mu
    assert gf_spec[0] == "gravity"
    assert pm_spec == ("point_mass", gf.gravitational_parameter)
    # C[0,0]=1、其余为零，degree=0 时只含中心项
    c_flat = gf_spec[1]
    assert c_flat[0] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0) for v in c_flat[1:])


def test_gravity_field_invalid_tide_mode_raises():
    """非法 tide_mode 抛 ValueError。"""
    with pytest.raises(ValueError, match="tide_mode"):
        GravityField("EARTH", tide_mode="weird")


def test_gravity_field_invalid_tide_convention_raises():
    """非法 tide_convention 抛 ValueError。"""
    with pytest.raises(ValueError, match="tide_convention"):
        GravityField("EARTH", tide_convention="weird")


def test_gravity_field_rejects_negative_degree():
    """degree 必须非负。"""
    with pytest.raises(ValueError, match="degree"):
        GravityField("EARTH", degree=-1)


def test_gravity_field_rejects_order_greater_than_degree():
    """order 不能超过 degree。"""
    with pytest.raises(ValueError, match="order"):
        GravityField("EARTH", degree=2, order=3)


def test_gravity_field_to_rust_spec_solid_and_pole_returns_none():
    """tide_mode='solid_and_pole' 暂不支持 Rust，to_rust_spec 返回 None。"""
    gf = GravityField("EARTH", degree=2, tide_mode="solid_and_pole")
    assert gf.to_rust_spec(None) is None


# ----------------------------------------------------------------------------
# 天体无关改造:按 body 切换 body-fixed frame 与默认重力文件
# ----------------------------------------------------------------------------

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "kernels"
)


class _SystemStub:
    """暴露 ``spice`` 与 ``coordinate_system`` 的最小系统桩,供坐标轴构造测试。"""

    def __init__(self, spice):
        self.spice = spice
        self.coordinate_system = None


@pytest.fixture
def body_frame_spice():
    """加载地月 SPICE 内核并返回 SPICEManager,测试后自动卸载。"""
    import spiceypy

    from e2m2e.data.kernels.manager import SPICEManager

    if not os.path.isdir(_KERNELS_DIR):
        pytest.skip("kernels directory not found")
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
    """GravityField 按 body 切换 body-fixed frame 与默认文件。"""

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

        ref = resources.files("e2m2e.algorithm.forces.data").joinpath("grgm900c.cof")
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
        from e2m2e.algorithm.coordinate.standard_axes import ITRFSpiceAxes

        gf = GravityField("EARTH", degree=2)
        cs = gf._get_input_coordinate_system(_SystemStub(body_frame_spice))
        assert isinstance(cs.axes, ITRFSpiceAxes)
        assert cs.axes.frame == "ITRF93"

    def test_moon_axes_use_moon_pa(self, body_frame_spice):
        """MOON 的 _get_input_coordinate_system 用 ITRFSpiceAxes(frame=MOON_PA)。"""
        from e2m2e.algorithm.coordinate.standard_axes import ITRFSpiceAxes

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
