"""重力场文件解析测试。

覆盖 .gfc 头/系数读取、dot 行、默认 GM/R 与异常处理。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces.gravity_file import load_gfc_file


def _minimal_gfc_content():
    return """modelname EGM96_TEST
earth_gravity_constant 398600.441500000
radius 6378.136300000
max_degree 2
norm fully_normalized
gfc 0 0 1.000000000000e+00 0.000000000000e+00
gfc 2 0 -4.841651437908e-04 0.000000000000e+00
gfc 2 1 -1.869876401566e-10 1.195280120309e-09
gfc 2 2 2.439143523982e-06 -1.400166836544e-06
END"""


def test_load_gfc_file_parses_header_and_coefficients(tmp_path):
    """解析器能读取 .gfc 文件头和系数。"""
    path = tmp_path / "test.gfc"
    path.write_text(_minimal_gfc_content())

    data = load_gfc_file(path)

    assert data.model_name == "EGM96_TEST"
    assert data.mu == pytest.approx(398600.4415)
    assert data.radius == pytest.approx(6378.1363)
    assert data.max_degree == 2
    assert data.normalized is True
    np.testing.assert_allclose(data.C[0, 0], 1.0)
    np.testing.assert_allclose(data.C[2, 0], -4.841651437908e-04)
    np.testing.assert_allclose(data.S[2, 2], -1.400166836544e-06)


def test_load_gfc_file_uses_default_mu_and_radius_when_missing(tmp_path):
    """缺失 GM/R 时使用默认值。"""
    content = """modelname MINIMAL
max_degree 0
gfc 0 0 1.0 0.0
END"""
    path = tmp_path / "minimal.gfc"
    path.write_text(content)

    data = load_gfc_file(path)

    assert data.mu == pytest.approx(398600.4415)
    assert data.radius == pytest.approx(6378.1363)


def test_load_gfc_file_rejects_unknown_norm(tmp_path):
    """无法识别的 norm 字段抛 ValueError。"""
    content = """modelname BAD
norm weird_format
gfc 0 0 1.0 0.0
END"""
    path = tmp_path / "bad.gfc"
    path.write_text(content)

    with pytest.raises(ValueError, match="norm"):
        load_gfc_file(path)


def test_load_gfc_file_rejects_degree_exceeds_max(tmp_path):
    """请求 degree 超过文件 max_degree 时抛 ValueError。"""
    content = """modelname SMALL
max_degree 1
gfc 0 0 1.0 0.0
END"""
    path = tmp_path / "small.gfc"
    path.write_text(content)

    with pytest.raises(ValueError, match="max_degree"):
        load_gfc_file(path, requested_degree=2)


def _gfc_content_with_dot_lines():
    return """modelname DOT_TEST
earth_gravity_constant 398600.441500000
radius 6378.136300000
max_degree 2
norm fully_normalized
gfc 0 0 1.000000000000e+00 0.000000000000e+00
gfc 2 0 -4.841651437908e-04 0.000000000000e+00
gfc 2 1 -1.869876401566e-10 1.195280120309e-09
gfc 2 2 2.439143523982e-06 -1.400166836544e-06
dot 2 0 1.162755e-11 0.000000000000e+00
dot 2 1 -7.000000e-12 1.800000e-11
END"""


def test_load_gfc_file_parses_dot_lines(tmp_path):
    """解析器能读取 dot 行(系数长期变化率,ICGEM 格式 dot n m dotC dotS)。"""
    path = tmp_path / "dot.gfc"
    path.write_text(_gfc_content_with_dot_lines())

    data = load_gfc_file(path)

    np.testing.assert_allclose(data.dotC[2, 0], 1.162755e-11)
    np.testing.assert_allclose(data.dotS[2, 1], 1.800000e-11)
    # 无 dot 行的项为零
    assert data.dotC[0, 0] == 0.0
    assert data.dotC[2, 2] == 0.0


def test_load_gfc_file_dot_zero_when_no_dot_lines(tmp_path):
    """无 dot 行时 dotC/dotS 全零(向后兼容现有 .gfc)。"""
    path = tmp_path / "nodot.gfc"
    path.write_text(_minimal_gfc_content())

    data = load_gfc_file(path)

    assert data.dotC.shape == data.C.shape
    assert data.dotS.shape == data.S.shape
    np.testing.assert_array_equal(data.dotC, 0.0)
    np.testing.assert_array_equal(data.dotS, 0.0)


def test_load_gfc_file_dot_truncated_to_requested_degree(tmp_path):
    """dot 行超过 requested_degree 时被截断,与 gfc 行一致。"""
    path = tmp_path / "dot.gfc"
    path.write_text(_gfc_content_with_dot_lines())

    data = load_gfc_file(path, requested_degree=0)

    assert data.dotC.shape == (1, 1)
    assert data.dotC[0, 0] == 0.0


# ----------------------------------------------------------------------------
# dot 项历元外推（Slice 2 / AC4）
# ----------------------------------------------------------------------------

_SECONDS_PER_YEAR = 365.25 * 86400.0


def test_extrapolate_coefficients_linear_drift():
    """dot 历元外推:C_nm(t) = C_nm(t0) + dotC_nm * (t-t0),时间单位转年。"""
    from e2m2e.algorithm.forces.gravity_file import extrapolate_coefficients

    C = np.array([[1.0, 0.0], [0.0, 0.0], [-4.84e-4, 0.0]])
    S = np.zeros((3, 2))
    dotC = np.array([[0.0, 0.0], [0.0, 0.0], [1.0e-11, 0.0]])
    dotS = np.zeros((3, 2))

    # t - t0 = 1 年
    C_out, _ = extrapolate_coefficients(C, S, dotC, dotS, t=_SECONDS_PER_YEAR, t0=0.0)

    np.testing.assert_allclose(C_out[2, 0], -4.84e-4 + 1.0e-11, rtol=1e-12)
    # 未给 dot 的项不变
    np.testing.assert_allclose(C_out[0, 0], 1.0)


def test_extrapolate_coefficients_no_change_at_reference_epoch():
    """t == t0 时系数不变(外推回到参考历元)。"""
    from e2m2e.algorithm.forces.gravity_file import extrapolate_coefficients

    C = np.array([[1.0, 0.0], [0.0, 0.0], [-4.84e-4, 0.0]])
    S = np.zeros((3, 2))
    dotC = np.array([[0.0, 0.0], [0.0, 0.0], [1.0e-11, 0.0]])
    dotS = np.zeros((3, 2))

    C_out, S_out = extrapolate_coefficients(C, S, dotC, dotS, t=1000.0, t0=1000.0)

    np.testing.assert_array_equal(C_out, C)
    np.testing.assert_array_equal(S_out, S)


def test_extrapolate_c20_dot_sign_and_magnitude_from_j2_dot():
    """AC4 核心厘清:J2_dot 与 C20_dot 的换算。

    C20 = -J2/sqrt(5)(正规化),故 C20_dot = -J2_dot/sqrt(5)。
    J2_dot ≈ -1.6e-11/yr(下降)→ C20_dot ≈ +7.2e-12/yr(上升,符号反转)。
    测试验证符号与量级,不混用 J2_dot 与 C20_dot。
    """
    from e2m2e.algorithm.forces.gravity_file import extrapolate_coefficients

    sqrt5 = np.sqrt(5.0)
    J2 = 1.0826e-3
    C20 = -J2 / sqrt5
    J2_dot = -1.6e-11  # /yr(非正规化 J2 的漂移,文献参考值)
    C20_dot = -J2_dot / sqrt5  # 正规化 C20 的漂移

    C = np.zeros((3, 3))
    C[0, 0] = 1.0
    C[2, 0] = C20
    S = np.zeros((3, 3))
    dotC = np.zeros((3, 3))
    dotC[2, 0] = C20_dot
    dotS = np.zeros((3, 3))

    C_out, _ = extrapolate_coefficients(C, S, dotC, dotS, t=_SECONDS_PER_YEAR, t0=0.0)

    # 1 年后 C20 增加 C20_dot
    np.testing.assert_allclose(C_out[2, 0], C20 + C20_dot, rtol=1e-12)
    # 符号:J2_dot<0 → C20_dot>0 → C20 上升
    assert C20_dot > 0.0, "J2_dot<0 应对应 C20_dot>0(符号反转)"
    assert C_out[2, 0] > C20
    # 量级:C20_dot ~ 7e-12/yr(不是 J2_dot 的 1.6e-11)
    assert 5e-12 < C20_dot < 1e-11
