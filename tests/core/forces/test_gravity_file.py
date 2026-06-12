"""重力场文件解析测试。"""

import io

import numpy as np
import pytest

from e2m2e.core.forces.gravity_file import GravityFileData, load_gfc_file


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
