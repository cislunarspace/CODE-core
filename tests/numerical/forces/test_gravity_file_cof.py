"""GMAT ``.cof`` 重力场文件解析测试。

覆盖:
- 头行（POTFIELD）解析与单位换算（Mu m³/s²×1e9 -> km³/s², R m×1e3 -> km）。
- RECOEF 固定列解析（含 m=0 无 Snm 列、负值紧贴等边界）。
- 注释行（COMMENT / ``C ``）跳过、END 终止。
- 系数截断（requested_degree）。
- 统一入口 load_gravity_file 按扩展名分发。
- 包内 GRGM900C 月球模型的 GM / R_ref / max_degree / 系数核对。
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from e2m2e.algorithm.forces.gravity_file import (
    load_cof_file,
    load_gravity_file,
)

pytestmark = pytest.mark.force


# ----------------------------------------------------------------------------
# 包内数据路径
# ----------------------------------------------------------------------------


def _grgm900c_path() -> Path:
    ref = resources.files("e2m2e.algorithm.forces.data").joinpath("grgm900c.cof")
    return Path(ref)


# ----------------------------------------------------------------------------
# 合成 .cof 文件（用于边界测试，不依赖大文件）
# ----------------------------------------------------------------------------


def _rec_line(n: int, m: int, c: float, s: float | None = None) -> str:
    """构造一条列对齐的 RECOEF 行,字段宽度与真实 COF 一致。

    列布局(0-based):RECOEF(0-5), 空格(6-7), n(8-10,右对齐 3),
    m(11-13,右对齐 3), 空格(14-16), Cnm(17-37,21), Snm(38-58,21)。
    """
    line = "RECOEF  " + f"{n:3d}{m:3d}   " + f"{c:23.14e}"[-21:]
    if s is not None:
        line += f"{s:23.14e}"[-21:]
    return line


def _minimal_cof_content() -> str:
    """一个最小但列对齐完整的 COF 文件。

    POTFIELD 头：NNN=005, MMM=005, flag=0, Mu=4.9028e12 (m³/s²×1e9),
    RefRadius=1.738e6 (m×1e3), Normalized=1.0。
    系数：含 C00、C10、C20、C21、C22（m=0/1/2 各一例）。
    """
    lines = [
        "COMMENT   4",
        "C 这是注释行",
        "POTFIELD005005  0 4.90280000000000e+12 1.73800000000000e+06 1.00000000000000e+00",
        _rec_line(0, 0, 1.0),
        _rec_line(1, 0, 0.0),
        _rec_line(2, 0, -9.08866163613439e-05, 0.0),
        _rec_line(2, 1, -1.94959323612596e-10, 1.07333253873312e-09),
        _rec_line(2, 2, 3.46733340854267e-05, 1.09084628698355e-10),
        _rec_line(3, 3, 1.22753848162928e-05, -1.77415609631060e-06),
        "END",
    ]
    return "\n".join(lines) + "\n"


def test_load_cof_file_parses_header_units(tmp_path):
    """头行 Mu/RefRadius 单位换算：Mu 除 1e9 得 km³/s², R 除 1e3 得 km。"""
    path = tmp_path / "mini.cof"
    path.write_text(_minimal_cof_content())

    data = load_cof_file(path)

    assert data.model_name == "mini"
    # 4.9028e12 / 1e9 = 4902.8 km³/s²
    assert data.mu == pytest.approx(4902.8)
    # 1.738e6 / 1e3 = 1738.0 km
    assert data.radius == pytest.approx(1738.0)
    assert data.max_degree == 5
    assert data.normalized is True


def test_load_cof_file_parses_coefficients_fixed_columns(tmp_path):
    """RECOEF 固定列解析：m=0 无 Snm 列、m>0 有 Snm、负值紧贴也正确。"""
    path = tmp_path / "mini.cof"
    path.write_text(_minimal_cof_content())

    data = load_cof_file(path)

    # C00 = 1.0
    np.testing.assert_allclose(data.C[0, 0], 1.0)
    # C20 (m=0, 无 Snm)
    np.testing.assert_allclose(data.C[2, 0], -9.08866163613439e-05)
    np.testing.assert_allclose(data.S[2, 0], 0.0)
    # C21 / S21
    np.testing.assert_allclose(data.C[2, 1], -1.94959323612596e-10)
    np.testing.assert_allclose(data.S[2, 1], 1.07333253873312e-09)
    # C22 / S22
    np.testing.assert_allclose(data.C[2, 2], 3.46733340854267e-05)
    np.testing.assert_allclose(data.S[2, 2], 1.09084628698355e-10)
    # C33 / S33 — S 为负值且与 C 紧贴（无空格）, 验证列边界
    np.testing.assert_allclose(data.C[3, 3], 1.22753848162928e-05)
    np.testing.assert_allclose(data.S[3, 3], -1.77415609631060e-06)


def test_load_cof_file_injects_c00_when_missing(tmp_path):
    """COF 通常省略 C00, 解析器补 1.0。"""
    content = (
        "POTFIELD002002  0 4.90280000000000e+12 1.73800000000000e+06 1.00000000000000e+00\n"
        + _rec_line(2, 0, -9.08866163613439e-05, 0.0)
        + "END\n"
    )
    path = tmp_path / "noc00.cof"
    path.write_text(content)

    data = load_cof_file(path)

    np.testing.assert_allclose(data.C[0, 0], 1.0)
    np.testing.assert_allclose(data.S[0, 0], 0.0)


def test_load_cof_file_dot_coefficients_all_zero(tmp_path):
    """COF 不含 dot 项, dotC/dotS 全零且形状与 C/S 一致。"""
    path = tmp_path / "mini.cof"
    path.write_text(_minimal_cof_content())

    data = load_cof_file(path)

    assert data.dotC.shape == data.C.shape
    assert data.dotS.shape == data.S.shape
    np.testing.assert_array_equal(data.dotC, 0.0)
    np.testing.assert_array_equal(data.dotS, 0.0)


def test_load_cof_file_truncates_to_requested_degree(tmp_path):
    """requested_degree 截断：只读指定阶次以下的系数。"""
    path = tmp_path / "mini.cof"
    path.write_text(_minimal_cof_content())

    data = load_cof_file(path, requested_degree=2)

    assert data.C.shape == (3, 3)
    # max_degree 仍为文件声明的 5
    assert data.max_degree == 5
    # C33 在 n=3, 被截断后应不在数组中
    np.testing.assert_allclose(data.C[2, 2], 3.46733340854267e-05)


def test_load_cof_file_rejects_degree_exceeds_max(tmp_path):
    """requested_degree 超过文件 max_degree 时抛 ValueError。"""
    path = tmp_path / "mini.cof"
    path.write_text(_minimal_cof_content())

    with pytest.raises(ValueError, match="max_degree"):
        load_cof_file(path, requested_degree=10)


def test_load_cof_file_rejects_missing_potfield_header(tmp_path):
    """缺 POTFIELD 头行抛 ValueError。"""
    path = tmp_path / "bad.cof"
    path.write_text(_rec_line(2, 0, -9.0e-05, 0.0) + "END\n")

    with pytest.raises(ValueError, match="POTFIELD"):
        load_cof_file(path)


def test_load_cof_file_handles_unnormalized_flag(tmp_path):
    """Normalized=0.0 时 normalized=False。"""
    content = (
        "POTFIELD002002  0 4.90280000000000e+12 1.73800000000000e+06 0.00000000000000e+00\n"
        + _rec_line(2, 0, -9.08866163613439e-05, 0.0)
        + "END\n"
    )
    path = tmp_path / "unnorm.cof"
    path.write_text(content)

    data = load_cof_file(path)

    assert data.normalized is False


def test_load_cof_file_skips_comment_lines(tmp_path):
    """COMMENT 与 'C ' 开头的注释行被跳过, 不影响解析。"""
    content = (
        "COMMENT   This is a comment\n"
        "C Another comment line\n"
        "COMMENT\n"
        "POTFIELD002002  0 4.90280000000000e+12 1.73800000000000e+06 1.0\n"
        + _rec_line(2, 0, -9.0e-05, 0.0)
        + "END\n"
    )
    path = tmp_path / "comments.cof"
    path.write_text(content)

    data = load_cof_file(path)

    np.testing.assert_allclose(data.C[2, 0], -9.0e-05)


def test_load_gravity_file_dispatches_by_extension(tmp_path):
    """load_gravity_file 按扩展名分发: .cof -> load_cof_file。"""
    path = tmp_path / "dispatch.cof"
    path.write_text(_minimal_cof_content())

    via_dispatch = load_gravity_file(path)
    direct = load_cof_file(path)

    assert via_dispatch.model_name == direct.model_name
    assert via_dispatch.mu == direct.mu
    assert via_dispatch.max_degree == direct.max_degree
    np.testing.assert_allclose(via_dispatch.C, direct.C)


def test_load_gravity_file_rejects_unknown_extension(tmp_path):
    """未知扩展名抛 ValueError。"""
    path = tmp_path / "data.txt"
    path.write_text("garbage")

    with pytest.raises(ValueError, match="Unsupported gravity file extension"):
        load_gravity_file(path)


# ----------------------------------------------------------------------------
# 包内 GRGM900C 月球模型核对
# ----------------------------------------------------------------------------


def test_grgm900c_header_gm_radius_degree():
    """GRGM900C: GM≈4902.8 km³/s², R_ref=1738.0 km, max_degree=360, normalized。"""
    data = load_cof_file(_grgm900c_path())

    assert data.mu == pytest.approx(4902.8, rel=1e-4)
    assert data.radius == pytest.approx(1738.0)
    assert data.max_degree == 360
    assert data.normalized is True
    assert data.C.shape == (361, 361)


def test_grgm900c_coefficients_match_file():
    """GRGM900C 若干系数与文件原始值一致（C20 ≈ -9.09e-5）。"""
    data = load_cof_file(_grgm900c_path())

    np.testing.assert_allclose(data.C[0, 0], 1.0)
    np.testing.assert_allclose(data.C[2, 0], -9.08866163613439e-05)
    np.testing.assert_allclose(data.S[2, 1], 1.07333253873312e-09)
    np.testing.assert_allclose(data.C[2, 2], 3.46733340854267e-05)
    # 边界:最高阶项
    np.testing.assert_allclose(data.C[360, 360], 1.32401674690313e-09)
    np.testing.assert_allclose(data.S[360, 360], 3.00216071582712e-09)


def test_grgm900c_truncates_to_requested_degree():
    """GRGM900C requested_degree=10 时只读 10×10。"""
    data = load_cof_file(_grgm900c_path(), requested_degree=10)

    assert data.C.shape == (11, 11)
    assert data.max_degree == 360  # 文件声明的最大阶
    # n<=10 的系数仍存在
    np.testing.assert_allclose(data.C[2, 0], -9.08866163613439e-05)
