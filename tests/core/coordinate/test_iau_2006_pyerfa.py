"""IAU 2006 简化岁差/章动矩阵与 pyerfa 黄金参考对比测试。

验证项目实现与 SOFA 官方参考值在 5e-5 rad 内一致。
"""

from __future__ import annotations

import erfa
import numpy as np
import pytest

from e2m2e.core.gmat_time import J2000_MJD
from e2m2e.core.iau_2006 import (
    iau2000eq_matrix,
    iau2000eq_true_matrix,
    nutation_angles,
    nutation_matrix,
    precession_angles,
    precession_matrix,
)
from e2m2e.core.standard_axes import ICRSAxes

# 单一 T 值:框架先立,后续要更严再加 T 扫描
T_TEST = 0.5
# T=0.5 时,et(秒) = T * 36525 * 86400
_SECONDS_PER_JULIAN_CENTURY = 36525.0 * 86400.0

# 容差(详见模块 docstring)
ANGLE_TOL = 5e-5      # 弧度,约 10 角秒
MATRIX_TOL = 5e-5
ORTHO_TOL = 1e-14

# MJD → JD 偏移(本地内联,gmat_time.py 未顶层暴露)
_JD_MJD_OFFSET = 2400000.5


def _tt_jd_from_et(et: float) -> tuple[float, float]:
    """把 et(TDB 秒)转成 TT 的两段儒略日,供 pyerfa 时间相关函数使用。

    et 是 SPICE 历书时(自 J2000.0 起算的 TDB 秒)。先用 J2000_MJD 拼出
    TDB 的 MJD,再调 erfa.tdbtt 得到两段 TT(MJD),最后加 JD_MJD_OFFSET
    转成 JD 形式喂给 pyerfa 的 p06e / nut06a / obl06。
    """
    tdb_mjd = J2000_MJD + et / 86400.0
    tt1_mjd, tt2_mjd = erfa.tdbtt(tdb_mjd, 0.0, 0.0)
    return tt1_mjd + _JD_MJD_OFFSET, tt2_mjd


def _rz(angle: float) -> np.ndarray:
    """绕 z 轴旋转矩阵(独立于项目 _rotation3)。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> np.ndarray:
    """绕 y 轴旋转矩阵(独立于项目 _rotation2)。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(angle: float) -> np.ndarray:
    """绕 x 轴旋转矩阵(独立于项目 _rotation1)。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ref_precession_matrix(t: float) -> np.ndarray:
    """用 pyerfa p06e 拼装参考岁差矩阵 P,独立于项目 precession_matrix 实现。

    公式与项目一致:P = R3(-z) @ R2(theta) @ R3(-zeta);但 zeta/theta/z
    直接从 pyerfa p06e 的 zetaa/thetaa/za 取,不复用项目 precession_angles。
    """
    et = t * _SECONDS_PER_JULIAN_CENTURY
    jd1, jd2 = _tt_jd_from_et(et)
    out = erfa.p06e(jd1, jd2)
    # p06e 索引:za=9, zetaa=10, thetaa=11
    zeta, theta, z = out[10], out[11], out[9]
    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


def _ref_nutation_matrix(t: float) -> np.ndarray:
    """用 pyerfa nut06a + obl06 拼装参考章动矩阵 N,独立于项目 nutation_matrix。

    公式与项目一致:N = R1(-(eps0+deps)) @ R3(-dpsi) @ R1(eps0);
    dpsi/deps 从 pyerfa nut06a 取,eps0 从 pyerfa obl06 取(不复用项目
    nutation_angles / mean_obliquity)。
    """
    et = t * _SECONDS_PER_JULIAN_CENTURY
    jd1, jd2 = _tt_jd_from_et(et)
    eps0 = erfa.obl06(jd1, jd2)
    dpsi, deps = erfa.nut06a(jd1, jd2)
    return _rx(-(eps0 + deps)) @ _rz(-dpsi) @ _rx(eps0)


class TestIcrsAxes:
    """ICRSAxes 锚点测试:作为框架入口,确认"恒等旋转"语义可被 pyerfa 风格对比。"""

    def test_rotation_matrix_is_identity_for_any_et(self):
        """ICRSAxes.rotation_matrix(et) 对任意 et 返回 np.eye(3)(恒等矩阵,1e-14 容差)。"""
        axes = ICRSAxes()
        for et in (0.0, 1.0, 86400.0, -86400.0):
            R = axes.rotation_matrix(et)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-14)


class TestPrecessionAngles:
    """项目 precession_angles(t) 与 pyerfa p06e 返回的 zetaa/thetaa/za 对比。

    pyerfa p06e 的 16 元素输出索引对应:
        0: eps0(平黄赤交角)
        9: za   10: zetaa   11: thetaa
    项目 precession_angles(t) 返回 (zeta, theta, z),对应 erfa zetaa/thetaa/za。
    """

    def test_precession_angles_match_pyerfa_p06e(self):
        """precession_angles(T_TEST) 与 erfa.p06e 的 zetaa/thetaa/za 元素差 < 5e-5 rad。"""
        et = T_TEST * _SECONDS_PER_JULIAN_CENTURY
        jd1, jd2 = _tt_jd_from_et(et)
        p06e_out = erfa.p06e(jd1, jd2)
        # p06e 索引:za=9, zetaa=10, thetaa=11
        ref_zeta, ref_theta, ref_z = p06e_out[10], p06e_out[11], p06e_out[9]

        zeta, theta, z = precession_angles(T_TEST)
        assert zeta == pytest.approx(ref_zeta, abs=ANGLE_TOL)
        assert theta == pytest.approx(ref_theta, abs=ANGLE_TOL)
        assert z == pytest.approx(ref_z, abs=ANGLE_TOL)


class TestNutationAngles:
    """项目 nutation_angles(t) 与 pyerfa nut06a 返回的 dpsi/deps 对比。

    项目 nutation_angles(t) 返回 (dpsi, deps, eps0),前两项为黄经章动 / 交角
    章动(弧度),第三项是平黄赤交角。pyerfa nut06a 返回 (dpsi, deps)(弧度)。
    """

    def test_nutation_angles_match_pyerfa_nut06a(self):
        """nutation_angles(T_TEST) 的 dpsi/deps 与 erfa.nut06a 差 < 5e-5 rad。"""
        et = T_TEST * _SECONDS_PER_JULIAN_CENTURY
        jd1, jd2 = _tt_jd_from_et(et)
        ref_dpsi, ref_deps = erfa.nut06a(jd1, jd2)

        dpsi, deps, _ = nutation_angles(T_TEST)
        assert dpsi == pytest.approx(ref_dpsi, abs=ANGLE_TOL)
        assert deps == pytest.approx(ref_deps, abs=ANGLE_TOL)


class TestPrecessionMatrix:
    """项目 precession_matrix(t) 与 pyerfa p06e 拼装的参考矩阵对比。"""

    def test_precession_matrix_matches_pyerfa_p06e_assembled(self):
        """precession_matrix(T_TEST) 与 pyerfa p06e 拼装的 P 元素差 < 5e-5。"""
        ref = _ref_precession_matrix(T_TEST)
        got = precession_matrix(T_TEST)
        np.testing.assert_allclose(got, ref, atol=MATRIX_TOL)


class TestNutationMatrix:
    """项目 nutation_matrix(t) 与 pyerfa nut06a + obl06 拼装的参考矩阵对比。"""

    def test_nutation_matrix_matches_pyerfa_nut06a_obl06_assembled(self):
        """nutation_matrix(T_TEST) 与 pyerfa 拼装的 N 元素差 < 5e-5。"""
        ref = _ref_nutation_matrix(T_TEST)
        got = nutation_matrix(T_TEST)
        np.testing.assert_allclose(got, ref, atol=MATRIX_TOL)


class TestIau2000EqMatrix:
    """iau2000eq_matrix(et) 与 precession_matrix(t) 等价性验证。

    项目实现里 iau2000eq_matrix 内部就是 precession_matrix(t),本质同函数,
    应严格相等(1e-14 容差)。
    """

    def test_iau2000eq_matrix_equivalent_to_precession_matrix(self):
        """iau2000eq_matrix(T_TEST * 36525 * 86400) ≡ precession_matrix(T_TEST)(1e-14)。"""
        et = T_TEST * _SECONDS_PER_JULIAN_CENTURY
        np.testing.assert_allclose(
            iau2000eq_matrix(et), precession_matrix(T_TEST), atol=1e-14
        )


class TestIau2000EqTrueMatrix:
    """iau2000eq_true_matrix(et) 与 pyerfa N @ P 合成参考对比。

    项目实现 iau2000eq_true_matrix 内部 = nutation_matrix(t) @ precession_matrix(t),
    pyerfa 侧用 _ref_nutation_matrix 与 _ref_precession_matrix 合成等价参考。
    """

    def test_iau2000eq_true_matrix_matches_pyerfa_n_times_p(self):
        """iau2000eq_true_matrix(T_TEST) 与 pyerfa N@P 合成元素差 < 5e-5。"""
        ref = _ref_nutation_matrix(T_TEST) @ _ref_precession_matrix(T_TEST)
        et = T_TEST * _SECONDS_PER_JULIAN_CENTURY
        got = iau2000eq_true_matrix(et)
        np.testing.assert_allclose(got, ref, atol=MATRIX_TOL)


class TestOrthogonality:
    """所有矩阵在 T_TEST 上满足 R @ R.T = I 且 det(R) = 1,容差 1e-14。

    正交性是纯双精度舍入下界,与参考源无关。
    """

    @pytest.mark.parametrize(
        "matrix_name",
        ["precession", "nutation", "iau2000eq", "iau2000eq_true"],
    )
    def test_matrix_orthogonal_at_T(self, matrix_name: str):
        """参数化 4 个矩阵,T=0.5 上 R @ R.T = I 与 det(R) = 1,容差 1e-14。"""
        et = T_TEST * _SECONDS_PER_JULIAN_CENTURY
        if matrix_name == "precession":
            R = precession_matrix(T_TEST)
        elif matrix_name == "nutation":
            R = nutation_matrix(T_TEST)
        elif matrix_name == "iau2000eq":
            R = iau2000eq_matrix(et)
        else:
            R = iau2000eq_true_matrix(et)

        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ORTHO_TOL)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=ORTHO_TOL)

