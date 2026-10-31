"""中心乘子对选取的合成谱测试。

按 ADR 0055 口径，被测对象是纯函数（单值矩阵与流方向），断言按特征结构
定义（单位模、辐角、法向支撑），全部用合成矩阵，不做动力学传播。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.torus import TorusError, central_multiplier_pair

pytestmark = pytest.mark.theory

#: 流方向取平凡若尔当块的特征向量（x 方向单位向量）。
_FLOW = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _planar_matrix(alpha: float, hyperbolic_log: float) -> np.ndarray:
    """平面周期轨道形态的单值矩阵。

    分量序 (x, y, z, vx, vy, vz)：面内块 (x, y, vx, vy) 含平凡若尔对与
    双曲对，法向块 (z, vz) 为转角 alpha 的旋转。
    """
    matrix = np.eye(6)
    matrix[0, 3] = 1.0  # 平凡若尔对（流方向特征值为 1）
    matrix[1, 1] = np.exp(hyperbolic_log)
    matrix[4, 4] = np.exp(-hyperbolic_log)
    cos, sin = np.cos(alpha), np.sin(alpha)
    matrix[2, 2] = cos
    matrix[2, 5] = sin
    matrix[5, 2] = -sin
    matrix[5, 5] = cos
    return matrix


def _vertical_hyperbolic_matrix(hyperbolic_log: float) -> np.ndarray:
    """面内双曲且垂直双曲的平面轨道形态单值矩阵（无任何中心对）。"""
    matrix = np.eye(6)
    matrix[0, 3] = 1.0
    matrix[1, 1] = np.exp(hyperbolic_log)
    matrix[4, 4] = np.exp(-hyperbolic_log)
    matrix[2, 2] = np.exp(hyperbolic_log)
    matrix[5, 5] = np.exp(-hyperbolic_log)
    return matrix


class TestCentralMultiplierPair:
    def test_planar_orbit_selects_normal_pair(self) -> None:
        """平面轨道：在法向块旋转与面内双曲对并存时选法向单位模对。"""
        alpha = 1.234
        pair = central_multiplier_pair(_planar_matrix(alpha, 0.7), _FLOW)
        assert pair.alpha == pytest.approx(alpha, abs=1e-12)
        assert pair.eigenvalue == pytest.approx(np.exp(1j * alpha), abs=1e-12)
        # 种子基向量集中在 z、vz 分量上
        for vector in (pair.a, pair.b):
            support = float(np.linalg.norm(vector[[2, 5]])) / float(np.linalg.norm(vector))
            assert support > 0.999

    def test_two_center_pairs_decoupled_still_pick_normal(self) -> None:
        """面内也是中心对（如 2:1 附近的 DRO）时，按法向支撑唯一判定。"""
        alpha = 1.5
        matrix = _planar_matrix(alpha, 0.0)
        cos, sin = np.cos(0.8), np.sin(0.8)
        matrix[1, 1] = cos
        matrix[1, 4] = sin
        matrix[4, 1] = -sin
        matrix[4, 4] = cos
        pair = central_multiplier_pair(matrix, _FLOW)
        assert pair.alpha == pytest.approx(alpha, abs=1e-12)

    def test_no_unit_modulus_pair_raises(self) -> None:
        """无非平凡单位模乘子对（垂直双曲的平面轨道）确定性报错。"""
        with pytest.raises(TorusError, match="单位模"):
            central_multiplier_pair(_vertical_hyperbolic_matrix(0.9), _FLOW)

    def test_ambiguous_coupled_center_pairs_raise(self) -> None:
        """两个中心对跨块耦合（无平面分块结构）时不可判，确定性报错。"""
        rng = np.random.default_rng(842)
        mix = np.linalg.qr(rng.standard_normal((4, 4)))[0]
        block = np.zeros((4, 4))
        for angle, base in ((1.1, 0), (2.0, 2)):
            cos, sin = np.cos(angle), np.sin(angle)
            block[base : base + 2, base : base + 2] = [[cos, -sin], [sin, cos]]
        coupled = mix @ block @ mix.T
        matrix = np.eye(6)
        matrix[0, 3] = 1.0
        indices = (1, 2, 4, 5)
        matrix[np.ix_(indices, indices)] = coupled
        with pytest.raises(TorusError, match="判别"):
            central_multiplier_pair(matrix, _FLOW)
