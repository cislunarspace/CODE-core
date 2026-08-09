"""``qf_projection.project_hamiltonian_to_qf`` 测试。

H→QF 映射：把平动点偏移坐标的 Hamiltonian 经 ``X = B·Y`` 投影到 QF 坐标，
对应 qiao Code09。核心验收：单位 B 投影不变、变量换位正确、输出系数
长度与 tlist 一致。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("sympy")

from e2m2e.algorithm.normal_form.qf_projection import project_hamiltonian_to_qf

pytestmark = pytest.mark.theory


def _make_qf_result(B_samples: np.ndarray, tlist: np.ndarray) -> MagicMock:
    """构造一个提供 ``B(t)`` 与 ``tlist`` 的 mock QF result。"""
    qf = MagicMock()
    qf.tlist = tlist
    # B(t) 线性插值访问器
    B_arr = np.asarray(B_samples, dtype=float)

    def B_at(t: float) -> np.ndarray:
        t_arr = np.asarray(tlist, dtype=float)
        if t_arr.size == 1:
            return B_arr[0]
        flat = B_arr.reshape(t_arr.size, 36)
        out = np.empty(36)
        for k in range(36):
            out[k] = np.interp(t, t_arr, flat[:, k])
        return out.reshape(6, 6)

    qf.B = B_at
    return qf


class TestProjectHamiltonianToQf:
    """``project_hamiltonian_to_qf`` 的投影语义。"""

    def test_identity_B_preserves_hamiltonian(self):
        """单位 B(t)=I：投影后的多项式应与输入一致（仅 ≥3 阶项）。"""
        tlist = np.array([0.0, 0.5, 1.0])
        B_samples = np.stack([np.eye(6)] * 3)  # 恒等
        qf = _make_qf_result(B_samples, tlist)

        H = {(3, 0, 0, 0, 0, 0): 2.0, (2, 0, 0, 0, 0, 0): 9.0}  # 含一个二阶项
        result = project_hamiltonian_to_qf(H, qf)

        # 二阶项不投影（CM reducer 自加实标准形）
        assert (2, 0, 0, 0, 0, 0) not in result
        # 3 阶项经单位 B 不变
        assert (3, 0, 0, 0, 0, 0) in result
        coef_arr = np.asarray(result[(3, 0, 0, 0, 0, 0)])
        assert coef_arr.shape == (3,)
        np.testing.assert_allclose(coef_arr, 2.0, atol=1e-12)

    def test_permutation_B_swaps_variables(self):
        """置换 B（x1↔x2）：``x1^3`` → ``x2^3``，幂次换位。"""
        tlist = np.array([0.0])
        B = np.eye(6)
        B[[0, 1]] = B[[1, 0]]  # 交换前两行：x1=B·y 中 x1=y2, x2=y1
        qf = _make_qf_result(B[np.newaxis], tlist)

        H = {(3, 0, 0, 0, 0, 0): 1.0}  # x1^3
        result = project_hamiltonian_to_qf(H, qf)
        # x1^3 → y2^3（B 第1行=[0,1,0,...] => x1=y2），即新 (0,3,0,0,0,0)
        assert (0, 3, 0, 0, 0, 0) in result
        assert abs(float(result[(0, 3, 0, 0, 0, 0)][0]) - 1.0) < 1e-12

    def test_coefficient_length_matches_tlist(self):
        """输出系数长度应与 ``qf_result.tlist`` 一致。"""
        tlist = np.linspace(0, 2 * np.pi, 50)
        # 时变 B：B(t) = I + ε·sin(t)·E_{12}（小幅时变）
        eps = 0.01
        B_samples = np.stack([np.eye(6) for _ in tlist])
        for i, t in enumerate(tlist):
            B_samples[i, 0, 1] = eps * np.sin(t)
        qf = _make_qf_result(B_samples, tlist)

        H = {(3, 0, 0, 0, 0, 0): 1.0}
        result = project_hamiltonian_to_qf(H, qf)
        for pow_tuple, arr in result.items():
            arr = np.asarray(arr)
            assert arr.shape == (50,), f"pow={pow_tuple} 长度 {arr.shape}≠(50,)"

    def test_drops_below_third_order(self):
        """阶 < 3 的项一律丢弃（CM reducer 自加二阶实标准形）。"""
        tlist = np.array([0.0])
        qf = _make_qf_result(np.eye(6)[np.newaxis], tlist)
        H = {
            (0, 0, 0, 0, 0, 0): 5.0,  # 0 阶
            (1, 0, 0, 0, 0, 0): 3.0,  # 1 阶
            (2, 0, 0, 0, 0, 0): 7.0,  # 2 阶
            (3, 0, 0, 0, 0, 0): 1.0,  # 3 阶
        }
        result = project_hamiltonian_to_qf(H, qf)
        # 只剩 3 阶
        for pow_tuple in result:
            assert sum(pow_tuple) >= 3

    def test_scaling_B_scales_coefficient(self):
        """B 缩放 α：``x1^3`` 经 ``x1=α·y1`` → ``α^3·y1^3``。"""
        tlist = np.array([0.0])
        alpha = 2.0
        B = alpha * np.eye(6)
        qf = _make_qf_result(B[np.newaxis], tlist)

        H = {(3, 0, 0, 0, 0, 0): 1.0}
        result = project_hamiltonian_to_qf(H, qf)
        assert abs(float(result[(3, 0, 0, 0, 0, 0)][0]) - alpha**3) < 1e-10
