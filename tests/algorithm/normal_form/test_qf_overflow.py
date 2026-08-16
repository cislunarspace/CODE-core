"""``_solve_qf_matrix`` 的数值稳定性测试（分段辛重投影）。

QF 步 ``Ḃ = M·B − B·D`` 单次 DOP853 积分沿双曲方向 ``e^(λt)`` 无界增长，
长窗口（``λT > 10``）数值 overflow、辛性破坏。本测试覆盖分段辛重投影
修复：把积分区间按 ``segment`` 分短段，每段末用 ``symplectic_project``
把 ``B`` 拉回辛群，抑制误差累积。

参考 qiao ``Code08_QuasiFloquet.m`` 的多点打靶思想（短弧 STM + 节点辛投影），
但此处用最小改动（分段积分 + 投影）而非完整块三对角牛顿法。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.normal_form import NormalFormContext
from e2m2e.algorithm.normal_form.quasi_floquet import (
    J6,
    _cr3bp_hessian_symmetric,
    _qf_rhs_factory,
    _solve_qf_matrix,
    real_normal_form_matrix,
)
from e2m2e.data.templates.enums import LibrationPoint

pytestmark = pytest.mark.theory


@pytest.fixture
def l2_M_D():
    """L2 平动点的常数 Hamilton 矩阵 M 与实标准形 D（CR3BP 线性化）。"""
    sys_em = CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")
    sys_em.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    ctx = NormalFormContext(
        system=sys_em, libration_point=LibrationPoint.L2, epoch=2451545.0, order=4
    )
    mu = float(ctx.mu)
    r_lp = np.asarray(ctx.libration_position, dtype=float)
    S = _cr3bp_hessian_symmetric(r_lp, mu)
    omega_x = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    H_block = np.block([[S, omega_x], [-omega_x, np.zeros((3, 3))]])
    M = J6 @ H_block
    nu1, nu2 = ctx.central_frequencies
    lam = float(ctx.characteristic_exponent)
    D = real_normal_form_matrix(lam, float(nu1), float(nu2))

    def M_at(t: float) -> np.ndarray:
        return M

    return M_at, D, lam


def _max_symplectic_error(B_samples: np.ndarray) -> float:
    """所有采样点上 ``‖BᵀJB − J‖∞`` 的最大值。"""
    err = 0.0
    for B in B_samples:
        e = float(np.max(np.abs(B.T @ J6 @ B - J6)))
        if e > err:
            err = e
    return err


class TestSolveQfMatrixStability:
    """分段辛重投影的数值稳定性。"""

    def test_short_window_single_integral_ok(self, l2_M_D):
        """短窗口（T=2 TU）：单次积分（segment=None）辛误差应很小。"""
        M_at, D, lam = l2_M_D
        tlist = np.linspace(0, 2.0, 50)
        B = _solve_qf_matrix(M_at, D, tlist, segment=None)
        assert _max_symplectic_error(B) < 1e-6

    def test_short_window_segmented_matches_single(self, l2_M_D):
        """短窗口下，分段（segment=0.4）与单次积分结果应一致。"""
        M_at, D, lam = l2_M_D
        tlist = np.linspace(0, 2.0, 50)
        B_single = _solve_qf_matrix(M_at, D, tlist, segment=None)
        B_seg = _solve_qf_matrix(M_at, D, tlist, segment=0.4)
        # 短窗口两者都辛，数值应接近（分段引入小幅投影差异）
        diff = np.max(np.abs(B_single - B_seg))
        assert diff < 1e-3, f"短窗口分段与单积分差异过大：{diff}"

    def test_medium_window_segmented_better_symplectic(self, l2_M_D):
        """中窗口（T≈10 TU，e^λT≈3e9）：分段辛误差应显著小于单积分。

        单次积分在 ``λT≈20`` 时辛误差 ~1e-2（辛性已破坏）；
        分段 + 每段投影把辛误差控制在 ~1e-2 以下（受 symplectic_project
        精度与 ``e^(λt)`` 增长双重限制）。这是完整多点打靶（issue #328）
        落地前的最小可行修复——端到端 Lissajous 有界性的实际效果由
        ``test_pipeline_lissajous`` 验证。
        """
        M_at, D, lam = l2_M_D
        tlist = np.linspace(0, 10.0, 100)
        B_seg = _solve_qf_matrix(M_at, D, tlist, segment=0.4)
        err_seg = _max_symplectic_error(B_seg)
        # 分段应优于单积分（单积分此处 ~1e-1）
        B_single = _solve_qf_matrix(M_at, D, tlist, segment=None)
        err_single = _max_symplectic_error(B_single)
        assert err_seg < err_single, f"分段未优于单积分：{err_seg} vs {err_single}"
        assert err_seg < 2e-2, f"分段辛误差过大：{err_seg}"

    def test_segmented_satisfies_ode_within_segment(self, l2_M_D):
        """分段结果在段内部应满足 ``Ḃ = M·B − B·D``（段内 ODE 残差小）。

        段边界处 B 因辛投影有跳变，故只在段内部（避开 ``segment`` 整数倍点）
        验 ODE 残差。
        """
        M_at, D, lam = l2_M_D
        segment = 0.4
        tlist = np.linspace(0, 6.0, 300)  # 密网格使段内有足够点
        B_samples = _solve_qf_matrix(M_at, D, tlist, segment=segment)
        rhs = _qf_rhs_factory(M_at, D)

        # 段边界（segment 整数倍）附近的点跳过
        def _near_boundary(t: float) -> bool:
            frac = (t - tlist[0]) / segment
            return abs(frac - round(frac)) < 0.05

        max_residual = 0.0
        for i in range(2, len(tlist) - 2):
            if _near_boundary(tlist[i]):
                continue
            h = tlist[i + 1] - tlist[i]
            Bdot_num = (B_samples[i + 1] - B_samples[i - 1]) / (2 * h)
            Bdot_ana = rhs(tlist[i], B_samples[i]).reshape(6, 6)
            scale = np.max(np.abs(Bdot_ana)) + 1e-30
            max_residual = max(max_residual, np.max(np.abs(Bdot_num - Bdot_ana)) / scale)
        assert max_residual < 2e-3, f"段内 ODE 相对残差过大：{max_residual}"
