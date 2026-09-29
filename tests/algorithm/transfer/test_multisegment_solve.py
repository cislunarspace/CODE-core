"""multisegment 求解档（solve_square / solve_minimax）的机制测试（#726）。

oracle 口径（ADR 0055）：

- ① 定义性公式：方系统取已知闭式根（Vieta 关系复算）；minimax 取线性
  Chebyshev 问题的等幅振荡特征（Haar 条件下最优解 n+1 个残差双号达到
  最大模）；
- ③ 退化：初值即根零迭代；零雅可比报 ``SINGULAR_JACOBIAN`` 而非崩溃；
- ④ 机制：Broyden 秩一更新使全量雅可比评估次数远少于迭代数。

⑦ 类文献结果性数值禁入 CI 断言。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.algorithm.transfer import (
    ChainLegRequest,
    LegKernel,
    evaluate_chain,
    solve_minimax,
    solve_square,
)
from e2m2e.integrators import propagate_kepler_py
from e2m2e.status import ConvergenceState, FailureCause

pytestmark = pytest.mark.theory

MU = 398600.435507  # 地球 GM（km³/s²），与 conic 档测试同源的固定值


def _vieta_system():
    """二元系统 [x0+x1−3, x0·x1−2]：闭式根 (1,2)/(2,1)。"""

    def defect(x):
        return np.array([x[0] + x[1] - 3.0, x[0] * x[1] - 2.0])

    def jac(x):
        return np.array([[1.0, 1.0], [x[1], x[0]]])

    return defect, jac


class TestSolveSquare:
    def test_closed_form_root(self):
        """① 已知闭式根的非线性方系统收敛，根满足 Vieta 关系。"""
        defect, jac = _vieta_system()
        result = solve_square(defect, jac, [0.3, 1.6], tol=1e-12)
        assert result.status is ConvergenceState.CONVERGED
        assert result.cause is FailureCause.NONE
        assert result.defect_norm < 1e-12
        # 闭式吻合：sum=3、prod=2（两根之一）。
        assert result.x[0] + result.x[1] == pytest.approx(3.0, abs=1e-10)
        assert result.x[0] * result.x[1] == pytest.approx(2.0, abs=1e-10)

    def test_broyden_mechanism_saves_full_jacobians(self):
        """④ Broyden 机制：全量雅可比次数 ≪ 迭代数。

        6 维温和非线性系统（x·roll(x) − b + 0.05x²），full_jac_every=8：
        实测 11 次迭代只重算 2 次全量 J。
        """
        n = 6
        rng = np.random.default_rng(7)
        b = rng.uniform(0.5, 1.5, n)

        def defect(x):
            return x * np.roll(x, -1) - b + 0.05 * x**2

        def jac(x):
            matrix = np.zeros((n, n))
            for i in range(n):
                matrix[i, i] = np.roll(x, -1)[i] + 0.1 * x[i]
                matrix[i, (i + 1) % n] = x[i]
            return matrix

        result = solve_square(defect, jac, np.full(n, 0.3), full_jac_every=8)
        assert result.status is ConvergenceState.CONVERGED, result.message
        assert result.defect_norm < 1e-10
        assert result.n_iter >= 6, f"系统须足够非线性以考察 Broyden 机制，n_iter={result.n_iter}"
        assert result.n_jac_evals <= 1 + math.ceil(result.n_iter / 8) + 2
        assert result.n_jac_evals < result.n_iter, (
            f"Broyden 未生效：n_jac_evals={result.n_jac_evals} ≮ n_iter={result.n_iter}"
        )

    def test_zero_iterations_when_start_is_root(self):
        """③ 初值即根：零迭代收敛。"""
        defect, jac = _vieta_system()
        root = np.array([1.0, 2.0])
        result = solve_square(defect, jac, root)
        assert result.status is ConvergenceState.CONVERGED
        assert result.n_iter == 0
        assert result.n_jac_evals == 0
        assert result.defect_norm <= 1e-10

    def test_singular_jacobian_reports_failure(self):
        """③ 零雅可比报 FAILED/SINGULAR_JACOBIAN（最小二乘兜底无效），不崩溃。"""
        defect, _ = _vieta_system()
        zero_jac = lambda x: np.zeros((2, 2))  # noqa: E731
        result = solve_square(defect, zero_jac, [0.3, 1.6])
        assert result.status is ConvergenceState.FAILED
        assert result.cause is FailureCause.SINGULAR_JACOBIAN


class TestSolveMinimax:
    def test_equioscillation_on_overdetermined_linear_system(self):
        """④/① 超定线性系统 (3,2)：max|F| 下降且残差等幅双号（Chebyshev 特征）。

        Haar 条件矩阵 [[1,0],[0,1],[1,1]] 的线性 Chebyshev 问题的最优解
        由 n+1 = 3 个残差以相反符号达到同一最大模刻画。
        """
        a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([0.1, 0.2, 0.5])
        defect = lambda x: a @ x - b  # noqa: E731
        x0 = np.zeros(2)
        result = solve_minimax(defect, x0)
        assert result.status is ConvergenceState.CONVERGED, result.message
        residual = defect(result.x)
        assert result.max_defect < float(np.max(np.abs(defect(x0))))
        # 等幅：三个残差模都达到 max（Haar 系统的最优特征）。
        assert np.all(np.abs(np.abs(residual) - result.max_defect) <= 1e-6 * result.max_defect)
        # 双号：残差符号不全同（等幅振荡）。
        assert len(set(np.sign(residual))) == 2
        # 解析核对（① 定义性公式）：本系统的 Chebyshev 解满足 A^T F* = 0。
        assert np.allclose(a.T @ residual, 0.0, atol=1e-6)


@requires_native_symbols("propagate_kepler_py")
class TestSquareTierOnFrameworkChain:
    """p=q 档在框架链上的定向问题：n=2 冲量（6 变量 6 方程）matchpoint 方系统。"""

    @staticmethod
    def _problem(departure, arrival, tof):
        """构造单 leg n=2 的归一化缺陷/雅可比闭包与决策向量形状。"""

        def defect(x_flat):
            (leg,) = evaluate_chain(
                [
                    ChainLegRequest(
                        kernel=LegKernel(MU),
                        n_segments=2,
                        imp_offset=0,
                        dv=np.asarray(x_flat, dtype=float).reshape(2, 3),
                        dt=tof / 2.0,
                        anchor_f=departure,
                        anchor_b=arrival,
                        anchor_sens_f=np.zeros((6, 6)),
                        anchor_sens_b=np.zeros((6, 6)),
                    )
                ],
                n_var=6,
                len_scale=7000.0,
                vel_scale=8.0,
                with_sens=False,
            )
            raw = leg.residual_raw
            return np.concatenate([raw[:3] / 7000.0, raw[3:] / 8.0])

        def jac(x_flat):
            (leg,) = evaluate_chain(
                [
                    ChainLegRequest(
                        kernel=LegKernel(MU),
                        n_segments=2,
                        imp_offset=0,
                        dv=np.asarray(x_flat, dtype=float).reshape(2, 3),
                        dt=tof / 2.0,
                        anchor_f=departure,
                        anchor_b=arrival,
                        anchor_sens_f=np.zeros((6, 6)),
                        anchor_sens_b=np.zeros((6, 6)),
                    )
                ],
                n_var=6,
                len_scale=7000.0,
                vel_scale=8.0,
                with_sens=True,
            )
            return leg.eq_jac

        return defect, jac

    def test_ballistic_degenerate_zero_impulse_is_root(self):
        """③ 弹道退化：出发/到达共 conic 弧，零冲量即根（零迭代收敛）。"""
        departure = np.array([7000.0, 0.0, 0.0, 0.0, float(np.sqrt(MU / 7000.0)), 0.0])
        tof = 5400.0
        arrival = np.asarray(
            propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
        )
        defect, jac = self._problem(departure, arrival, tof)
        result = solve_square(defect, jac, np.zeros(6))
        assert result.status is ConvergenceState.CONVERGED
        assert result.n_iter == 0
        assert result.defect_norm <= 1e-10

    def test_perturbed_arrival_converges(self):
        """① 定向问题：到达端速度加常值偏置后 solve_square 收敛到残差 < 1e-9。"""
        departure = np.array([7000.0, 0.0, 0.0, 0.0, float(np.sqrt(MU / 7000.0)), 0.0])
        tof = 5400.0
        arrival = np.asarray(
            propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
        ).copy()
        arrival[3:] += np.array([0.02, -0.01, 0.0])  # km/s，需非零冲量修正
        defect, jac = self._problem(departure, arrival, tof)
        result = solve_square(defect, jac, np.zeros(6))
        assert result.status is ConvergenceState.CONVERGED, result.message
        assert result.defect_norm < 1e-9
        # 解处冲量非零（问题确被求解而非弹道退化）。
        assert float(np.max(np.abs(result.x))) > 1e-6
