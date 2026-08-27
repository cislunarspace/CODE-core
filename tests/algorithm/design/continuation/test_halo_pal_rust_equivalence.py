"""Halo PAL 延拓 Rust 后端 vs Python 后端等价性对照。

对照权威基准是 Python 参照路径（``backend="python"``，numpy 实现），
验证 Rust 数值内核（``pal_newton_step_py`` / ``pal_f_df_tangent_py``）
在同一初猜、同一容差下产出一致的延拓步与轨道族：

- F / dF：两后端底层走同一个 Rust ``propagate_cr3bp_stm``，传播逐位
  一致，组装是纯拷贝——容差仅吸收 Python 侧标量往返。
- 切向量：Rust 广义叉积零空间 vs numpy SVD 零空间，方向一致、符号
  约定各自任意——按 |dot| = 1 对照。
- PAL 牛顿步：4×4 高斯消元（部分主元）vs LAPACK ``dgesv``，迭代序列
  在机器精度内一致——allclose + 收敛/奇异标志与迭代次数精确相等。
- 端到端：同一种子、同一参数各跑一条 PAL 支，族成员初态与周期
  allclose（微分修正把两后端的机器精度级差异重新闭合到 TolDiffCorr
  量级以内）。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# 扩展未构建时（doc build 等合法场景）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.solver.continuation import (
    Continuation,
    _pal_newton_step_python,
    compute_F_and_dF_symmetric_xz_plane,
    compute_tangent_vector,
)
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.integrators import pal_f_df_tangent_py, pal_newton_step_py

pytestmark = pytest.mark.orchestration

#: 与 Halo 正向支默认伪弧长步长一致
DS = 0.0045
TOL_PAL = 1e-6
ITER_MAX = 100


def _free_vars(orbit) -> np.ndarray:
    """轨道首态/周期 → PAL 自由变量 X = [rx, rz, vy, T/2]"""
    sv0 = orbit.states[0]
    return np.array([sv0[0], sv0[2], sv0[4], float(orbit.period) / 2.0])


def _rust_f_df_tangent(x: np.ndarray, sv0: np.ndarray, dynamics) -> dict:
    return pal_f_df_tangent_py(
        mu=float(dynamics.system.mu),
        x=[float(v) for v in x],
        sv0=[float(v) for v in sv0],
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
    )


def _rust_newton_step(
    x_start: np.ndarray,
    x_ref: np.ndarray,
    sv0: np.ndarray,
    tangent: np.ndarray,
    dynamics,
) -> dict:
    return pal_newton_step_py(
        mu=float(dynamics.system.mu),
        x_start=[float(v) for v in x_start],
        x_ref=[float(v) for v in x_ref],
        sv0=[float(v) for v in sv0],
        tangent_ref=[float(v) for v in tangent],
        ds=DS,
        tol=TOL_PAL,
        iter_max=ITER_MAX,
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
    )


class TestFdfTangentEquivalence:
    """约束向量/雅可比/切向量的单点对照。"""

    def test_f_df_consistent(self, corrected_halo_l1, earth_moon_dynamics):
        seed, _ = corrected_halo_l1
        x = _free_vars(seed)
        sv0 = seed.states[0].copy()

        f_py, df_py = compute_F_and_dF_symmetric_xz_plane(x, sv0, earth_moon_dynamics)
        rust = _rust_f_df_tangent(x, sv0, earth_moon_dynamics)

        assert_allclose(np.asarray(rust["f"]), f_py, rtol=1e-12, atol=1e-14)
        assert_allclose(np.asarray(rust["df"]), df_py, rtol=1e-12, atol=1e-14)

    def test_tangent_up_to_sign(self, corrected_halo_l1, earth_moon_dynamics):
        seed, _ = corrected_halo_l1
        x = _free_vars(seed)
        sv0 = seed.states[0].copy()

        _, df_py = compute_F_and_dF_symmetric_xz_plane(x, sv0, earth_moon_dynamics)
        t_py = compute_tangent_vector(df_py)
        t_rust = np.asarray(_rust_f_df_tangent(x, sv0, earth_moon_dynamics)["tangent"])

        # 零空间一维，两实现必共线；符号约定各自任意
        assert abs(float(np.dot(t_rust, t_py))) == pytest.approx(1.0, abs=1e-9)


class TestPalNewtonStepEquivalence:
    """单步 PAL 牛顿迭代对照（同一预测起点、同一参考点与切向量）。"""

    def test_single_step(self, corrected_halo_l1, earth_moon_dynamics):
        seed, _ = corrected_halo_l1
        x = _free_vars(seed)
        sv0 = seed.states[0].copy()
        _, df = compute_F_and_dF_symmetric_xz_plane(x, sv0, earth_moon_dynamics)
        tangent = compute_tangent_vector(df)
        x_start = x + DS * tangent

        py = _pal_newton_step_python(
            x_start=x_start,
            x_ref=x,
            sv0=sv0,
            tangent_ref=tangent,
            ds=DS,
            tol=TOL_PAL,
            iter_max=ITER_MAX,
            dynamics=earth_moon_dynamics,
        )
        rust = _rust_newton_step(x_start, x, sv0, tangent, earth_moon_dynamics)

        assert rust["converged"] == py["converged"]
        assert rust["singular"] == py["singular"]
        assert rust["iterations"] == py["iterations"]
        assert_allclose(np.asarray(rust["x_new"]), py["x_new"], rtol=1e-9, atol=1e-12)
        assert_allclose(np.asarray(rust["tangent"]), py["tangent"], rtol=1e-9, atol=1e-12)
        assert float(rust["residual"]) == pytest.approx(float(py["residual"]), rel=1e-9)


class TestPalFamilyEquivalence:
    """端到端：同一种子各跑一条 PAL 支，族成员逐条对照。"""

    N_ORBITS = 3

    def _run_branch(self, corrected_halo_l1, earth_moon_dynamics, backend: str):
        seed, _ = corrected_halo_l1
        continuation = Continuation(corrector=DifferentialCorrection(earth_moon_dynamics))
        result = continuation.pseudo_arclength_continuation(
            seed,
            n_orbits=self.N_ORBITS,
            step_size=DS,
            direction="positive",
            verbose=False,
            TolPAL=TOL_PAL,
            directional_increment=True,
            target_vector=1,
            target_direction=1,
            backend=backend,
        )
        return result.family

    def test_positive_branch(self, corrected_halo_l1, earth_moon_dynamics):
        family_py = self._run_branch(corrected_halo_l1, earth_moon_dynamics, "python")
        family_rust = self._run_branch(corrected_halo_l1, earth_moon_dynamics, "rust")

        assert len(family_rust) == len(family_py) == 1 + self.N_ORBITS

        for orb_py, orb_rust in zip(family_py.orbits, family_rust.orbits, strict=True):
            assert_allclose(
                orb_rust.states[0],
                orb_py.states[0],
                rtol=1e-6,
                atol=1e-9,
                err_msg="族成员初态两后端不一致",
            )
            assert float(orb_rust.period) == pytest.approx(float(orb_py.period), rel=1e-6)
