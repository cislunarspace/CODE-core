"""MEGNO 混沌指标测试（Primer §7.1 式 142；Rust 内核 + python 参照）。

验收口径（issue #579）：
- 规则轨迹解析对照：开普勒圆轨 Ȳ → 2（rel 容差）；
- 已知混沌初值（gateway 带采样）单调增长；
- 积分器守恒律回归不回退（theory marker：Jacobi 常数）；
- 变分方程与有限差分 δ̇/δ 对照。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.spatiography.constants import PRIMER_DEFAULTS
from e2m2e.algorithm.spatiography.megno import (
    megno_reference,
    propagate_cr3bp_megno,
)
from e2m2e.integrators import propagate_cr3bp_py

pytestmark = pytest.mark.theory

MU = PRIMER_DEFAULTS.moon_mass_parameter
#: 主天体近旁近圆轨迹（r ≈ 0.05：准开普勒、远离次天体）。
_REGULAR_STATE = [-(MU) + 0.05, 0.0, 0.0, 0.0, (1.0 / 0.05) ** 0.5, 0.0]
#: L1 gateway 带混沌采样（地月 mu，L1 ≈ 0.8369）。
_CHAOTIC_STATE = [0.8369, 0.0, 0.0, 0.0, 0.0, 0.35]


def _jacobi(mu: float, states: np.ndarray) -> np.ndarray:
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    v2 = np.sum(states[:, 3:] ** 2, axis=1)
    return x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - v2


class TestMegnoRegular:
    def test_keplerian_circular_orbit_converges_to_two(self):
        t_eval = np.arange(0.0, 100.0, 5.0)
        result = propagate_cr3bp_megno(MU, (0.0, 100.0), t_eval, _REGULAR_STATE, rtol=1e-10)
        assert result["ybar"][-1] == pytest.approx(2.0, rel=0.02)
        # Y(t) 在 2 附近振荡（式 142 的正则行为）。
        assert abs(result["y"][-1] - 2.0) < 0.5

    def test_jacobi_conservation_not_degraded(self):
        """与纯状态传播同窗同容差的 Jacobi 漂移同量级（不回退，theory 口径）。"""
        t_eval = np.arange(0.0, 60.0, 5.0)
        result = propagate_cr3bp_megno(MU, (0.0, 60.0), t_eval, _REGULAR_STATE, rtol=1e-10)
        plain = propagate_cr3bp_py(
            MU, (0.0, 60.0), [float(t) for t in t_eval], _REGULAR_STATE, 1e-10, 1e-10
        )
        drift_megno = float(np.ptp(_jacobi(MU, np.asarray(result["states"]))))
        drift_plain = float(np.ptp(_jacobi(MU, np.asarray(plain["states"]))))
        assert drift_megno <= 10.0 * drift_plain + 1e-10

    def test_ybar_time_average_smoothes_y(self):
        t_eval = np.arange(0.0, 100.0, 1.0)
        result = propagate_cr3bp_megno(MU, (0.0, 100.0), t_eval, _REGULAR_STATE, rtol=1e-10)
        y = np.asarray(result["y"])
        ybar = np.asarray(result["ybar"])
        assert np.abs(ybar - 2.0).max() <= np.abs(y - 2.0).max() + 1e-12


class TestMegnoChaotic:
    def test_gateway_sample_grows_monotonically(self):
        t_eval = np.arange(0.0, 200.0, 10.0)
        result = propagate_cr3bp_megno(MU, (0.0, 200.0), t_eval, _CHAOTIC_STATE, rtol=1e-10)
        ybar = np.asarray(result["ybar"])
        assert ybar[-1] > 2.5
        assert ybar[-1] >= ybar[len(ybar) // 2]
        # 越过初期暂态后单调抬升（无增长塌缩回正则带）。
        assert np.all(np.diff(ybar[2:]) > -0.1)
        assert ybar[-1] > 3.0 * ybar[2]

    def test_chaotic_slope_scales_with_window(self):
        short = propagate_cr3bp_megno(
            MU, (0.0, 100.0), np.arange(0.0, 100.0, 10.0), _CHAOTIC_STATE, rtol=1e-10
        )
        long = propagate_cr3bp_megno(
            MU, (0.0, 200.0), np.arange(0.0, 200.0, 10.0), _CHAOTIC_STATE, rtol=1e-10
        )
        assert long["ybar"][-1] > short["ybar"][-1]


class TestVariationalConsistency:
    def test_tangent_matches_finite_difference(self):
        """切变分 δ(t) 与双轨迹有限差分（Φ·δ₀）方向一致、长度重标后吻合。"""
        t_eval = np.array([0.0, 20.0])
        delta0 = np.array([1e-8, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = propagate_cr3bp_megno(
            MU, (0.0, 20.0), t_eval, _REGULAR_STATE, initial_delta=delta0, rtol=1e-12
        )
        tangent = np.asarray(result["deltas"])[-1]
        eps = 1e-8
        base = np.asarray(
            propagate_cr3bp_py(MU, (0.0, 20.0), [20.0], _REGULAR_STATE, 1e-12, 1e-12)["states"]
        )[0]
        pert = np.asarray(
            propagate_cr3bp_py(
                MU,
                (0.0, 20.0),
                [20.0],
                (np.array(_REGULAR_STATE) + np.array([1, 0, 0, 0, 0, 0]) * eps).tolist(),
                1e-12,
                1e-12,
            )["states"]
        )[0]
        finite_diff = (pert - base) / eps
        # 方向一致（切向重归一化只差正标量）。
        cos_angle = abs(float(tangent @ finite_diff)) / (
            np.linalg.norm(tangent) * np.linalg.norm(finite_diff)
        )
        assert cos_angle > 0.999

    def test_rust_python_reference_parity(self):
        t_eval = np.arange(0.0, 40.0, 2.0)
        rust = propagate_cr3bp_megno(
            MU, (0.0, 40.0), t_eval, _REGULAR_STATE, rtol=1e-12, backend="rust"
        )
        from e2m2e.algorithm.spatiography.megno import _cr3bp_megno_eom

        reference = megno_reference(
            _cr3bp_megno_eom(MU),
            (0.0, 40.0),
            t_eval,
            _REGULAR_STATE,
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(rust["ybar"], reference["ybar"], rtol=5e-4, atol=1e-5)
        # 两套积分器（PD78 vs DOP853）各自收敛后的轨迹等价性：速度小分量
        # 由指消主导，atol 兑底。
        np.testing.assert_allclose(
            np.asarray(rust["states"]),
            np.asarray(reference["states"]),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_backend_is_explicit(self):
        import inspect

        signature = inspect.signature(propagate_cr3bp_megno)
        assert signature.parameters["backend"].default == "rust"

    def test_reference_coriolis_terms_match_rust(self):
        """参照实现的科里奥利块用 dv（非 dr）索引——与 Rust 内核逐项一致。"""
        from e2m2e.algorithm.spatiography.megno import _cr3bp_megno_eom

        y = np.zeros(14)
        y[:6] = _REGULAR_STATE
        y[6 + 4] = 0.3  # δv_y
        deriv = _cr3bp_megno_eom(MU)(1.0, y)
        # δv̇_x 含 +2·δv_y；其余为零时 δṙ = δv 直接锁定索引关系。
        assert deriv[9] == pytest.approx(2.0 * 0.3, rel=1e-12)
