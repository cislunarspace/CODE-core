"""Floquet 投影生产路径测试：对偶基表、任意时刻模态基、注入构造（#843）。

断言一律按定义：对偶基是模态基之逆、漂移对偶行是有界子空间法向量的
逐历元推广、注入表与状态表按模 T 对齐。论文数值属结果性对照，不进
断言（ADR 0055）。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from e2m2e.algorithm.proximity import (
    FloquetModeError,
    build_nominal_orbit,
    dual_basis,
    floquet_modes,
    modal_basis_at,
)
from e2m2e.data.types.orbit import Orbit
from e2m2e.data.types.trajectory import NominalOrbit

pytestmark = pytest.mark.orchestration


def _grid_states(dro21, modes21) -> np.ndarray:
    """把标称轨道采样到模态网格上，返回 (n, 6) 状态表。"""
    orbit, dynamics = dro21
    out = dynamics.propagate(
        orbit.states[0],
        (modes21.t_origin, modes21.t_origin + modes21.period),
        t_eval=modes21.times,
    )
    return np.asarray(out["states"], dtype=float)


def _shifted_origin_orbit(dro21):
    """相位原点平移到 7.5 TU 的单点轨道（period 保留）。"""
    orbit, _ = dro21
    shifted = Orbit(states=orbit.states[:1].copy(), times=np.array([float(orbit.times[0]) + 7.5]))
    shifted.period = orbit.period
    return shifted


class TestDualBasis:
    """对偶基表：模态基逐历元之逆。"""

    def test_inverts_basis_at_every_node(self, modes21) -> None:
        """每个历元的对偶基与模态基相乘为单位阵。"""
        dual = dual_basis(modes21)
        assert dual.shape == modes21.basis.shape
        identity = np.eye(6)
        residual = max(
            float(np.max(np.abs(dual[k] @ modes21.basis[k] - identity)))
            for k in range(dual.shape[0])
        )
        assert residual < 1e-9

    def test_drift_row_at_origin_is_bounded_normal(self, modes21) -> None:
        """原点处的漂移对偶行即有界子空间法向量（#832 的构造同一对象）。"""
        dual = dual_basis(modes21)
        np.testing.assert_allclose(dual[0][5], modes21.bounded_normal, rtol=0.0, atol=1e-9)

    def test_drift_row_is_periodic(self, modes21) -> None:
        """漂移对偶行 T 周期：首末节点的取值一致，模 T 延拓语义成立。"""
        dual = dual_basis(modes21)
        np.testing.assert_allclose(dual[-1][5], dual[0][5], rtol=0.0, atol=1e-8)

    def test_singular_basis_rejected(self, modes21) -> None:
        """模态基退化（漂移列与切向列共线）时确定性报错。"""
        broken = dataclasses.replace(modes21, basis=modes21.basis.copy())
        broken.basis[:, :, 5] = broken.basis[:, :, 0]
        with pytest.raises(FloquetModeError, match="奇异"):
            dual_basis(broken)


class TestModalBasisAt:
    """任意时刻的模态基（在线重算路径）。"""

    @pytest.mark.parametrize("index", [1, 50, 200])
    def test_matches_grid_at_nodes(self, dro21, modes21, index: int) -> None:
        """网格节点上的在线重构与采样表一致（积分容差量级）。"""
        _, dynamics = dro21
        t = float(modes21.times[index])
        rebuilt = modal_basis_at(modes21, t, dynamics)
        np.testing.assert_allclose(rebuilt, modes21.basis[index], rtol=1e-9, atol=1e-9)

    def test_columns_match_nonlinear_difference(self, dro21, modes21) -> None:
        """非节点时刻：切向、漂移与两对中心列与非线性传播有限差分一致。

        偏差 ε·v 的非线性传播减标称传播，除以 ε 是 STM 作用在 v 上的
        有限差分近似（线性化误差 O(ε)，ε 取 1e-7，对照容差 1e-4 相对量）。
        中心对的复原式：初值 a 给出 cosθ·u − sinθ·v，θ 为该对的转角。
        """
        orbit, dynamics = dro21
        t = modes21.t_origin + 0.37 * modes21.period
        state0 = orbit.states[0]
        eps = 1e-7

        nominal = np.asarray(
            dynamics.propagate(state0, (modes21.t_origin, t), t_eval=[t])["states"],
            dtype=float,
        )[0]

        def perturbed(initial: np.ndarray) -> np.ndarray:
            out = dynamics.propagate(state0 + eps * initial, (modes21.t_origin, t), t_eval=[t])
            return (np.asarray(out["states"], dtype=float)[0] - nominal) / eps

        basis_at = modal_basis_at(modes21, t, dynamics)
        np.testing.assert_allclose(perturbed(modes21.tangent), basis_at[:, 0], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(perturbed(modes21.drift), basis_at[:, 5], rtol=1e-4, atol=1e-6)

        theta_in = modes21.in_plane.alpha * 0.37
        center = np.cos(theta_in) * basis_at[:, 1] - np.sin(theta_in) * basis_at[:, 2]
        np.testing.assert_allclose(perturbed(modes21.in_plane.a), center, rtol=1e-4, atol=1e-6)

        theta_out = modes21.out_of_plane.alpha * 0.37
        normal = np.cos(theta_out) * basis_at[:, 3] - np.sin(theta_out) * basis_at[:, 4]
        np.testing.assert_allclose(perturbed(modes21.out_of_plane.a), normal, rtol=1e-4, atol=1e-6)

    def test_before_phase_origin_rejected(self, modes21) -> None:
        """早于相位原点的时刻确定性报错（STM 正向传播不支持）。"""
        with pytest.raises(FloquetModeError, match="相位原点"):
            modal_basis_at(modes21, modes21.t_origin - 0.1, None)  # type: ignore[arg-type]

    def test_at_phase_origin_returns_initial_basis(self, modes21) -> None:
        """相位原点处直接返回模态初值矩阵，不触发传播。"""
        got = modal_basis_at(modes21, modes21.t_origin, None)  # type: ignore[arg-type]
        np.testing.assert_allclose(got, modes21.basis[0], rtol=0.0, atol=1e-15)


class TestBuildNominalOrbit:
    """注入构造与相位原点一致性校验。"""

    def test_populates_aligned_table(self, dro21, modes21) -> None:
        """默认历元取模态网格，基表与对偶基表注入，形状与网格一致。"""
        states = _grid_states(dro21, modes21)
        injected = build_nominal_orbit(modes21, states)
        assert isinstance(injected, NominalOrbit)
        np.testing.assert_allclose(injected.epochs, modes21.times)
        np.testing.assert_allclose(injected.states, states, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(injected.floquet_basis, modes21.basis)
        np.testing.assert_allclose(injected.projection_factors, dual_basis(modes21))
        assert injected.interpolator is None

    def test_integer_period_shift_accepted(self, dro21, modes21) -> None:
        """历元整体平移整数个周期仍按模 T 对齐，构造通过。"""
        states = _grid_states(dro21, modes21)
        shifted = modes21.times + 2.0 * modes21.period
        injected = build_nominal_orbit(modes21, states, epochs=shifted)
        np.testing.assert_allclose(injected.epochs, shifted)

    def test_shifted_epochs_with_nonzero_origin(self, dro21, modes21) -> None:
        """非零相位原点的模态集配合整数周期平移历元，构造通过。

        平移校验按历元差折算圈数，非零网格原点加整数周期平移是合法输入，
        误用绝对时间折算会误拒。
        """
        orbit, dynamics = dro21
        modes = floquet_modes(_shifted_origin_orbit(dro21), dynamics)
        epochs = modes.times + 2.0 * modes.period
        states = _grid_states((orbit, dynamics), modes)
        injected = build_nominal_orbit(modes, states, epochs=epochs)
        np.testing.assert_allclose(injected.epochs, epochs)

    def test_misaligned_epochs_rejected(self, dro21, modes21) -> None:
        """非整数周期平移破坏相位原点一致性，确定性报错。"""
        states = _grid_states(dro21, modes21)
        shifted = modes21.times + 0.3 * modes21.period
        with pytest.raises(FloquetModeError, match="相位原点"):
            build_nominal_orbit(modes21, states, epochs=shifted)

    def test_nonuniform_epochs_rejected(self, dro21, modes21) -> None:
        """局部扰动的历元（不再对应网格）确定性报错。"""
        states = _grid_states(dro21, modes21)
        epochs = modes21.times.copy()
        epochs[3] += 0.5 * (epochs[4] - epochs[3])
        with pytest.raises(FloquetModeError, match="相位原点"):
            build_nominal_orbit(modes21, states, epochs=epochs)

    def test_nonfinite_epochs_rejected(self, modes21) -> None:
        """历元含 NaN 时先于折算确定性报错，不泄漏数值异常。"""
        states = np.zeros((modes21.times.shape[0], 6))
        epochs = modes21.times.astype(float).copy()
        epochs[5] = np.nan
        with pytest.raises(FloquetModeError, match="有限"):
            build_nominal_orbit(modes21, states, epochs=epochs)

    def test_nonuniform_grid_rejected(self, dro21) -> None:
        """非等间距模态网格违反容器的等间距历元表要求，确定性报错。"""
        orbit, dynamics = dro21
        period = float(orbit.period)
        grid = np.concatenate(
            [np.linspace(0.0, 0.5 * period, 5), np.linspace(0.5 * period, period, 40)[1:]]
        )
        ragged = floquet_modes(orbit, dynamics, times=grid)
        with pytest.raises(FloquetModeError, match="等间距"):
            build_nominal_orbit(ragged, np.zeros((grid.shape[0], 6)))

    def test_epoch_count_mismatch_rejected(self, modes21) -> None:
        """历元长度与模态网格不一致时确定性报错。"""
        states = np.zeros((modes21.times.shape[0], 6))
        with pytest.raises(FloquetModeError, match="长度"):
            build_nominal_orbit(modes21, states, epochs=modes21.times[:-1])

    def test_state_shape_mismatch_rejected(self, modes21) -> None:
        """状态表形状与网格不一致时确定性报错。"""
        with pytest.raises(FloquetModeError, match="形状"):
            build_nominal_orbit(modes21, np.zeros((modes21.times.shape[0], 5)))

    def test_default_container_fields_stay_empty(self) -> None:
        """不注入时三个可选字段保持 None（容器默认值不变）。"""
        bare = NominalOrbit(epochs=np.linspace(0.0, 1.0, 11), states=np.zeros((11, 6)))
        assert bare.floquet_basis is None
        assert bare.projection_factors is None
        assert bare.interpolator is None
