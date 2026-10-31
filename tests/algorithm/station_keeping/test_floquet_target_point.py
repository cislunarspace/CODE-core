"""Floquet 投影目标点律与模态量来源测试（#843）。

合成场景（固定对偶基）按定义断言清零与最小范数性质，并用时变来源与
时变标称钉住偏差与时刻的取值链。真实 2:1 DRO 上断言注入表与在线数值
投影两种模态量来源的等价（积分容差量级）与端到端 Δv 序列一致。论文
数值属结果性对照，不进断言（ADR 0055）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.proximity import (
    FloquetModeError,
    build_nominal_orbit,
    dual_basis,
    floquet_modes,
)
from e2m2e.algorithm.station_keeping.target_point import (
    FloquetTargetPointLaw,
    InjectedModalSource,
    OnlineModalSource,
)
from e2m2e.data.types.orbit import Orbit
from e2m2e.data.types.trajectory import NominalOrbit

pytestmark = pytest.mark.orchestration


class StubModalSource:
    """固定对偶基的模态量来源（合成场景）。"""

    def __init__(self, dual: np.ndarray) -> None:
        self._dual = np.asarray(dual, dtype=float)

    def dual_basis_at(self, t: float) -> np.ndarray:
        return self._dual.copy()


class ConstantNominal:
    """常值标称状态视图（合成场景）。"""

    def __init__(self, state: np.ndarray) -> None:
        self._state = np.asarray(state, dtype=float)

    def state_at(self, t: float) -> np.ndarray:
        return self._state.copy()


class TableNominalView:
    """状态表线性插值的标称轨道视图（运行时视图，非数据容器）。

    状态表跨一个周期，查询按模 T 回卷到圈内，与注入表的回卷口径一致。
    """

    def __init__(self, epochs: np.ndarray, states: np.ndarray) -> None:
        self._epochs = epochs
        self._states = states

    def state_at(self, t: float) -> np.ndarray:
        span = float(self._epochs[-1] - self._epochs[0])
        phase = (float(t) - float(self._epochs[0])) % span
        return np.array([np.interp(phase, self._epochs, self._states[:, i]) for i in range(6)])


def _synthetic_dual() -> np.ndarray:
    """可逆的合成对偶基，第 5 行为给定的漂移对偶行。"""
    dual = np.eye(6)
    dual[5] = np.array([0.1, 0.2, 0.0, 0.3, 0.1, 0.05])
    return dual


def _nominal_table(dro21, modes21):
    """模态网格上的标称状态表与其视图（最小注入示例的全链路）。"""
    orbit, dynamics = dro21
    out = dynamics.propagate(
        orbit.states[0],
        (modes21.t_origin, modes21.t_origin + modes21.period),
        t_eval=modes21.times,
    )
    states = np.asarray(out["states"], dtype=float)
    return modes21.times.copy(), states


class TestFloquetTargetPointLaw:
    """控制律定义性质（合成对偶基，精确代数）。"""

    def test_zeroes_drift_coordinate(self) -> None:
        """机动后偏差的漂移模态坐标归零。"""
        dual = _synthetic_dual()
        nominal_state = np.zeros(6)
        deviation = np.array([1e-3, -2e-3, 4e-4, 1e-4, -3e-5, 2e-5])
        law = FloquetTargetPointLaw()
        dv = law.compute_maneuver(
            nominal_state + deviation,
            0.0,
            nominal=ConstantNominal(nominal_state),
            modal=StubModalSource(dual),
        )
        residual = float(dual[5] @ (deviation + np.concatenate([np.zeros(3), dv])))
        assert abs(residual) < 1e-15

    def test_minimum_norm_property(self) -> None:
        """最小范数：可行、与 n_v 共线，且不劣于任何其他可行解。"""
        dual = _synthetic_dual()
        n = dual[5]
        n_v = n[3:]
        deviation = np.array([5e-4, 1e-4, -7e-4, 2e-4, 3e-5, -1e-5])
        law = FloquetTargetPointLaw()
        dv = law.compute_maneuver(
            deviation, 0.0, nominal=ConstantNominal(np.zeros(6)), modal=StubModalSource(dual)
        )
        # 可行性：机动后漂移坐标为零（约束 n_v·Δv = −α）。
        np.testing.assert_allclose(n_v @ dv, -float(n @ deviation), atol=1e-15)
        # 共线：Δv 落在 n_v 张成的一维子空间。
        np.testing.assert_allclose(np.cross(n_v, dv), np.zeros(3), atol=1e-18)
        # 最小性：任意可行解（本解加零空间扰动）的范数不小于本解。
        null = np.array([n_v[1], -n_v[0], 0.0])
        for weight in (-1.0, 0.5, 3.0):
            alternative = dv + weight * null
            assert float(np.linalg.norm(dv)) <= float(np.linalg.norm(alternative)) + 1e-15

    def test_zero_deviation_gives_zero_maneuver(self) -> None:
        """零偏差不解控制量。"""
        dual = _synthetic_dual()
        law = FloquetTargetPointLaw()
        dv = law.compute_maneuver(
            np.zeros(6), 0.0, nominal=ConstantNominal(np.zeros(6)), modal=StubModalSource(dual)
        )
        np.testing.assert_array_equal(dv, np.zeros(3))

    def test_uses_deviation_and_source_at_maneuver_time(self) -> None:
        """偏差取测量减控制时刻标称，模态量取控制时刻而非零时刻的值。"""
        dual_a = _synthetic_dual()
        dual_b = _synthetic_dual()
        dual_b[5] = np.array([0.2, 0.0, 0.1, 0.0, 0.4, 0.1])

        class TwoEpochSource:
            """t0 取 2.0 时返回 dual_b，其余时刻返回 dual_a。"""

            def dual_basis_at(self, t: float) -> np.ndarray:
                return dual_b.copy() if float(t) == 2.0 else dual_a.copy()

        class DriftingNominal:
            """标称状态随时刻线性漂移，state_at(t) 取 t 时刻的值。"""

            def state_at(self, t: float) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]) + 0.05 * float(t) * np.ones(6)

        measured = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.25])
        law = FloquetTargetPointLaw()
        dv = law.compute_maneuver(measured, 2.0, nominal=DriftingNominal(), modal=TwoEpochSource())
        deviation = measured - DriftingNominal().state_at(2.0)
        n = dual_b[5]
        n_v = n[3:]
        expected = -(float(n @ deviation) / float(n_v @ n_v)) * n_v
        np.testing.assert_allclose(dv, expected, rtol=0.0, atol=1e-15)
        # 双重保险：零时刻的标称与零时刻的对偶行都给不出上式。
        assert not np.allclose(deviation, measured)
        assert not np.allclose(deviation, measured - DriftingNominal().state_at(0.0))
        assert not np.allclose(dual_b[5], dual_a[5])

    def test_degenerate_normal_velocity_rejected(self) -> None:
        """速度分块相对全行退化且漂移坐标非零时确定性报错。

        相对判据：‖n_v‖² 低于 ‖n‖² 的 1e-16（相对范数 1e-8）即退化，
        绝对值不小但占比可忽略的速度分块同样被拦下。
        """
        dual = np.eye(6)
        dual[5] = np.array([0.1, 0.2, 0.0, 1e-9, 0.0, 0.0])
        law = FloquetTargetPointLaw()
        with pytest.raises(FloquetModeError, match="退化"):
            law.compute_maneuver(
                np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
                0.0,
                nominal=ConstantNominal(np.zeros(6)),
                modal=StubModalSource(dual),
            )

    def test_degenerate_normal_with_zero_drift_gives_zero(self) -> None:
        """速度分块退化但漂移坐标已为零时，零 Δv 是最小范数解。"""
        dual = np.eye(6)
        dual[5] = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        law = FloquetTargetPointLaw()
        dv = law.compute_maneuver(
            np.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]),
            0.0,
            nominal=ConstantNominal(np.zeros(6)),
            modal=StubModalSource(dual),
        )
        np.testing.assert_array_equal(dv, np.zeros(3))


class TestInjectedModalSource:
    """注入表来源：查表与校验。"""

    def test_exact_lookup_at_table_nodes(self, dro21, modes21) -> None:
        """表内部节点上的查表按分量等于注入的对偶基表行（1e-12 容差）。

        末节点（相位原点加一个周期）按模 T 语义回卷到相位 0，行 0 跨圈
        有移位，故断言只取内部节点，周期行为由回卷测试覆盖。
        """
        epochs, states = _nominal_table(dro21, modes21)
        injected = build_nominal_orbit(modes21, states)
        source = InjectedModalSource(injected)
        dual = dual_basis(modes21)
        for k in (0, 40, 100, 190):
            np.testing.assert_allclose(
                source.dual_basis_at(epochs[k]), dual[k], rtol=0.0, atol=1e-12
            )

    def test_periodic_rows_wrap_beyond_one_period(self, dro21, modes21) -> None:
        """模 T 回卷：满整数圈后的对偶基第 1 至 5 行与圈内同相位一致。"""
        epochs, states = _nominal_table(dro21, modes21)
        source = InjectedModalSource(build_nominal_orbit(modes21, states))
        wrapped = source.dual_basis_at(epochs[60] + 2.0 * modes21.period)
        np.testing.assert_allclose(
            wrapped[1:], source.dual_basis_at(epochs[60])[1:], rtol=0.0, atol=1e-9
        )

    def test_lookup_after_integer_period_shift(self, dro21, modes21) -> None:
        """整数周期平移历元的容器查表与原容器同相位一致。"""
        epochs, states = _nominal_table(dro21, modes21)
        shifted = build_nominal_orbit(modes21, states, epochs=epochs + 2.0 * modes21.period)
        source = InjectedModalSource(shifted)
        dual = dual_basis(modes21)
        for k in (0, 60, 120, 190):
            got = source.dual_basis_at(shifted.epochs[k])
            np.testing.assert_allclose(got[1:], dual[k][1:], rtol=0.0, atol=1e-9)

    def test_lookup_with_nonzero_phase_origin(self, dro21, modes21) -> None:
        """相位原点非零的模态集：注入查表与在线重构都按相对相位进行。"""
        orbit, dynamics = dro21
        shifted_orbit = Orbit(
            states=orbit.states[:1].copy(), times=np.array([float(orbit.times[0]) + 7.5])
        )
        shifted_orbit.period = orbit.period
        modes = floquet_modes(shifted_orbit, dynamics)
        out = dynamics.propagate(
            orbit.states[0],
            (modes.t_origin, modes.t_origin + modes.period),
            t_eval=modes.times,
        )
        source = InjectedModalSource(
            build_nominal_orbit(modes, np.asarray(out["states"], dtype=float))
        )
        online = OnlineModalSource(modes, dynamics)
        dual = dual_basis(modes)
        for k in (0, 50, 130, 190):
            got = source.dual_basis_at(modes.times[k])
            np.testing.assert_allclose(got[1:], dual[k][1:], rtol=0.0, atol=1e-9)
            # 在线重构与注入表互相独立（传播路径 vs 采样表），同相位一致。
            np.testing.assert_allclose(
                online.dual_basis_at(modes.times[k])[1:], got[1:], rtol=1e-8, atol=1e-8
            )

    def test_tangent_row_corrected_across_laps(self, dro21, modes21) -> None:
        """跨圈查表对切向行按整圈数乘每圈增长系数修正，与在线真值一致。"""
        epochs, states = _nominal_table(dro21, modes21)
        source = InjectedModalSource(build_nominal_orbit(modes21, states))
        in_cycle = source.dual_basis_at(epochs[60])
        wrapped = source.dual_basis_at(epochs[60] + 2.0 * modes21.period)
        lap_shift = float(
            (modes21.basis[-1][:, 5] - modes21.basis[0][:, 5]) @ modes21.basis[0][:, 0]
        )
        np.testing.assert_allclose(
            wrapped[0], in_cycle[0] - 2.0 * lap_shift * in_cycle[5], rtol=0.0, atol=1e-9
        )
        online = OnlineModalSource(modes21, dro21[1])
        truth = online.dual_basis_at(epochs[60] + 2.0 * modes21.period)
        np.testing.assert_allclose(wrapped[0], truth[0], rtol=1e-8, atol=1e-8)

    def test_missing_floquet_basis_rejected(self) -> None:
        """floquet_basis 未注入的容器无法做跨圈切向修正，确定性报错。"""
        orbit = NominalOrbit(
            epochs=np.linspace(0.0, 1.0, 11),
            states=np.zeros((11, 6)),
            projection_factors=np.zeros((11, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="floquet_basis"):
            InjectedModalSource(orbit)

    def test_single_epoch_table_rejected(self) -> None:
        """单历元表跨距为零，无法回卷查表，确定性报错。"""
        orbit = NominalOrbit(
            epochs=np.zeros(1),
            states=np.zeros((1, 6)),
            floquet_basis=np.zeros((1, 6, 6)),
            projection_factors=np.zeros((1, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="至少"):
            InjectedModalSource(orbit)

    def test_nonfinite_epochs_rejected(self) -> None:
        """历元含 NaN 时确定性报错，不静默产出 NaN 查表。"""
        epochs = np.linspace(0.0, 1.0, 11)
        epochs[4] = np.nan
        orbit = NominalOrbit(
            epochs=epochs,
            states=np.zeros((11, 6)),
            floquet_basis=np.zeros((11, 6, 6)),
            projection_factors=np.zeros((11, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="有限"):
            InjectedModalSource(orbit)

    def test_non_monotonic_epochs_rejected(self) -> None:
        """历元非严格递增时确定性报错。"""
        epochs = np.linspace(0.0, 1.0, 11)
        epochs[4] = epochs[3]
        orbit = NominalOrbit(
            epochs=epochs,
            states=np.zeros((11, 6)),
            floquet_basis=np.zeros((11, 6, 6)),
            projection_factors=np.zeros((11, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="严格递增"):
            InjectedModalSource(orbit)

    def test_missing_projection_factors_rejected(self) -> None:
        """projection_factors 未注入的容器确定性报错。"""
        orbit = NominalOrbit(epochs=np.linspace(0.0, 1.0, 11), states=np.zeros((11, 6)))
        with pytest.raises(FloquetModeError, match="projection_factors"):
            InjectedModalSource(orbit)

    def test_shape_mismatch_rejected(self) -> None:
        """历元数量与对偶基表行数不一致时确定性报错。"""
        orbit = NominalOrbit(
            epochs=np.linspace(0.0, 1.0, 11),
            states=np.zeros((11, 6)),
            floquet_basis=np.zeros((5, 6, 6)),
            projection_factors=np.zeros((5, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="对齐"):
            InjectedModalSource(orbit)

    def test_basis_dual_shape_mismatch_rejected(self) -> None:
        """两张模态表行数互不一致时确定性报错（同形校验）。"""
        orbit = NominalOrbit(
            epochs=np.linspace(0.0, 1.0, 11),
            states=np.zeros((11, 6)),
            floquet_basis=np.zeros((5, 6, 6)),
            projection_factors=np.zeros((11, 6, 6)),
        )
        with pytest.raises(FloquetModeError, match="同形"):
            InjectedModalSource(orbit)


class TestSourceEquivalence:
    """注入表与在线数值投影两种来源的等价。"""

    def test_dual_agrees_at_nodes(self, dro21, modes21) -> None:
        """表节点上两种来源的对偶基一致（积分容差量级）。"""
        epochs, states = _nominal_table(dro21, modes21)
        table_source = InjectedModalSource(build_nominal_orbit(modes21, states))
        online = OnlineModalSource(modes21, dro21[1])
        for k in (10, 70, 130, 190):
            np.testing.assert_allclose(
                table_source.dual_basis_at(epochs[k]),
                online.dual_basis_at(epochs[k]),
                rtol=1e-8,
                atol=1e-8,
            )

    def test_interpolation_error_second_order(self, dro21) -> None:
        """非节点查表误差随网格加密按二阶收敛（对拍在线重算值）。

        探针取粗网格（101 点）首区间中点：它同时是细网格（301 点）第二
        区间的中点，两张表在同相位比例位置插值，误差比应为 9（(1/3)²
        的倒数），容差放宽到 7 至 11 吸收曲率起伏。
        """
        orbit, dynamics = dro21
        coarse = floquet_modes(orbit, dynamics, times=np.linspace(0.0, float(orbit.period), 101))
        fine = floquet_modes(orbit, dynamics, times=np.linspace(0.0, float(orbit.period), 301))
        probe = 0.5 * float(orbit.period) / 100.0
        truth = OnlineModalSource(fine, dynamics).dual_basis_at(probe)
        errors = []
        for modes in (coarse, fine):
            source = InjectedModalSource(
                build_nominal_orbit(
                    modes,
                    np.zeros((modes.times.shape[0], 6)),  # 状态表仅占位，查表只读对偶基
                )
            )
            errors.append(float(np.max(np.abs(source.dual_basis_at(probe) - truth))))
        ratio = errors[0] / errors[1]
        assert 7.0 < ratio < 11.0, f"插值误差未按二阶收敛，误差比 {ratio:.2f}"

    def test_delta_v_sequences_agree(self, dro21, modes21) -> None:
        """端到端：两种模态量来源的控制序列一致（积分容差量级）。

        控制节点取表节点及其加两个周期的平移（覆盖模 T 回卷），偏差由
        摄动初值的真实传播给出，标称状态经状态表视图插值。
        """
        orbit, dynamics = dro21
        epochs, states = _nominal_table(dro21, modes21)
        view = TableNominalView(epochs, states)
        table_source = InjectedModalSource(build_nominal_orbit(modes21, states))
        online = OnlineModalSource(modes21, dynamics)
        law = FloquetTargetPointLaw()

        nodes = np.array([epochs[k] for k in (10, 60, 110, 160)])
        nodes = np.concatenate([nodes, nodes + 2.0 * modes21.period])
        eps = 1e-6
        deviation_seed = np.array([1e-3, -2e-3, 0.0, 0.0, 5e-4, 0.0])
        perturbed = np.asarray(
            dynamics.propagate(
                orbit.states[0] + eps * deviation_seed,
                (modes21.t_origin, float(nodes[-1])),
                t_eval=nodes,
            )["states"],
            dtype=float,
        )

        for k, t in enumerate(nodes):
            dv_table = law.compute_maneuver(
                perturbed[k], float(t), nominal=view, modal=table_source
            )
            dv_online = law.compute_maneuver(perturbed[k], float(t), nominal=view, modal=online)
            np.testing.assert_allclose(dv_table, dv_online, rtol=1e-8, atol=1e-10)
