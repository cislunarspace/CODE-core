"""porkchop Rust 后端（#446）：规格路径等价性、串/并位级一致、路由与错误语义。

porkchop 的数值网格评估（Lambert + ΔV + 分发）全部在 Rust；Python 只做
问题构造：内置终端（``OrbitTerminal``/``StateTerminal``）+ CR3BP 动力学时
把终端规格直接交给 Rust（终端传播同步下沉）；其余终端经
``get_arrival_state`` 协议提取状态网格后交给同一 Rust 评估核。本文件验证：

- 规格路径与协议路径逐点逐位一致——规格路径逐式复现
  ``propagate_orbit_state_at_time`` 的周期取模、`linspace` 与传播参数，
  两条路径共用同一 Rust Lambert/ΔV 核。
- ``E2M2E_PORKCHOP_PARALLEL=0`` 强制串行，与并行执行逐位一致。
- monkeypatch / 自定义子类路由到协议路径，patch 语义生效。
- 扩展符号缺失抛 ``RustExtensionUnavailableError``（#378，无静默回退）。
- 无效轨道（无周期）保留原 ``ValueError``。
"""

from __future__ import annotations

import numpy as np
import pytest

# 扩展未构建时（doc build 等场景）整模块跳过。
pytest.importorskip("e2m2e._integrators")

from e2m2e import integrators
from e2m2e.algorithm.transfer.porkchop import (
    PorkchopData,
    _builtin_grid_spec,
    porkchop,
)
from e2m2e.algorithm.transfer.terminal import OrbitTerminal, StateTerminal, TerminalCondition
from e2m2e.data.types.orbit import Orbit
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = pytest.mark.orchestration

# Lambert 中心天体 GM：CR3BP 无量纲单位下的示意值。等价性对照不依赖物理
# 合理性；网格内容许无解组合，以同时覆盖 NaN 分支。
MU_CENTRAL = 1.0

T_DEP = np.linspace(0.0, 1.5, 4)
TOF = np.linspace(0.5, 3.0, 5)


@pytest.fixture
def porkchop_dynamics(cr3bp_dynamics):
    """网格扫描用的 CR3BP 动力学（显式容差与步长，控制测试耗时）。"""
    cr3bp_dynamics.rtol = cr3bp_dynamics.atol = 1e-9
    cr3bp_dynamics.max_step = 0.05
    return cr3bp_dynamics


@pytest.fixture
def orbit_terminals(corrected_dro):
    """出发/到达终端均为同一修正后 DRO（不同相位的几何组合）。"""
    return OrbitTerminal(corrected_dro), OrbitTerminal(corrected_dro)


def _assert_grid_equal(data: PorkchopData, ref: PorkchopData) -> None:
    for field in ("dv1", "dv2", "total"):
        actual = getattr(data, field)
        expected = getattr(ref, field)
        assert np.array_equal(actual, expected, equal_nan=True), f"{field} 逐位结果不一致"


class TestSpecPathEquivalence:
    """规格路径（内置终端 + CR3BP）与协议路径（get_arrival_state 提取）等价。

    类级 monkeypatch 一个行为恒等的 wrapper 即迫使路由走协议路径——
    既充当对照基准，又验证 patch 检测本身。
    """

    def test_orbit_terminal_grid_matches_protocol_path(
        self, orbit_terminals, porkchop_dynamics, monkeypatch
    ):
        dep, arr = orbit_terminals
        assert _builtin_grid_spec(dep, arr, porkchop_dynamics) is not None
        data = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)

        original = OrbitTerminal.get_arrival_state

        def passthrough(self, t_ins, dynamics):
            return original(self, t_ins, dynamics)

        monkeypatch.setattr(OrbitTerminal, "get_arrival_state", passthrough)
        assert _builtin_grid_spec(dep, arr, porkchop_dynamics) is None
        ref = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)

        _assert_grid_equal(data, ref)

    def test_state_terminal_grid_matches_protocol_path(self, monkeypatch):
        dep = StateTerminal([0.8, 0.0, 0.0, 0.0, 0.5, 0.0], time=0.0)
        arr = StateTerminal([1.0, 0.2, 0.1, 0.0, 0.3, 0.0], time=0.0)
        assert _builtin_grid_spec(dep, arr, None) is not None
        data = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=None)

        original = StateTerminal.get_arrival_state

        def passthrough(self, t_ins, dynamics):
            return original(self, t_ins, dynamics)

        monkeypatch.setattr(StateTerminal, "get_arrival_state", passthrough)
        assert _builtin_grid_spec(dep, arr, None) is None
        ref = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=None)

        _assert_grid_equal(data, ref)

    def test_serial_parallel_bit_identical(self, orbit_terminals, porkchop_dynamics, monkeypatch):
        """E2M2E_PORKCHOP_PARALLEL=0 强制串行，与并行执行逐位一致。"""
        dep, arr = orbit_terminals
        monkeypatch.setenv("E2M2E_PORKCHOP_PARALLEL", "0")
        serial = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)
        monkeypatch.setenv("E2M2E_PORKCHOP_PARALLEL", "1")
        parallel = porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)
        for field in ("dv1", "dv2", "total"):
            assert np.array_equal(
                getattr(serial, field), getattr(parallel, field), equal_nan=True
            ), f"{field} 串/并结果不逐位一致"


class TestRouting:
    """终端/动力学不满足规格路径条件时，走协议提取路径，patch 语义生效。"""

    def test_instance_level_monkeypatch_takes_effect(
        self, orbit_terminals, porkchop_dynamics, monkeypatch
    ):
        """实例级 patch 必须被协议路径执行，而不是被 Rust 规格路径绕过。"""
        dep, arr = orbit_terminals
        calls = []
        original = OrbitTerminal.get_arrival_state

        def spy(t_ins, dynamics):
            calls.append(t_ins)
            return original(dep, t_ins, dynamics)

        monkeypatch.setattr(dep, "get_arrival_state", spy)
        assert _builtin_grid_spec(dep, arr, porkchop_dynamics) is None
        porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)
        # 出发终端按 t_dep 逐点调用（协议路径）
        assert sorted(calls) == sorted(float(t) for t in T_DEP)

    def test_custom_terminal_subclass_uses_protocol_path(self, porkchop_dynamics):
        calls = []

        class CustomTerminal(TerminalCondition):
            def get_initial_state(self):
                return np.zeros(6)

            def get_arrival_state(self, t_ins, dynamics):
                calls.append(float(t_ins))
                return np.array([0.8, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])

        term = CustomTerminal()
        fixed = StateTerminal([1.0, 0.2, 0.1, 0.0, 0.3, 0.0], time=0.0)
        data = porkchop(
            term,
            fixed,
            T_DEP,
            TOF,
            mu=MU_CENTRAL,
            dynamics=porkchop_dynamics,
        )
        assert data.total.shape == (T_DEP.shape[0], TOF.shape[0])
        assert sorted(calls) == sorted(float(t) for t in T_DEP)

    def test_empty_departure_grid_keeps_empty_result(self):
        class CustomTerminal(TerminalCondition):
            def get_initial_state(self):
                return np.zeros(6)

            def get_arrival_state(self, t_ins, dynamics):
                raise AssertionError("空出发网格不应提取终端状态")

        dep = CustomTerminal()
        arr = CustomTerminal()
        data = porkchop(dep, arr, [], TOF, mu=MU_CENTRAL, dynamics=None)
        assert data.dv1.shape == (0, TOF.shape[0])
        assert data.dv2.shape == (0, TOF.shape[0])
        assert data.total.shape == (0, TOF.shape[0])

    def test_invalid_orbit_keeps_value_error(self, porkchop_dynamics):
        """无效轨道（无周期）走协议路径，保留原 ValueError 语义。"""
        orbit = Orbit(states=np.zeros((1, 6)), times=[0.0])
        term = OrbitTerminal(orbit)
        assert _builtin_grid_spec(term, term, porkchop_dynamics) is None
        with pytest.raises(ValueError, match="轨道周期无效"):
            porkchop(term, term, [0.0], [1.0], mu=MU_CENTRAL, dynamics=porkchop_dynamics)
        # 原路径先提取终端状态、后校验 Lambert 方向；双重无效时维持该优先级。
        with pytest.raises(ValueError, match="轨道周期无效"):
            porkchop(
                term,
                term,
                [0.0],
                [1.0],
                mu=MU_CENTRAL,
                dynamics=porkchop_dynamics,
                direction="invalid",
            )


class TestExtensionMissing:
    """扩展符号缺失即抛 RustExtensionUnavailableError（#378），无静默回退。"""

    @pytest.mark.parametrize("symbol", ["porkchop_grid_py", "porkchop_grid_states_py"])
    def test_missing_symbol_raises(self, orbit_terminals, porkchop_dynamics, monkeypatch, symbol):
        dep, arr = orbit_terminals
        monkeypatch.setattr(integrators, symbol, None)
        with pytest.raises(RustExtensionUnavailableError):
            porkchop(dep, arr, T_DEP, TOF, mu=MU_CENTRAL, dynamics=porkchop_dynamics)
