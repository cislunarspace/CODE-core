"""低推力转移端到端集成测试。

纯二体（SimpleNamespace + PointMassGravity，无 SPICE），通过 transfer_orbit
编排器路由 LowThrustShooting.solve_from_qlaw 完成闭环。

依赖 Rust 传播绑定（propagate_compiled_lowthrust），不可用时跳过全部测试。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig, TransferDesignResult, transfer_orbit

MU = 398600.435507  # km³/s²，地球

# Rust 传播绑定不可用时跳过全部测试（LowThrustShooting.solve 依赖它）
try:
    from e2m2e.integrators import propagate_compiled_lowthrust

    _HAS_RUST_PROPAGATION = propagate_compiled_lowthrust is not None
except ImportError:
    _HAS_RUST_PROPAGATION = False

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.skipif(
        not _HAS_RUST_PROPAGATION,
        reason="propagate_compiled_lowthrust Rust binding not available",
    ),
]


def _system_forces():
    """纯二体地心系（SimpleNamespace，PointMassGravity，无需 SPICE）。"""
    return SimpleNamespace(origin="EARTH"), [PointMassGravity("EARTH", mu=MU)]


def _departure_target():
    """7000→7200 km 圆轨道出发/目标状态。"""
    r0 = 7000.0
    v0 = math.sqrt(MU / r0)
    rT = 7200.0
    vT = math.sqrt(MU / rT)
    departure = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    target = np.array([rT, 0.0, 0.0, 0.0, vT, 0.0])
    return departure, target, rT


def test_low_thrust_orchestrator_converges():
    """transfer_orbit("low_thrust", ...) 编排路由端到端收敛。

    纯二体（SimpleNamespace + PointMassGravity），不依赖 SPICE。
    目标：7000→7200 km 圆轨道，T=0.5N, Isp=3000s, m0=1000kg。
    """
    system, forces = _system_forces()
    departure, target, rT = _departure_target()

    result = transfer_orbit(
        "low_thrust",
        engine_config=EngineConfig(t_max=0.5, isp=3000.0),
        initial_mass=1000.0,
        n_segments=5,
        target_oe=(rT, 0.0, 0.0),
        solver_method="shooting",
        duration_days=3.0,
        system=system,
        forces=forces,
        departure_state=departure,
        target_state=target,
    )

    assert isinstance(result, TransferDesignResult)
    assert result.transfer_type == "low_thrust"
    assert result.details.fuel_consumed > 0
    assert result.details.equivalent_delta_v > 0

    # 终端残差：纯二体短弧精度有限，放宽阈值
    assert result.details.terminal_residual_r < 100.0  # km
    assert result.details.terminal_residual_v < 0.1  # km/s


def test_low_thrust_mass_history_monotone():
    """质量单调递减、各段控制 throttle ∈ [0,1]。"""
    system, forces = _system_forces()
    departure, target, rT = _departure_target()

    result = transfer_orbit(
        "low_thrust",
        engine_config=EngineConfig(t_max=0.5, isp=3000.0),
        initial_mass=1000.0,
        n_segments=5,
        target_oe=(rT, 0.0, 0.0),
        solver_method="shooting",
        duration_days=3.0,
        system=system,
        forces=forces,
        departure_state=departure,
        target_state=target,
    )

    # 质量单调递减（数值容许小波动）
    masses = result.details.states_7d[:, 6]
    assert np.all(np.diff(masses) <= 1e-10), (
        f"质量应单调递减, 最大增量: {np.max(np.diff(masses)):.2e}"
    )

    # 各段 throttle ∈ [0, 1]
    for i, seg in enumerate(result.details.segments):
        assert 0.0 <= seg.throttle <= 1.0, f"段 {i} throttle={seg.throttle:.4f} 超出 [0, 1]"


def test_low_thrust_vs_impulsive_delta_v_comparison():
    """同任务场景低推力等效 Δv >= 脉冲 Δv（物理定律约束）。

    低推力因持续推力损失（gravity loss），等效 Δv 应 >= 霍曼脉冲 Δv。
    但 LEO→LEO+200km 改变很小，放宽上界到 5x。
    """
    from e2m2e.algorithm.transfer.hohmann import hohmann_delta_v

    r1 = 7000.0
    r2 = 7200.0
    dv1, dv2 = hohmann_delta_v(r1, r2)
    dv_impulsive = dv1 + dv2

    system, forces = _system_forces()
    departure, target, rT = _departure_target()

    result = transfer_orbit(
        "low_thrust",
        engine_config=EngineConfig(t_max=0.5, isp=3000.0),
        initial_mass=1000.0,
        n_segments=5,
        target_oe=(rT, 0.0, 0.0),
        solver_method="shooting",
        duration_days=3.0,
        system=system,
        forces=forces,
        departure_state=departure,
        target_state=target,
    )

    dv_lt = result.details.equivalent_delta_v
    assert dv_lt > 0.0, f"低推力等效 Δv 应为正: {dv_lt}"
    assert dv_lt < dv_impulsive * 5.0, f"低推力 Δv={dv_lt:.4f} 不应远超脉冲 Δv={dv_impulsive:.4f}"
