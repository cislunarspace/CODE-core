"""五周期共振定位与分岔支起链的行为测试（#820）。

定位与起链走真实链路（定位沿 DRO 族十余次修正与单值矩阵传播，起链
穷举 deflection 组合），按模块缓存共振点。验收全部定义性：闭合残差、
Jacobi 漂移、平面约束、单圈不闭合。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.family.resonant_branch import (
    CRITICAL_MULTIPLIER_TOLERANCE,
    ResonancePoint,
    generate_resonant_branch_family,
    locate_dro_revolution_resonance,
    per_revolution_closure,
)
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


@pytest.fixture(scope="module")
def dynamics():
    return CR3BP_Dynamics(earth_moon_system())


@pytest.fixture(scope="module")
def resonance(dynamics):
    return locate_dro_revolution_resonance(dynamics, revolutions=5)


def test_located_resonance_hits_target_phase(resonance):
    """二分定位收敛到 72 度目标辐角，乘子偏差进门限，特征向量平面占优。"""
    assert abs(resonance.phase_deg - 72.0) < 0.5
    assert resonance.multiplier_gap < CRITICAL_MULTIPLIER_TOLERANCE
    vector = resonance.eigenvector
    assert max(abs(vector[2].real), abs(vector[5].real)) < 0.2
    assert max(abs(vector[2].imag), abs(vector[5].imag)) < 0.2
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, rel=1e-9)


def test_branch_members_close_and_leave_trivial_family(dynamics, resonance):
    """起链生成的成员五圈闭合达标、留在平面内、单圈偏差沿链增长。"""
    result = generate_resonant_branch_family(
        resonance,
        n_orbits=3,
        step_size=0.005,
        initial_direction="increase-x0",
        dynamics=dynamics,
    )
    assert result.status.value == "converged", result.message
    members = list(result.family.orbits)
    assert len(members) == 4
    deviations = []
    for orbit in members:
        state = orbit.states[0]
        assert abs(state[2]) < 1e-12
        assert abs(state[5]) < 1e-12
        assert orbit.closure_error is not None
        assert float(orbit.closure_error) < 1e-8
        assert float(orbit.parameters["jacobi_drift"]) < 1e-9
        deviations.append(per_revolution_closure(dynamics, orbit, 5))
    assert deviations[0] > 1e-8
    assert deviations[-1] > deviations[0]


def test_gate_rejects_far_from_critical(dynamics, resonance):
    """临界乘子偏差超门限的平凡成员处起链被显式拒绝。"""
    far = ResonancePoint(
        state=resonance.state,
        period=resonance.period,
        x0=resonance.x0,
        phase_deg=resonance.phase_deg,
        multiplier_gap=0.9,
        eigenvector=resonance.eigenvector,
        revolutions=5,
    )
    with pytest.raises(Exception, match="起链不可行"):
        generate_resonant_branch_family(
            far,
            n_orbits=1,
            step_size=0.005,
            initial_direction="increase-x0",
            dynamics=dynamics,
        )


def test_per_revolution_closure_distinguishes_branch_from_trivial(dynamics, resonance):
    """平凡族成员单圈即闭合，分岔支成员单圈末端不回到初态。"""
    trivial = Orbit(
        states=np.asarray(resonance.state, dtype=float).reshape(1, -1),
        times=np.array([0.0]),
        system=dynamics.system,
    )
    # 平凡成员在 n 圈形式下的闭合周期是 5T（每圈都闭合），分岔支的
    # 闭合周期是 nT_branch（单圈不闭合）。两者单圈度量的对比才有意义。
    trivial.period = 5.0 * float(resonance.period)
    assert per_revolution_closure(dynamics, trivial, 5) < 1e-7

    result = generate_resonant_branch_family(
        resonance,
        n_orbits=0,
        step_size=0.005,
        initial_direction="increase-x0",
        dynamics=dynamics,
    )
    branch_seed = result.family.orbits[0]
    assert per_revolution_closure(dynamics, branch_seed, 5) > 1e-6
