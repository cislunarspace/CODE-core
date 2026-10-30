"""DPO 主族单圈分支回归（issue #841）。

旧种子（x0=0.90，T=2.5022）本身是双圈解，族行走从小振幅端就沿多圈
分支收敛：20000 km 返回 7 圈轨道（T≈23.3 天），21000/22000 km 返回
3/4 圈。主族单圈分支的自然参数延拓给出振幅折叠极值 27752 km，族行走
在 27500 km 仍收敛；设计请求包络取 27000 km，由 design_orbit 的端到端
用例覆盖。本文件钉住：20000 km 返回单圈主族成员、族行走在接近折叠
极值的 27500 km 仍收敛、主族成员整周期绕月恰好一圈，以及多圈解与
周期突变两条拒绝分支确实被触发。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_dpo
from e2m2e.algorithm.family.orbits.dro import _correct_dpo, _dpo_winding_number
from e2m2e.algorithm.family.orbits.walk import (
    Cr3bpOrbitError,
    _moon_distance_minmax,
    earth_moon_system,
)
from e2m2e.algorithm.family.strategies import symmetric_2d_fixed_x0
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.templates.seed import _DPO_SEED_PERIOD, _DPO_SEED_VY0, _DPO_SEED_X0
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


def _seed_guess(dynamics: CR3BP_Dynamics, period: float) -> Orbit:
    """种子状态配指定周期，用于构造拒绝分支的初猜。"""
    state = np.array([_DPO_SEED_X0, 0.0, 0.0, 0.0, _DPO_SEED_VY0, 0.0])
    orbit = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=dynamics.system)
    orbit.period = period
    return orbit


def test_20000_km_returns_single_rev_principal_member() -> None:
    """20000 km 是 DPO 请求默认振幅，必须落在主族单圈分支（#841）。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    lu = dynamics.system.characteristic_length
    assert lu is not None

    orbit = design_dpo(20_000.0, dynamics=dynamics)
    d_min, d_max = _moon_distance_minmax(dynamics, orbit)
    assert abs(0.5 * (d_min + d_max) * lu - 20_000.0) <= 20.0
    assert abs(_dpo_winding_number(orbit, 1.0 - orbit.system.mu) - 1.0) < 0.05
    assert orbit.period is not None
    # 主族 20000 km 成员周期约 0.77 TU（3.33 天）；多圈伪解在 5 TU 以上，
    # 另一条单圈分支周期明显更长
    assert 0.70 < orbit.period < 0.85
    assert orbit.closure_error is not None
    assert orbit.closure_error <= 1.0e-10


def test_near_upper_bound_27500_km_converges() -> None:
    """族行走在 27500 km 收敛，距折叠极值 27752 km 约 250 km。"""
    dynamics = CR3BP_Dynamics(earth_moon_system())
    lu = dynamics.system.characteristic_length
    assert lu is not None

    orbit = design_dpo(27_500.0, dynamics=dynamics)
    d_min, d_max = _moon_distance_minmax(dynamics, orbit)
    assert abs(0.5 * (d_min + d_max) * lu - 27_500.0) <= 20.0
    assert abs(_dpo_winding_number(orbit, 1.0 - orbit.system.mu) - 1.0) < 0.05
    # 近上界成员已相当偏心（近月点约 7000 km），仍在月面以上
    assert d_min * lu > 1737.0
    assert orbit.closure_error is not None
    assert orbit.closure_error <= 1.0e-10


def test_multirev_solution_is_rejected_by_winding_guard() -> None:
    """两圈解必须按圈数判据拒绝（#841）。

    种子状态在自身周期后重复，半周期取一个完整周期时同一初值就是两圈
    对称周期解。先用修正器直接收敛这条解，确认整周期绕月两圈，再断言
    ``_correct_dpo`` 把它按伪解拒绝。
    """
    dynamics = CR3BP_Dynamics(earth_moon_system())
    guess = _seed_guess(dynamics, 2.0 * _DPO_SEED_PERIOD)
    corrector = DifferentialCorrection(dynamics)
    corrector.configure(symmetric_2d_fixed_x0(x0=_DPO_SEED_X0))
    result = corrector.iterate_correction(initial_guess=guess)
    assert result.orbit is not None
    assert abs(_dpo_winding_number(result.orbit, 1.0 - dynamics.system.mu) - 2.0) < 0.05
    with pytest.raises(Cr3bpOrbitError, match="圈分支"):
        _correct_dpo(dynamics, _DPO_SEED_X0, guess)


def test_period_jump_guess_is_rejected() -> None:
    """收敛周期相对初猜突变超过 30% 必须按跳支拒绝（#841）。

    初猜周期取种子周期的 0.7 倍，初值本身仍在主族单圈解上，修正后周期
    回到种子值，比值 1.43，判为跳到另一条单圈分支。
    """
    dynamics = CR3BP_Dynamics(earth_moon_system())
    guess = _seed_guess(dynamics, 0.7 * _DPO_SEED_PERIOD)
    with pytest.raises(Cr3bpOrbitError, match="另一条单圈分支"):
        _correct_dpo(dynamics, _DPO_SEED_X0, guess)
