"""Rust 传播绑定的类型化失败契约测试。"""

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import BCR4BPSystem, CR3BP_Dynamics, CR3BP_System
from e2m2e.data.constants import Datum
from e2m2e.exceptions import E2M2EError, PropagationFailure
from e2m2e.integrators import propagate_bcr4bp_py, propagate_cr3bp_py, propagate_with_stm_py

pytestmark = pytest.mark.integrator


def _collapsing_state() -> list[float]:
    return [1.0 - Datum.DE421.mu + 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_propagation_failure_has_the_domain_exception_type():
    assert issubclass(PropagationFailure, E2M2EError)
    assert not issubclass(PropagationFailure, RuntimeError)


def test_rust_ffi_translates_step_collapse_to_propagation_failure():
    with pytest.raises(PropagationFailure):
        propagate_cr3bp_py(
            mu=Datum.DE421.mu,
            t_span=(0.0, 2.0),
            t_eval=np.linspace(0.0, 2.0, 21).tolist(),
            initial_state=_collapsing_state(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_degenerate_span_returns_all_requested_points():
    """退化弧段（span < 1e-12）也必须返回全部请求点（#627 周期相位绕回）。

    DRO→RO 转移 NLP 把插入时间推到 RO 周期边界时，轨道相位重传播弧段
    宽 ~1e-13：传播器的时间界判据一步不走、末点不发射，报
    "output length mismatch"。修复后首末两点都应返回。
    """
    result = propagate_cr3bp_py(
        mu=Datum.DE421.mu,
        t_span=(0.0, 2.6e-13),
        t_eval=[0.0, 2.6e-13],
        initial_state=[0.8, 0.0, 0.0, 0.0, 0.5, 0.0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(result["time"]) == 2
    assert len(result["states"]) == 2
    # 末态与初态的差距在弧段宽度量级内（轨道未实际推进）
    delta = np.asarray(result["states"][1]) - np.asarray(result["states"][0])
    assert np.linalg.norm(delta) < 1e-9


def test_bcr4bp_rust_ffi_translates_step_collapse_to_propagation_failure():
    system = BCR4BPSystem.earth_moon()
    with pytest.raises(PropagationFailure):
        propagate_bcr4bp_py(
            mu=system.mu,
            mu_sun=system.sun_mass,
            sun_distance=system.sun_distance,
            sun_angular_rate=system.sun_angular_rate,
            sun_phase0=system.sun_phase0,
            t_span=(0.0, 2.0),
            t_eval=np.linspace(0.0, 2.0, 21).tolist(),
            initial_state=_collapsing_state(),
            rtol=1e-12,
            atol=1e-12,
            max_step=0.01,
        )


def test_public_cr3bp_propagation_does_not_hide_step_collapse_as_empty_states():
    dynamics = CR3BP_Dynamics(
        CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()
    )
    with pytest.raises(PropagationFailure):
        dynamics.propagate(
            _collapsing_state(),
            (0.0, 2.0),
            t_eval=np.linspace(0.0, 2.0, 21),
        )


def test_ffi_unknown_body_does_not_return_a_truncated_trajectory(spice_manager, reference_epoch):
    reference_et = spice_manager.utc_to_et(reference_epoch)
    state = [Datum.WGS84.earth_radius_km + 400.0, 0.0, 0.0, 0.0, 7.0, 0.0]
    with pytest.raises(RuntimeError, match="STM propagation failed"):
        propagate_with_stm_py(
            bodies=["EARTH", "FAKEBODY"],
            origin="EARTH",
            gm_values=[Datum.DE440.earth_gm, 1.0],
            t_span=(reference_et, reference_et + 100.0),
            t_eval=[reference_et, reference_et + 100.0],
            initial_state=state,
            rtol=1e-12,
            atol=1e-12,
            max_step=10.0,
        )
