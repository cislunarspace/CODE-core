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
