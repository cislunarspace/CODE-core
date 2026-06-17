"""correct_with_homotopy 参数校验与编排测试。

验证 lambda_steps 约束、中间步容差策略、
上一步输出作为下一步种子、以及聚合结果语义。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from e2m2e.algorithms import homotopy_correction


def _fake_dynamics():
    return SimpleNamespace(
        system=SimpleNamespace(
            bodies=["EARTH", "MOON", "SUN"],
            origin="EARTH",
            frame="J2000",
            spice=object(),
        ),
        rtol=1e-9, atol=1e-9, max_step=60.0, integrator="DOP853",
    )


def test_inner_method_homotopy_is_rejected():
    with pytest.raises(ValueError, match="inner_method='homotopy' is not allowed"):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=np.array([0.0, 100.0]),
            state_patch=np.ones((2, 6)),
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            inner_method="homotopy",
        )


def test_invalid_base_bodies_not_subset_raises():
    with pytest.raises(ValueError, match="subset of system.bodies"):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=np.array([0.0, 100.0]),
            state_patch=np.ones((2, 6)),
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "JUPITER"],  # JUPITER not in bodies
        )


def test_invalid_base_bodies_missing_origin_raises():
    with pytest.raises(ValueError, match="must include origin"):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=np.array([0.0, 100.0]),
            state_patch=np.ones((2, 6)),
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["MOON"],  # missing EARTH (origin)
        )


@pytest.mark.parametrize(
    "lambda_steps",
    [
        [],                              # empty
        [0.5],                           # missing terminal 1.0
        [0.5, 0.3],                      # not strictly increasing
        [0.5, 0.75, 1.1],                # > 1.0
        [-0.1, 0.5, 1.0],                # < 0.0
    ],
)
def test_invalid_lambda_steps_raises(lambda_steps):
    with pytest.raises(ValueError, match="lambda_steps"):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=np.array([0.0, 100.0]),
            state_patch=np.ones((2, 6)),
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=lambda_steps,
        )


def test_intermediate_steps_use_loose_tolerance_final_uses_strict():
    """Each non-final lambda step gets tolerance*10; the final step gets tolerance."""
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))
    captured_tols: list[float] = []

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            self.dynamics = dynamics

        def correct(self, **kwargs):
            captured_tols.append(kwargs["tolerance"])
            _ = self.dynamics.lambda_weight  # ensure attribute is readable
            return SimpleNamespace(
                converged=True, outer_iterations=1, max_residual=1e-12,
                residual_history=[1e-12],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.25, 0.5, 1.0],
        )

    assert captured_tols == [1e-7, 1e-7, 1e-8]


def test_each_step_seeded_with_previous_step_output():
    """Step N+1 must be initialized with step N's t_patch/state_patch."""
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))
    seeded: list[tuple[np.ndarray, np.ndarray]] = []

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            pass

        def correct(self, **kwargs):
            seeded.append((np.array(kwargs["t_patch"]), np.array(kwargs["state_patch"])))
            # Return a slightly perturbed result so we can check seeding
            return SimpleNamespace(
                converged=True, outer_iterations=1, max_residual=1e-12,
                residual_history=[1e-12],
                t_patch=kwargs["t_patch"] + 0.5,
                state_patch=kwargs["state_patch"] + 0.1,
            )

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 1.0],
        )

    assert len(seeded) == 2
    # step 0 uses original input
    np.testing.assert_array_equal(seeded[0][0], t_patch)
    np.testing.assert_array_equal(seeded[0][1], state_patch)
    # step 1 uses step 0's output
    np.testing.assert_array_equal(seeded[1][0], t_patch + 0.5)
    np.testing.assert_array_equal(seeded[1][1], state_patch + 0.1)


def test_failed_intermediate_step_still_seeds_next_step():
    """Even if a lambda step reports converged=False, its output is used to seed the next."""
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))
    seeded: list[tuple[np.ndarray, np.ndarray]] = []

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            pass

        def correct(self, **kwargs):
            seeded.append((np.array(kwargs["t_patch"]), np.array(kwargs["state_patch"])))
            return SimpleNamespace(
                converged=False,  # intermediate failure
                outer_iterations=5,
                max_residual=1.0e-3,
                residual_history=[1e-2, 1e-3],
                t_patch=kwargs["t_patch"] + 1.0,
                state_patch=kwargs["state_patch"] + 0.2,
            )

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 1.0],
        )

    # Both steps were attempted
    assert len(seeded) == 2
    # Aggregated: converged comes from the LAST step, residuals flatten
    assert result.converged is False
    assert result.iterations == 10
    assert result.residual_history == [1e-2, 1e-3, 1e-2, 1e-3]
    assert result.max_residual == 1.0e-3


def test_aggregated_fields_follow_spec():
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            pass

        def correct(self, **kwargs):
            return SimpleNamespace(
                converged=True, outer_iterations=2, max_residual=2.0e-9,
                residual_history=[1.0e-7, 2.0e-9],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.25, 0.5, 0.75, 1.0],
        )

    assert result.converged is True
    assert result.iterations == 2 * 4  # 2 iterations per step * 4 steps
    assert result.max_residual == 2.0e-9
    assert result.residual_history == [1.0e-7, 2.0e-9] * 4
    # Last step's output is returned (4 steps; t_patch +0.1, state_patch +0.01)
    np.testing.assert_allclose(result.t_patch, t_patch + 0.4)
    np.testing.assert_allclose(result.state_patch, state_patch + 0.04)
    # velocity_residual fields are left as None (not part of standard path)
    assert result.velocity_residual is None
    assert result.velocity_residual_history is None


# ---------------------------------------------------------------------------
# Failure-path coverage (Issue #240)
# ---------------------------------------------------------------------------

def test_final_step_nonconvergence_aggregates_to_converged_false():
    """If the final step returns converged=False, the aggregated result is converged=False.

    The aggregated max_residual matches the final step's max_residual, and
    residual_history still includes every step's contribution.
    """
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            pass

        def correct(self, **kwargs):
            return SimpleNamespace(
                converged=False, outer_iterations=10, max_residual=1.0e-5,
                residual_history=[1.0e-3, 1.0e-5],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 1.0],
        )

    assert result.converged is False
    assert result.max_residual == 1.0e-5  # final step's max_residual
    # residual_history is the concatenation of both steps' histories
    assert result.residual_history == [1.0e-3, 1.0e-5, 1.0e-3, 1.0e-5]
    assert result.iterations == 20


def test_inner_step_exception_raises_with_context():
    """When an inner step raises, the re-raised exception mentions the step.

    The context string should include the step index, the lambda value, and
    the inner method so that the failure is debuggable in isolation.
    """
    class ExplodingMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            self.dynamics = dynamics

        def correct(self, **kwargs):
            # Fail only when we are at lambda=0.75 (step 1)
            if abs(self.dynamics.lambda_weight - 0.75) < 1e-12:
                raise RuntimeError("upstream solver failure")
            return SimpleNamespace(
                converged=True, outer_iterations=1, max_residual=1.0e-9,
                residual_history=[1.0e-9],
                t_patch=kwargs["t_patch"] + 0.1,
                state_patch=kwargs["state_patch"] + 0.01,
            )

    with patch.object(homotopy_correction, "MultipleShooting", ExplodingMS), pytest.raises(
        RuntimeError, match=r"lambda step 1.*lambda=0\.75.*inner_method=standard"
    ):
        homotopy_correction.correct_with_homotopy(
                dynamics=_fake_dynamics(),
                t_patch=np.array([0.0, 100.0]),
                state_patch=np.ones((2, 6)),
                tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
                base_bodies=["EARTH", "MOON"],
                lambda_steps=[0.5, 0.75, 1.0],
                inner_method="standard",
            )

def test_residual_history_not_dropped_when_intermediate_step_fails():
    """residual_history accumulates all per-step histories, even if some steps are nonconverged.

    This is the observability contract: a user inspecting residual_history
    can see exactly what happened at every lambda step.
    """
    t_patch = np.array([0.0, 100.0])
    state_patch = np.ones((2, 6))
    per_step_results = iter([
        # step 0: converges
        SimpleNamespace(
            converged=True, outer_iterations=3, max_residual=1.0e-9,
            residual_history=[1.0e-7, 1.0e-8, 1.0e-9],
            t_patch=t_patch + 0.1, state_patch=state_patch + 0.01,
        ),
        # step 1: does NOT converge, but reports a residual history
        SimpleNamespace(
            converged=False, outer_iterations=5, max_residual=1.0e-4,
            residual_history=[1.0e-3, 1.0e-4],
            t_patch=t_patch + 0.2, state_patch=state_patch + 0.02,
        ),
        # step 2: final step
        SimpleNamespace(
            converged=True, outer_iterations=2, max_residual=1.0e-9,
            residual_history=[1.0e-8, 1.0e-9],
            t_patch=t_patch + 0.3, state_patch=state_patch + 0.03,
        ),
    ])

    class FakeMS:
        def __init__(self, dynamics, n_workers, kernel_dir):
            pass

        def correct(self, **kwargs):
            return next(per_step_results)

    with patch.object(homotopy_correction, "MultipleShooting", FakeMS):
        result = homotopy_correction.correct_with_homotopy(
            dynamics=_fake_dynamics(),
            t_patch=t_patch, state_patch=state_patch,
            tolerance=1e-8, max_iter=5, n_workers=1, kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
            lambda_steps=[0.5, 0.75, 1.0],
        )

    # Aggregated: final step converged → True
    assert result.converged is True
    # No history entries lost: 3 + 2 + 2 = 7 entries in order
    assert result.residual_history == [
        1.0e-7, 1.0e-8, 1.0e-9,    # step 0
        1.0e-3, 1.0e-4,            # step 1
        1.0e-8, 1.0e-9,            # step 2
    ]
    # final max_residual is from step 2
    assert result.max_residual == 1.0e-9
