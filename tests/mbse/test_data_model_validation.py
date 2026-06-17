"""Pydantic 数据模型校验测试。

验证 PropagationResult 状态形状与 STM 维度。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from e2m2e.mbse.data.core_models import OrbitProperties, OrbitStability, PropagationResult


class TestPropagationResult:
    """PropagationResult must validate state shapes correctly."""

    def test_validates_states_shape(self):
        states = np.random.randn(100, 6)
        result = PropagationResult(
            time=np.linspace(0, 1, 100),
            states=states,
        )
        assert result.states.shape == (100, 6)

    def test_rejects_wrong_states_shape(self):
        with pytest.raises(ValidationError):
            PropagationResult(
                time=np.linspace(0, 1, 100),
                states=np.random.randn(6, 100),  # wrong shape
            )

    def test_rejects_1d_states(self):
        with pytest.raises(ValidationError):
            PropagationResult(
                time=np.linspace(0, 1, 100),
                states=np.random.randn(100),
            )

    def test_accepts_stm_when_correct_shape(self):
        states = np.random.randn(50, 6)
        stm = np.random.randn(50, 6, 6)
        result = PropagationResult(
            time=np.linspace(0, 1, 50),
            states=states,
            stm=stm,
        )
        assert result.stm.shape == (50, 6, 6)

    def test_rejects_wrong_stm_shape(self):
        states = np.random.randn(50, 6)
        with pytest.raises(ValidationError):
            PropagationResult(
                time=np.linspace(0, 1, 50),
                states=states,
                stm=np.random.randn(50, 6, 5),  # wrong shape
            )

    def test_stm_defaults_to_none(self):
        states = np.random.randn(50, 6)
        result = PropagationResult(
            time=np.linspace(0, 1, 50),
            states=states,
        )
        assert result.stm is None

    def test_jacobi_error_defaults_to_zero(self):
        states = np.random.randn(50, 6)
        result = PropagationResult(
            time=np.linspace(0, 1, 50),
            states=states,
        )
        assert result.jacobi_error == 0.0


class TestOrbitProperties:
    """OrbitProperties must provide sensible defaults."""

    def test_period_defaults_to_none(self):
        props = OrbitProperties()
        assert props.period is None

    def test_amplitudes_defaults_to_none(self):
        props = OrbitProperties()
        assert props.amplitudes is None

    def test_extrema_defaults_to_none(self):
        props = OrbitProperties()
        assert props.extrema is None

    def test_is_periodic_defaults_to_false(self):
        props = OrbitProperties()
        assert props.is_periodic is False

    def test_accepts_full_initialization(self):
        props = OrbitProperties(
            period=6.192,
            amplitudes={"x": 0.1, "y": 0.2, "z": 0.05},
            is_periodic=True,
            periodicity_error=1e-12,
        )
        assert props.period == 6.192
        assert props.is_periodic is True


class TestOrbitStability:
    """OrbitStability must validate monodromy matrix shape."""

    def test_monodromy_defaults_to_none(self):
        stab = OrbitStability()
        assert stab.monodromy_matrix is None

    def test_accepts_correct_monodromy(self):
        mono = np.eye(6)
        stab = OrbitStability(monodromy_matrix=mono)
        assert stab.stability is None

    def test_rejects_wrong_monodromy_shape(self):
        with pytest.raises(ValidationError):
            OrbitStability(monodromy_matrix=np.eye(5))

    def test_eigenvalues_optional(self):
        stab = OrbitStability(eigenvalues=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        assert stab.eigenvalues is not None
