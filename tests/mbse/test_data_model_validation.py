"""Pydantic 数据模型校验测试。

验证 OrbitProperties 提供合理的默认值。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from e2m2e.mbse.data.core_models import OrbitProperties

pytestmark = pytest.mark.aux


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

    def test_rejects_wrong_state_vector_shapes(self):
        with pytest.raises(ValueError, match="mean_state"):
            OrbitProperties(mean_state=np.zeros(5))
        with pytest.raises(ValueError, match="center"):
            OrbitProperties(center=np.zeros((3, 1)))

    def test_rejects_non_array_state_vectors(self):
        with pytest.raises(ValidationError):
            OrbitProperties(mean_state="not an array")
