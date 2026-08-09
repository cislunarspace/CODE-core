"""坐标模块公开 API 与冒烟测试。

验证顶层导出符号可用性与标准工厂函数。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate import (
    Axes,
    CoordinateSystem,
    GMATITRFAxes,
    ICRSAxes,
    InertialOrigin,
    ITRFApproxAxes,
    ITRFAxes,
    ITRFSpiceAxes,
    Origin,
    standard_itrf,
)

pytestmark = pytest.mark.data


def test_public_coordinate_exports_are_available():
    assert Axes is not None
    assert Origin is not None
    assert CoordinateSystem is not None
    assert ITRFSpiceAxes is not None
    assert ITRFAxes is not None
    assert ITRFApproxAxes is not None
    assert GMATITRFAxes is not None


def test_standard_itrf_defaults_to_spice_itrf93():
    axes = standard_itrf()

    assert isinstance(axes, ITRFSpiceAxes)
    assert axes.frame == "ITRF93"
    assert isinstance(ITRFAxes(), ITRFSpiceAxes)
    assert ITRFAxes().frame == "ITRF93"


def test_explicit_gmat_itrf_runs_with_committed_fixtures():
    axes = GMATITRFAxes(eop_extrapolation="clamp")
    rotation, rate = axes.rotation_and_rate(0.0)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    assert rate.shape == (3, 3)


def test_coordinate_smoke_with_explicit_gmat_axes_round_trips_vector():
    icrf = CoordinateSystem(ICRSAxes(), InertialOrigin())
    gmat_itrf = CoordinateSystem(GMATITRFAxes(compatibility="gmat"), InertialOrigin())
    vector = np.array([1.0, 2.0, 3.0])

    transformed = icrf.transform_vector(vector, from_cs=icrf, to_cs=gmat_itrf, et=0.0)
    result = gmat_itrf.transform_vector(transformed, from_cs=gmat_itrf, to_cs=icrf, et=0.0)

    np.testing.assert_allclose(result, vector, atol=1e-12)
