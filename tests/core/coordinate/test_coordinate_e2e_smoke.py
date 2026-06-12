"""聚焦坐标 E2E smoke tests。"""

import numpy as np

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.gmat_data import CoordinateDataError
from e2m2e.core.standard_axes import GMATITRFAxes, ICRSAxes, ITRFSpiceAxes
from e2m2e.core.standard_origins import InertialOrigin
from e2m2e.core.system import System
from e2m2e.mbse.data.enums import ReferenceFrame, UnitSystem


class SmokeSystem(System):
    def __init__(self, coordinate_system):
        self._coordinate_system = coordinate_system

    @property
    def frame(self):
        return ReferenceFrame.INERTIAL

    @property
    def unit_system(self):
        return UnitSystem.DIMENSIONLESS

    @property
    def coordinate_system(self):
        return self._coordinate_system

    def gravitational_parameter(self, body: str) -> float:
        return 1.0


def test_default_system_itrf_smoke_matches_direct_coordinate_transform():
    source = CoordinateSystem(ICRSAxes(), InertialOrigin())
    target = CoordinateSystem(ITRFSpiceAxes(), InertialOrigin())
    system = SmokeSystem(source)
    state = np.array([7000.0, 20.0, -30.0, 0.1, 7.4, 0.2])

    try:
        result = system.transform(state, to_coordinate_system=target, et=0.0)
        expected = source.transform_state(state, from_cs=source, to_cs=target, et=0.0)
    except CoordinateDataError as exc:
        assert "ITRF93" in str(exc)
        return

    np.testing.assert_allclose(result, expected)


def test_explicit_gmat_itrf_smoke_with_committed_fixtures():
    source = CoordinateSystem(ICRSAxes(), InertialOrigin())
    target = CoordinateSystem(GMATITRFAxes(eop_extrapolation="clamp"), InertialOrigin())
    system = SmokeSystem(source)
    state = np.array([7000.0, 20.0, -30.0, 0.1, 7.4, 0.2])

    result = system.transform(state, to_coordinate_system=target, et=0.0)
    expected = source.transform_state(state, from_cs=source, to_cs=target, et=0.0)

    np.testing.assert_allclose(result, expected)
