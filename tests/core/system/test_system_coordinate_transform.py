"""坐标系最小 System 委托测试。"""

import numpy as np

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.gmat_data import CoordinateDataError
from e2m2e.core.standard_axes import GMATITRFAxes, ICRSAxes, ITRFSpiceAxes
from e2m2e.core.standard_origins import InertialOrigin
from e2m2e.core.system import System
from e2m2e.mbse.data.enums import ReferenceFrame, UnitSystem


class ToySystem(System):
    def __init__(self, coordinate_system=None):
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


def test_system_transform_delegates_to_coordinate_system():
    coordinate_system = CoordinateSystem(ICRSAxes(), InertialOrigin())
    system = ToySystem(coordinate_system)
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    result = system.transform(state, to_coordinate_system=coordinate_system, et=0.0)

    np.testing.assert_allclose(result, state)

def test_system_transform_default_itrf_matches_direct_coordinate_system():
    source = CoordinateSystem(ICRSAxes(), InertialOrigin())
    target = CoordinateSystem(ITRFSpiceAxes(), InertialOrigin())
    system = ToySystem(source)
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    try:
        result = system.transform(state, to_coordinate_system=target, et=0.0)
        expected = source.transform_state(state, from_cs=source, to_cs=target, et=0.0)
    except CoordinateDataError:
        target = CoordinateSystem(GMATITRFAxes(eop_extrapolation="clamp"), InertialOrigin())
        result = system.transform(state, to_coordinate_system=target, et=0.0)
        expected = source.transform_state(state, from_cs=source, to_cs=target, et=0.0)

    np.testing.assert_allclose(result, expected)


def test_system_transform_requires_default_coordinate_system():
    system = ToySystem()

    try:
        system.transform(
            np.zeros(6),
            to_coordinate_system=CoordinateSystem(ICRSAxes(), InertialOrigin()),
            et=0.0,
        )
    except ValueError as exc:
        assert "coordinate_system" in str(exc)
    else:
        raise AssertionError("expected ValueError")
