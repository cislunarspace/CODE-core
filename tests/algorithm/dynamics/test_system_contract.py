"""System 抽象接口与两个系统实现的共同契约测试。"""

import pytest

from e2m2e.algorithm.dynamics import CR3BP_System, EphemerisSystem, System
from e2m2e.data.constants import Datum
from e2m2e.mbse.data.enums import ReferenceFrame, UnitSystem

pytestmark = pytest.mark.interface


class FakeSpiceManager:
    """只提供 EphemerisSystem 所需接口的测试适配器。"""

    def get_gm(self, body: str) -> float:
        return {
            "EARTH": Datum.DE440.earth_gm,
            "MOON": Datum.DE440.moon_gm,
        }[body]

    def get_body_position(self, body: str, et: float, frame: str, origin: str):  # noqa: ARG002
        return [0.0, 0.0, 0.0]

    def get_body_state(self, body: str, et: float, frame: str, origin: str):  # noqa: ARG002
        return [0.0] * 6


@pytest.fixture
def cr3bp_system():
    return CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")


@pytest.fixture
def ephemeris_system():
    return EphemerisSystem(
        bodies=["EARTH", "MOON"],
        spice=FakeSpiceManager(),
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )


def test_system_is_abstract():
    with pytest.raises(TypeError):
        System()  # type: ignore[abstract]


def test_concrete_systems_implement_system(cr3bp_system, ephemeris_system):
    assert isinstance(cr3bp_system, System)
    assert isinstance(ephemeris_system, System)


def test_cr3bp_system_interface(cr3bp_system):
    assert cr3bp_system.frame is ReferenceFrame.SYNODIC
    assert cr3bp_system.unit_system is UnitSystem.DIMENSIONLESS
    assert cr3bp_system.gravitational_parameter("primary") == pytest.approx(1 - cr3bp_system.mu)
    assert cr3bp_system.gravitational_parameter("secondary") == pytest.approx(cr3bp_system.mu)


def test_ephemeris_system_interface(ephemeris_system):
    assert ephemeris_system.frame is ReferenceFrame.J2000
    assert ephemeris_system.unit_system is UnitSystem.SI
    assert ephemeris_system.gravitational_parameter("EARTH") == pytest.approx(Datum.DE440.earth_gm)
    assert ephemeris_system.gravitational_parameter("MOON") == pytest.approx(Datum.DE440.moon_gm)


def test_invalid_body_names_are_rejected(cr3bp_system):
    with pytest.raises(ValueError):
        cr3bp_system.gravitational_parameter("Jupiter")
