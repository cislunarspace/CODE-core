"""Test shared enum definitions in mbse/data/enums.py."""

import pytest

from e2m2e.mbse.data.enums import (
    BifurcationLabel,
    ConvergenceState,
    OrbitFamilyType,
    ProjectionPlane,
    ReferenceFrame,
    StabilityLabel,
    TransferPhase,
    TransferType,
)


class TestUnitSystem:
    """UnitSystem 枚举验证。"""

    def test_unit_system_exists(self):
        """UnitSystem 枚举应该存在。"""
        from e2m2e.mbse.data.enums import UnitSystem

        assert UnitSystem is not None

    def test_unit_system_has_dimensionless(self):
        """UnitSystem 应有 DIMENSIONLESS 成员。"""
        from e2m2e.mbse.data.enums import UnitSystem

        assert UnitSystem.DIMENSIONLESS.value == "dimensionless"

    def test_unit_system_has_si(self):
        """UnitSystem 应有 SI 成员。"""
        from e2m2e.mbse.data.enums import UnitSystem

        assert UnitSystem.SI.value == "si"

    def test_unit_system_member_count(self):
        """UnitSystem 应恰好有 2 个成员。"""
        from e2m2e.mbse.data.enums import UnitSystem

        assert len(UnitSystem) == 2


class TestReferenceFrame:
    """ReferenceFrame 枚举验证。"""

    def test_reference_frame_has_j2000(self):
        """ReferenceFrame 应有 J2000 成员。"""
        assert ReferenceFrame.J2000.value == "j2000"

    def test_reference_frame_has_synodic(self):
        """ReferenceFrame 应有 SYNODIC 成员。"""
        assert ReferenceFrame.SYNODIC.value == "synodic"

    def test_reference_frame_has_rotating(self):
        """ReferenceFrame 应有 ROTATING 成员。"""
        assert ReferenceFrame.ROTATING.value == "rotating"
