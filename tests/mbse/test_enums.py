"""Test shared enum definitions in mbse/data/enums.py."""

import pytest

from e2m2e.mbse.data.enums import (
    BifurcationLabel,
    BoundaryMode,
    ConvergenceState,
    OrbitFamilyType,
    ProjectionPlane,
    ReferenceFrame,
    StabilityLabel,
    TransferPhase,
    TransferType,
    TwoLevelMultipleShootingStatus,
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


class TestBoundaryMode:
    """BoundaryMode 枚举验证。"""

    def test_boundary_mode_exists(self):
        """BoundaryMode 枚举应该存在。"""
        from e2m2e.mbse.data.enums import BoundaryMode

        assert BoundaryMode is not None

    def test_boundary_mode_has_fixed_endpoints(self):
        """BoundaryMode 应有 FIXED_ENDPOINTS 成员。"""
        from e2m2e.mbse.data.enums import BoundaryMode

        assert BoundaryMode.FIXED_ENDPOINTS.value == "fixed_endpoints"

    def test_boundary_mode_member_count(self):
        """BoundaryMode 应恰好有 1 个成员（当前只实现 fixed_endpoints）。"""
        from e2m2e.mbse.data.enums import BoundaryMode

        assert len(BoundaryMode) == 1


class TestTwoLevelMultipleShootingStatus:
    """TwoLevelMultipleShootingStatus 枚举验证。"""

    def test_status_exists(self):
        """TwoLevelMultipleShootingStatus 枚举应该存在。"""
        from e2m2e.mbse.data.enums import TwoLevelMultipleShootingStatus

        assert TwoLevelMultipleShootingStatus is not None

    def test_status_has_all_members(self):
        """应有全部三个状态值。"""
        from e2m2e.mbse.data.enums import TwoLevelMultipleShootingStatus

        assert TwoLevelMultipleShootingStatus.CONVERGED.value == "converged"
        assert TwoLevelMultipleShootingStatus.MAX_ITERATIONS.value == "max_iterations"
        assert TwoLevelMultipleShootingStatus.LEVEL1_FAILED.value == "level1_failed"

    def test_status_member_count(self):
        """应恰好有 3 个成员。"""
        from e2m2e.mbse.data.enums import TwoLevelMultipleShootingStatus

        assert len(TwoLevelMultipleShootingStatus) == 3


class TestReferenceFrame:
    """ReferenceFrame 枚举验证。"""

    def test_reference_frame_has_j2000(self):
        """ReferenceFrame 应有 J2000 成员。"""
        assert ReferenceFrame.J2000.value == "J2000"

    def test_reference_frame_has_synodic(self):
        """ReferenceFrame 应有 SYNODIC 成员。"""
        assert ReferenceFrame.SYNODIC.value == "synodic"

    def test_reference_frame_has_rotating(self):
        """ReferenceFrame 应有 ROTATING 成员。"""
        assert ReferenceFrame.ROTATING.value == "rotating"
