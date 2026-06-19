"""System ABC 与 CR3BP_System/EphemerisSystem 接口实现测试。

验证抽象基类契约、子类关系与接口一致性。
"""

import pytest

from e2m2e.core.cr3bp_system import CR3BP_System, LibrationPoint
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.system import System
from e2m2e.mbse.data.enums import ReferenceFrame, UnitSystem


class FakeSpiceManager:
    """System 接口测试使用的最小 SPICE 替身。"""

    def get_gm(self, body: str) -> float:
        return {"EARTH": 398600.435436, "MOON": 4902.800066}[body]

    def get_body_position(self, body: str, et: float, frame: str, origin: str):
        return [0.0, 0.0, 0.0]

    def get_body_state(self, body: str, et: float, frame: str, origin: str):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def earth_moon():
    return CR3BP_System(mu=1.21506683e-2, primary="Earth", secondary="Moon")


@pytest.fixture
def ephemeris_system():
    return EphemerisSystem(
        bodies=["EARTH", "MOON"],
        spice=FakeSpiceManager(),
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )


class TestSystemABC:
    """System 抽象基类契约。"""

    def test_system_is_abstract(self):
        """System 不能直接实例化。"""
        with pytest.raises(TypeError):
            System()  # type: ignore[abstract]

    def test_cr3bp_system_is_subclass(self):
        """CR3BP_System 应是 System 的子类。"""
        assert issubclass(CR3BP_System, System)

    def test_ephemeris_system_is_subclass(self):
        """EphemerisSystem 应是 System 的子类。"""
        assert issubclass(EphemerisSystem, System)

    def test_cr3bp_instance_is_system(self, earth_moon):
        """CR3BP_System 实例应是 System。"""
        assert isinstance(earth_moon, System)

    def test_ephemeris_instance_is_system(self, ephemeris_system):
        """EphemerisSystem 实例应是 System。"""
        assert isinstance(ephemeris_system, System)


class TestCR3BPSystemInterface:
    """CR3BP_System 对 System 接口的实现。"""

    def test_frame_is_synodic(self, earth_moon):
        """CR3BP 坐标框架应为 SYNODIC。"""
        assert earth_moon.frame == ReferenceFrame.SYNODIC

    def test_unit_system_is_dimensionless(self, earth_moon):
        """CR3BP 单位系统应为 DIMENSIONLESS。"""
        assert earth_moon.unit_system == UnitSystem.DIMENSIONLESS

    def test_gravitational_parameter_primary(self, earth_moon):
        """primary 引力参数应为 1 - mu。"""
        assert earth_moon.gravitational_parameter("primary") == pytest.approx(1 - earth_moon.mu)

    def test_gravitational_parameter_secondary(self, earth_moon):
        """secondary 引力参数应为 mu。"""
        assert earth_moon.gravitational_parameter("secondary") == pytest.approx(earth_moon.mu)

    def test_gravitational_parameter_invalid_body(self, earth_moon):
        """无效天体名应抛出 ValueError。"""
        with pytest.raises(ValueError):
            earth_moon.gravitational_parameter("jupiter")


class TestCR3BPSystemPreserved:
    """迁移后 CR3BP_System 原有功能保持不变。"""

    def test_libration_point_enum(self):
        """LibrationPoint 枚举可从 cr3bp_system 导入。"""
        assert LibrationPoint.L1.value == 1

    def test_compute_libration_points(self, earth_moon):
        """平动点计算正常工作。"""
        points = earth_moon.compute_libration_points()
        assert LibrationPoint.L1 in points

    def test_jacobi_constant(self, earth_moon):
        """Jacobi 常数计算正常工作。"""
        state = [0.8, 0.0, 0.0, 0.0, 0.3, 0.0]
        c = earth_moon.get_jacobi_constant(state)
        assert isinstance(c, float)


class TestEphemerisSystemInterface:
    """EphemerisSystem 对 System 接口的实现。"""

    def test_frame_is_j2000(self, ephemeris_system):
        """默认 frame 应为 J2000。"""
        assert ephemeris_system.frame == ReferenceFrame.J2000

    def test_unit_system_is_si(self, ephemeris_system):
        """星历单位系统应为 SI。"""
        assert ephemeris_system.unit_system == UnitSystem.SI

    def test_gravitational_parameter_earth(self, ephemeris_system):
        """EARTH 引力参数应来自 SPICE。"""
        assert ephemeris_system.gravitational_parameter("EARTH") == pytest.approx(398600.435436)

    def test_gravitational_parameter_moon(self, ephemeris_system):
        """MOON 引力参数应来自 SPICE。"""
        assert ephemeris_system.gravitational_parameter("MOON") == pytest.approx(4902.800066)


class TestSystemInterfaceContract:
    """System 子类通用契约。"""

    def test_cr3bp_and_ephemeris_share_interface(self, earth_moon, ephemeris_system):
        """两个子类都应实现 frame / unit_system / gravitational_parameter。"""
        for system in (earth_moon, ephemeris_system):
            assert isinstance(system.frame, ReferenceFrame)
            assert isinstance(system.unit_system, UnitSystem)
        # CR3BP 接受 "primary"/"secondary"
        assert isinstance(earth_moon.gravitational_parameter("primary"), float)
        # 星历接受 SPICE 天体名
        assert isinstance(ephemeris_system.gravitational_parameter("EARTH"), float)

    def test_ephemeris_frame_value_used_by_spice(self, ephemeris_system):
        """SPICE 字符串应来自 frame.value。"""
        assert ephemeris_system.frame.value == "J2000"
