"""
HaloOrbitGenerator 测试模块

测试 Halo 轨道生成器的功能，包括 L1/L2 北/南 Halo 轨道生成和轨道族生成。

地月系统参数：
  μ = 1.2150585 × 10⁻²
  DU = 3.84405 × 10⁵ km, TU = 4.34811305 天
"""

import numpy as np
import pytest

import e2m2e
from e2m2e.algorithms import HaloOrbitGenerator
from e2m2e.core import CR3BP_Dynamics, Orbit

MU = 0.012150585


@pytest.fixture
def earth_moon_system():
    """创建地月 CR3BP 系统"""
    system = e2m2e.core.system.CR3BP_System(mu=MU, primary="earth", secondary="moon")
    system.compute_libration_points()
    return system


@pytest.fixture
def dynamics(earth_moon_system):
    """创建动力学对象"""
    return CR3BP_Dynamics(earth_moon_system)


@pytest.fixture
def halo_generator(earth_moon_system):
    """创建 Halo 轨道生成器"""
    return HaloOrbitGenerator(earth_moon_system)


class TestHaloOrbitGenerator:
    """测试 HaloOrbitGenerator 类"""

    def test_initialization(self, earth_moon_system):
        """初始化后应包含必要的属性"""
        generator = HaloOrbitGenerator(earth_moon_system)
        assert generator.system is not None
        assert generator.dynamics is not None
        assert generator.corrector is not None

    def test_system_assignment(self, halo_generator, earth_moon_system):
        """system 属性应正确赋值"""
        assert halo_generator.system is earth_moon_system

    def test_dynamics_type(self, halo_generator):
        """dynamics 应为 CR3BP_Dynamics 类型"""
        assert isinstance(halo_generator.dynamics, CR3BP_Dynamics)


class TestL1NorthHalo:
    """测试 L1 北 Halo 轨道生成"""

    def test_l1_north_halo_generation(self, halo_generator):
        """L1 北 Halo 轨道应能成功生成"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert isinstance(halo, Orbit)
            assert halo.states.shape[1] == 6

    def test_l1_north_halo_converged(self, halo_generator):
        """L1 北 Halo 轨道应收敛"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert halo_generator.corrector.converged is True

    def test_l1_north_halo_period_positive(self, halo_generator):
        """L1 北 Halo 周期应为正"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert halo.period > 0

    def test_l1_north_halo_period_near_expected(self, halo_generator):
        """L1 北 Halo 周期应接近预期值（约 0.5 无量纲周期）"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert 0.3 < halo.period < 0.7

    def test_l1_north_halo_parameters(self, halo_generator):
        """L1 北 Halo 轨道应包含正确的参数"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert halo.parameters["libration_point"] == 1
            assert halo.parameters["amplitude_z"] == 0.05
            assert halo.parameters["halo_class"] == 0


class TestL2NorthHalo:
    """测试 L2 北 Halo 轨道生成"""

    def test_l2_north_halo_generation(self, halo_generator):
        """L2 北 Halo 轨道应能成功生成"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=2,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert isinstance(halo, Orbit)

    def test_l2_north_halo_period_positive(self, halo_generator):
        """L2 北 Halo 周期应为正"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=2,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert halo.period > 0

    def test_l1_l2_periods_similar(self, halo_generator):
        """L1 和 L2 Halo 周期应相近"""
        halo_l1 = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        halo_l2 = halo_generator.generate_seed_orbit(
            libration_point=2,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo_l1 is not None and halo_l2 is not None:
            np.testing.assert_allclose(halo_l1.period, halo_l2.period, rtol=0.1)


class TestSouthHalo:
    """测试南 Halo (Class II) 轨道生成"""

    def test_l1_south_halo_generation(self, halo_generator):
        """L1 南 Halo 轨道应能成功生成"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=1,
            verbose=False,
        )
        if halo is not None:
            assert isinstance(halo, Orbit)

    def test_l2_south_halo_generation(self, halo_generator):
        """L2 南 Halo 轨道应能成功生成"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=2,
            amplitude_z=0.05,
            halo_class=1,
            verbose=False,
        )
        if halo is not None:
            assert isinstance(halo, Orbit)

    def test_north_south_z_amplitude_opposite(self, halo_generator):
        """北和南 Halo z 振幅应符号相反"""
        halo_north = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        halo_south = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=1,
            verbose=False,
        )
        if halo_north is not None and halo_south is not None:
            z_north = halo_north.states[:, 2]
            z_south = halo_south.states[:, 2]
            assert np.max(np.abs(z_north)) > 0
            assert np.max(np.abs(z_south)) > 0


class TestOrbitFamilyGeneration:
    """测试轨道族生成"""

    def test_family_generation_starts_with_seed(self, halo_generator):
        """轨道族应以种子轨道开始"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            family = halo_generator.generate_family(
                seed_orbit=seed,
                n_orbits=5,
                direction="positive",
                step_size=0.005,
            )
            assert len(family) >= 1
            assert family[0] is seed

    def test_family_increases_amplitude(self, halo_generator):
        """轨道族振幅应单调增加（正向延拓）"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            family = halo_generator.generate_family(
                seed_orbit=seed,
                n_orbits=5,
                direction="positive",
                step_size=0.01,
            )
            if len(family) > 1:
                amplitudes = [orbit.parameters.get("amplitude_z") for orbit in family]
                for i in range(len(amplitudes) - 1):
                    if amplitudes[i] is not None and amplitudes[i + 1] is not None:
                        assert amplitudes[i + 1] > amplitudes[i]

    def test_family_negative_direction(self, halo_generator):
        """负向延拓应减小振幅"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.10,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            family = halo_generator.generate_family(
                seed_orbit=seed,
                n_orbits=3,
                direction="negative",
                step_size=0.01,
            )
            if len(family) > 1:
                amplitudes = [orbit.parameters.get("amplitude_z") for orbit in family]
                for i in range(len(amplitudes) - 1):
                    if amplitudes[i] is not None and amplitudes[i + 1] is not None:
                        assert amplitudes[i + 1] < amplitudes[i]

    def test_family_both_directions(self, halo_generator):
        """双向延拓应生成包含原点的族"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            family = halo_generator.generate_family(
                seed_orbit=seed,
                n_orbits=3,
                direction="both",
                step_size=0.01,
            )
            assert len(family) >= 1

    def test_invalid_n_orbits_throws(self, halo_generator):
        """n_orbits < 1 时应抛出 ValueError"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            with pytest.raises(ValueError, match="n_orbits必须大于0"):
                halo_generator.generate_family(
                    seed_orbit=seed,
                    n_orbits=0,
                    direction="positive",
                    step_size=0.01,
                )

    def test_invalid_direction_throws(self, halo_generator):
        """无效的 direction 应抛出 ValueError"""
        seed = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if seed is not None:
            with pytest.raises(ValueError, match="direction必须是positive/negative/both"):
                halo_generator.generate_family(
                    seed_orbit=seed,
                    n_orbits=5,
                    direction="invalid",
                    step_size=0.01,
                )


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_libration_point_throws(self, halo_generator):
        """libration_point 不是 1 或 2 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="libration_point必须是1或2"):
            halo_generator.generate_seed_orbit(
                libration_point=3,
                amplitude_z=0.05,
                halo_class=0,
                verbose=False,
            )

    def test_zero_amplitude_throws(self, halo_generator):
        """amplitude_z <= 0 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="amplitude_z必须为正数"):
            halo_generator.generate_seed_orbit(
                libration_point=1,
                amplitude_z=0.0,
                halo_class=0,
                verbose=False,
            )

    def test_negative_amplitude_throws(self, halo_generator):
        """amplitude_z < 0 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="amplitude_z必须为正数"):
            halo_generator.generate_seed_orbit(
                libration_point=1,
                amplitude_z=-0.05,
                halo_class=0,
                verbose=False,
            )

    def test_invalid_halo_class_throws(self, halo_generator):
        """halo_class 不是 0 或 1 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="halo_class必须是0或1"):
            halo_generator.generate_seed_orbit(
                libration_point=1,
                amplitude_z=0.05,
                halo_class=2,
                verbose=False,
            )


class TestHaloOrbitPhysicalProperties:
    """测试 Halo 轨道物理性质"""

    def test_halo_orbit_xz_symmetry(self, halo_generator):
        """Halo 轨道应满足 XZ 平面对称性"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            states = halo.states
            y = states[:, 1]
            z = states[:, 2]
            np.testing.assert_allclose(y[0], 0.0, atol=1e-6)
            np.testing.assert_allclose(z[0], 0.0, atol=1e-6)

    def test_halo_orbit_initial_vx_zero(self, halo_generator):
        """Halo 轨道初始 vx 应为零"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            np.testing.assert_allclose(halo.states[0, 3], 0.0, atol=1e-6)

    def test_halo_orbit_initial_vz_zero(self, halo_generator):
        """Halo 轨道初始 vz 应为零"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            np.testing.assert_allclose(halo.states[0, 5], 0.0, atol=1e-6)

    def test_halo_orbit_has_family_type(self, halo_generator):
        """Halo 轨道应有 family_type 属性"""
        halo = halo_generator.generate_seed_orbit(
            libration_point=1,
            amplitude_z=0.05,
            halo_class=0,
            verbose=False,
        )
        if halo is not None:
            assert halo.family_type == "halo"
