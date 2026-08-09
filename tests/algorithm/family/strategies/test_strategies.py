"""微分修正策略模式测试。

验证 CorrectionConfig 数据类与各类策略函数的正确性。
"""

import pytest

from e2m2e.algorithm.family.strategies import (
    CorrectionConfig,
    halo_fixed_x0,
    halo_fixed_z0,
    symmetric_2d_fixed_t,
    symmetric_2d_fixed_x0,
    symmetric_2d_fixed_y0,
    symmetric_3d_fixed_x0,
    symmetric_xz_fixed_x0,
    symmetric_xz_fixed_z0,
)

pytestmark = pytest.mark.orchestration


class TestCorrectionConfig:
    """CorrectionConfig 不可变数据类"""

    def test_frozen(self):
        config = CorrectionConfig(setup_type="test", symmetry_condition="x_axis")
        with pytest.raises(AttributeError):
            config.setup_type = "other"

    def test_defaults(self):
        config = CorrectionConfig(setup_type="test", symmetry_condition="x_axis")
        assert config.fixed_parameters == {}
        assert config.free_variables == []
        assert config.free_variable_indices == []
        assert config.target_conditions == {}
        assert config.constraint_indices == []

    def test_full_config(self):
        config = CorrectionConfig(
            setup_type="2D_symmetric_x_fixed_x0",
            symmetry_condition="x_axis",
            fixed_parameters={"x0": 0.5},
            free_variables=["y_dot0", "T_half"],
            free_variable_indices=[4, 6],
            target_conditions={"y": 0.0, "x_dot": 0.0},
            constraint_indices=[1, 3],
        )
        assert config.setup_type == "2D_symmetric_x_fixed_x0"
        assert len(config.free_variable_indices) == 2


class TestSymmetric2DStrategies:
    def test_fixed_x0(self):
        config = symmetric_2d_fixed_x0(x0=0.5)
        assert config.setup_type == "2D_symmetric_x_fixed_x0"
        assert config.symmetry_condition == "x_axis"
        assert config.fixed_parameters["x0"] == 0.5
        assert config.free_variables == ["y_dot0", "T_half"]
        assert config.free_variable_indices == [4, 6]
        assert config.constraint_indices == [1, 3]

    def test_fixed_x0_default(self):
        config = symmetric_2d_fixed_x0()
        assert config.fixed_parameters["x0"] == 0.0

    def test_fixed_t(self):
        config = symmetric_2d_fixed_t(t_half=1.5)
        assert config.setup_type == "2D_symmetric_x_fixed_t"
        assert config.fixed_parameters["T_half"] == 1.5
        assert config.free_variable_indices == [0, 4]

    def test_fixed_y0(self):
        config = symmetric_2d_fixed_y0(y0=0.3)
        assert config.setup_type == "2D_symmetric_y_fixed_y0"
        assert config.symmetry_condition == "y_axis"
        assert config.free_variable_indices == [3, 6]
        assert config.constraint_indices == [0, 3]


class TestSymmetric3DStrategies:
    def test_fixed_x0(self):
        config = symmetric_3d_fixed_x0(x0=0.8)
        assert config.setup_type == "3D_symmetric_x_fixed_x0"
        assert config.free_variables == ["z0", "y_dot0", "T_half"]
        assert config.free_variable_indices == [2, 4, 6]
        assert len(config.constraint_indices) == 3

    def test_xz_fixed_x0(self):
        config = symmetric_xz_fixed_x0(x0=0.8)
        assert config.setup_type == "3D_symmetric_xz_fixed_x0"
        assert config.symmetry_condition == "xz_plane"

    def test_xz_fixed_z0(self):
        config = symmetric_xz_fixed_z0(z0=0.1)
        assert config.setup_type == "3D_symmetric_xz_fixed_z0"
        assert config.free_variable_indices == [0, 4, 6]
        assert config.fixed_parameters["z0"] == 0.1


class TestHaloStrategies:
    def test_fixed_z0(self):
        config = halo_fixed_z0(z0=0.1, libration_point=1)
        assert config.setup_type == "halo_orbit_fixed_z0"
        assert config.symmetry_condition == "xz_plane"
        assert config.fixed_parameters["z0"] == 0.1
        assert config.fixed_parameters["libration_point"] == 1
        assert config.free_variables == ["x0", "y_dot0", "T_half"]

    def test_fixed_x0(self):
        config = halo_fixed_x0(x0=0.8, libration_point=2)
        assert config.setup_type == "halo_orbit_fixed_x0"
        assert config.fixed_parameters["x0"] == 0.8
        assert config.free_variables == ["z0", "y_dot0", "T_half"]

    def test_default_libration_point(self):
        config = halo_fixed_z0(z0=0.1)
        assert config.fixed_parameters["libration_point"] == 1


class TestStrategyIntegration:
    """策略与 DifferentialCorrection 集成"""

    def test_setup_delegates_to_strategy(self):
        from e2m2e.algorithm.dynamics import CR3BP_System
        from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
        from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        dynamics = CR3BP_Dynamics(system)
        dc = DifferentialCorrection(dynamics)

        dc.setup_2D_symmetric_x_fixed_x0(x0=0.5)
        assert dc.setup_type == "2D_symmetric_x_fixed_x0"
        assert dc.free_variable_indices == [4, 6]
        assert dc.constraint_indices == [1, 3]
        assert dc.fixed_parameters.get("x0") == 0.5

    def test_setup_halo_delegates(self):
        from e2m2e.algorithm.dynamics import CR3BP_System
        from e2m2e.algorithm.dynamics.dynamics import CR3BP_Dynamics
        from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

        system = CR3BP_System(
            mu=0.0121506683, primary="Earth", secondary="Moon"
        )._with_default_scales()
        dynamics = CR3BP_Dynamics(system)
        dc = DifferentialCorrection(dynamics)

        dc.setup_halo_orbit_fixed_z0(z0=0.1, libration_point=1)
        assert dc.setup_type == "halo_orbit_fixed_z0"
        assert dc.free_variable_indices == [0, 4, 6]
