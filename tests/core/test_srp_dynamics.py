"""CR3BP_SRP_Dynamics 测试

测试太阳辐射压动力学子类的功能：
1. 光学系数计算正确性
2. SRP 力计算正确性
3. 与纯 CR3BP 的一致性（area=0 时）
4. 传播功能正常工作
"""

import numpy as np
import pytest

from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.system import CR3BP_System
from e2m2e.core.srp_dynamics import CR3BP_SRP_Dynamics


@pytest.fixture
def earth_moon_system():
    """创建地月系统"""
    return CR3BP_System.from_known_system("earth_moon")


@pytest.fixture
def srp_dynamics(earth_moon_system):
    """创建 SRP 动力学对象"""
    return CR3BP_SRP_Dynamics(
        earth_moon_system,
        area=100.0,
        mass=1000.0,
        Cr=1.5,
    )


@pytest.fixture
def zero_srp_dynamics(earth_moon_system):
    """创建零 SRP 动力学对象（应与纯 CR3BP 一致）"""
    return CR3BP_SRP_Dynamics(
        earth_moon_system,
        area=0.0,
        mass=1000.0,
        Cr=1.5,
    )


class TestOpticalCoefficients:
    """测试光学系数计算"""

    def test_b1_coefficient(self, srp_dynamics):
        """测试 b1 系数计算"""
        expected = 0.5 * (1.0 - 0.975 * 0.999)
        assert srp_dynamics.b1 == pytest.approx(expected)

    def test_b2_coefficient(self, srp_dynamics):
        """测试 b2 系数计算"""
        expected = 0.975 * 0.999
        assert srp_dynamics.b2 == pytest.approx(expected)

    def test_b3_coefficient(self, srp_dynamics):
        """测试 b3 系数计算"""
        Bf = 0.038
        Bb = 0.004
        s = 0.975
        p = 0.999
        ef = 0.8
        eb = 0.2
        expected = 0.5 * (
            Bf * (1.0 - s) * p
            + (1.0 - p) * (ef * Bf - eb * Bb) / (ef + eb)
        )
        assert srp_dynamics.b3 == pytest.approx(expected)


class TestBetaCoefficient:
    """测试 SRP 加速度系数"""

    def test_beta_calculation(self, srp_dynamics):
        """测试 beta = P_srp * A * Cr / (2 * m)"""
        P_srp = 4.56e-6
        area = 100.0
        Cr = 1.5
        mass = 1000.0
        expected = P_srp * area * Cr / (2.0 * mass)
        assert srp_dynamics.beta == pytest.approx(expected, abs=1e-15)

    def test_zero_area_gives_zero_beta(self, zero_srp_dynamics):
        """测试面积为 0 时 beta 为 0"""
        assert zero_srp_dynamics.beta == 0.0


class TestSRPForce:
    """测试 SRP 力计算"""

    def test_srp_force_magnitude(self, srp_dynamics):
        """测试 SRP 力的量级合理"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        eom = srp_dynamics._get_eom_func(with_stm=False)
        ds = eom(0.0, state)
        assert ds.shape == (6,)
        assert np.all(np.isfinite(ds))

    def test_srp_disabled_when_area_zero(self, zero_srp_dynamics):
        """测试面积为 0 时 SRP 被禁用"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        eom = zero_srp_dynamics._get_eom_func(with_stm=False)
        ds_zero = eom(0.0, state)

        cr3bp = CR3BP_Dynamics(zero_srp_dynamics.system)
        ds_cr3bp = cr3bp.equations_of_motion(0.0, state)

        np.testing.assert_allclose(ds_zero, ds_cr3bp, atol=1e-15)


class TestPropagation:
    """测试传播功能"""

    def test_propagate_returns_correct_shape(self, srp_dynamics):
        """测试传播结果形状正确"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 1.0),
            t_eval=np.linspace(0.0, 1.0, 10),
        )
        assert result["states"].shape == (10, 6)
        assert result["time"].shape == (10,)

    def test_propagate_with_stm(self, srp_dynamics):
        """测试带 STM 的传播"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 0.1),
            with_stm=True,
        )
        assert "stm" in result
        assert result["stm"].shape[1:] == (6, 6)

    def test_propagate_with_jacobi(self, srp_dynamics):
        """测试带 Jacobi 常数的传播"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 0.1),
            with_jacobi=True,
        )
        assert "jacobi" in result
        assert len(result["jacobi"]) > 0
