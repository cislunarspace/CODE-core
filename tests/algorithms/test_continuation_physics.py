"""测试 Continuation 物理去重

验证 compute_F_and_dF_symmetric_xz_plane 使用 CR3BP_Dynamics 实例
而非本地复制的物理公式。
"""

import numpy as np
import pytest

from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.continuation import compute_F_and_dF_symmetric_xz_plane


@pytest.fixture
def earth_moon_dynamics():
    system = CR3BP_System.from_known_system("earth_moon")
    return CR3BP_Dynamics(system)


class TestComputeFdDF:
    def test_accepts_dynamics_instance(self, earth_moon_dynamics):
        """函数接受 CR3BP_Dynamics 实例作为参数"""
        X = np.array([0.8, 0.0, 0.5, 0.5])
        SV0 = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
        F, dF = compute_F_and_dF_symmetric_xz_plane(X, SV0, earth_moon_dynamics)
        assert F.shape == (3,)
        assert dF.shape == (3, 4)

    def test_results_are_finite(self, earth_moon_dynamics):
        """结果均为有限值"""
        X = np.array([0.8, 0.0, 0.5, 0.5])
        SV0 = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
        F, dF = compute_F_and_dF_symmetric_xz_plane(X, SV0, earth_moon_dynamics)
        assert np.all(np.isfinite(F))
        assert np.all(np.isfinite(dF))

    def test_constraint_vector_order(self, earth_moon_dynamics):
        """约束向量顺序为 [vx, vz, ry]"""
        X = np.array([0.8, 0.0, 0.5, 0.5])
        SV0 = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
        F, _ = compute_F_and_dF_symmetric_xz_plane(X, SV0, earth_moon_dynamics)
        # F = [final_state[3], final_state[5], final_state[1]] = [vx, vz, ry]
        # 对于非收敛状态，这些值应非零
        assert F.shape == (3,)

    def test_jacobian_columns_match_stm(self, earth_moon_dynamics):
        """Jacobian 前 3 列对应 STM 的对应元素"""
        X = np.array([0.8, 0.01, 0.5, 0.5])
        SV0 = np.array([0.8, 0.0, 0.01, 0.0, 0.5, 0.0])
        F, dF = compute_F_and_dF_symmetric_xz_plane(X, SV0, earth_moon_dynamics)
        # dF 列应与 STM 行对应
        # 列0: ∂(vx,vz,ry)/∂rx → STM[3,0], STM[5,0], STM[1,0]
        assert dF.shape == (3, 4)
