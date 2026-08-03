"""momentum_management.py 单元测试（纯数学，无 SPICE/传播器依赖）。

布局设计原则：每个发动机的 rᵢ 与 eᵢ 不平行，使 E（力臂矩阵）与 E_r（力矩
方向矩阵）的增广矩阵满秩（秩 6）。6 发动机采用循环配对：±x 装 y 方向发动机、
±y 装 z 方向发动机、±z 装 x 方向发动机，使 E 和 E_r 各自满秩。
"""

import numpy as np
import pytest

from e2m2e.algorithm.station_keeping.momentum_management import (
    EngineLayout,
    compute_delta_m,
    compute_srp_torque,
    solve_joint_control,
    solve_momentum_unload,
    validate_engine_layout,
)


def _full_rank_layout() -> EngineLayout:
    """6 发动机满秩布局（rᵢ 与 eᵢ 不平行，E 与 E_r 各自满秩）。

    安装方案（循环配对，保证 E 与 E_r 各自满秩）：
    - ±x 位置装 y 方向发动机
    - ±y 位置装 z 方向发动机
    - ±z 位置装 x 方向发动机
    """
    positions = np.array([
        [ 1,  0,  0],   # +x 位置
        [-1,  0,  0],   # -x 位置
        [ 0,  1,  0],   # +y 位置
        [ 0, -1,  0],   # -y 位置
        [ 0,  0,  1],   # +z 位置
        [ 0,  0, -1],   # -z 位置
    ], dtype=float)
    directions = np.array([
        [ 0,  1,  0],   # +x 位置 → y 方向
        [ 0, -1,  0],   # -x 位置 → -y 方向
        [ 0,  0,  1],   # +y 位置 → z 方向
        [ 0,  0, -1],   # -y 位置 → -z 方向
        [ 1,  0,  0],   # +z 位置 → x 方向
        [-1,  0,  0],   # -z 位置 → -x 方向
    ], dtype=float)
    return EngineLayout(positions_m=positions, directions=directions)


# ── EngineLayout ──

class TestEngineLayout:
    def test_full_rank_construction(self):
        layout = _full_rank_layout()
        assert layout.num_engines == 6
        assert layout.E.shape == (3, 6)
        assert layout.E_r.shape == (3, 6)
        # E 与 E_r 各自满秩
        assert np.linalg.matrix_rank(layout.E) == 3
        assert np.linalg.matrix_rank(layout.E_r) == 3

    def test_directions_normalized(self):
        layout = _full_rank_layout()
        norms = np.linalg.norm(layout.directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_too_few_engines_rejected(self):
        with pytest.raises(ValueError, match="不足 6"):
            EngineLayout(
                positions_m=np.ones((5, 3)),
                directions=np.array([[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,0,0]]),
            )

    def test_zero_direction_rejected(self):
        """6 发动机但含零方向 → 被拒。"""
        with pytest.raises(ValueError, match="零矢量"):
            EngineLayout(
                positions_m=np.tile(np.eye(3), (2, 1)),
                directions=np.array([
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [0, 0, 0], [1, 0, 0], [0, 1, 0],
                ]),
            )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="不匹配"):
            EngineLayout(
                positions_m=np.ones((6, 3)),
                directions=np.ones((5, 3)),
            )


# ── validate_engine_layout ──

class TestValidateEngineLayout:
    def test_full_rank_passes(self):
        validate_engine_layout(_full_rank_layout())

    def test_rank_deficient_Er_rejected(self):
        """6 个发动机全沿 z 方向喷气 → E_r 秩 1。"""
        positions = np.array([
            [1,0,0], [-1,0,0], [0,1,0],
            [0,-1,0], [1,1,0], [-1,-1,0],
        ], dtype=float)
        directions = np.tile([0, 0, 1.0], (6, 1))
        layout = EngineLayout(positions_m=positions, directions=directions)
        with pytest.raises(ValueError, match="E_r 秩不足"):
            validate_engine_layout(layout)

    def test_rank_deficient_augmented_rejected(self):
        """布局使 E 与 E_r 的增广矩阵不满秩。"""
        # 6 个发动机全沿 z 方向，安装在 xy 平面
        # E[:,i] = rᵢ × [0,0,1] = [r_y, -r_x, 0] → E 行秩最多 2
        # E_r = [[0]*6, [0]*6, [1]*6] → E_r 秩 1
        # 增广秩 < 6
        positions = np.array([
            [1,0,0], [-1,0,0], [0,1,0],
            [0,-1,0], [1,1,0], [-1,-1,0],
        ], dtype=float)
        directions = np.tile([0, 0, 1.0], (6, 1))
        layout = EngineLayout(positions_m=positions, directions=directions)
        with pytest.raises(ValueError, match="秩不足"):
            validate_engine_layout(layout)


# ── compute_srp_torque ──

class TestSrpTorque:
    def test_perpendicular(self):
        # r=[0,2,0], F=[1,0,0] → r×F = [0*0-2*0, 2*1-0*0, 0*1-0*0] = [0,0,-2]（右手系）
        torque = compute_srp_torque([1, 0, 0], [0, 2, 0])
        np.testing.assert_allclose(torque, [0, 0, -2])

    def test_parallel(self):
        torque = compute_srp_torque([1, 0, 0], [3, 0, 0])
        np.testing.assert_allclose(torque, [0, 0, 0], atol=1e-15)


# ── compute_delta_m ──

class TestDeltaM:
    def test_basic(self):
        dm = compute_delta_m([100, 0, 0], dt_sec=3600)
        np.testing.assert_allclose(dm, [360000, 0, 0])


# ── solve_momentum_unload ──

class TestSolveMomentumUnload:
    def test_single_axis_unload(self):
        """z 轴角动量需求 → 只有 z 方向发动机（索引 2,3）工作。"""
        layout = _full_rank_layout()
        delta_m = np.array([0, 0, 100.0])
        V = solve_momentum_unload(layout.E_r, delta_m, mass_kg=1000.0)
        # E_r[:,0]=[0,1,0], E_r[:,1]=[0,-1,0] → 对 z 无贡献
        # E_r[:,4]=[1,0,0], E_r[:,5]=[-1,0,0] → 对 z 无贡献
        np.testing.assert_allclose(V[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(V[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(V[4], 0.0, atol=1e-12)
        np.testing.assert_allclose(V[5], 0.0, atol=1e-12)
        # E_r[:,2]=[0,0,1], E_r[:,3]=[0,0,-1]：1000·(V[2]-V[3])=100 → V[2]-V[3]=0.1
        assert V[2] - V[3] == pytest.approx(0.1, abs=1e-12)

    def test_constraint_satisfied(self):
        layout = _full_rank_layout()
        delta_m = np.array([50, -30, 20.0])
        V = solve_momentum_unload(layout.E_r, delta_m, mass_kg=500.0)
        residual = 500.0 * layout.E_r @ V - delta_m
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_min_norm_property(self):
        """解是最小范数的（V 非零元素尽可能小）。"""
        layout = _full_rank_layout()
        delta_m = np.array([0, 0, 100.0])
        V = solve_momentum_unload(layout.E_r, delta_m, mass_kg=1000.0)
        # 期望 V[2]=V[3]=0.05（各分担一半），‖V‖²=0.005
        expected_norm_sq = 2 * 0.05 ** 2
        assert np.dot(V, V) == pytest.approx(expected_norm_sq, abs=1e-12)


# ── solve_joint_control ──

class TestSolveJointControl:
    def test_orbital_only(self):
        """纯轨道控制，无角动量需求 → E·V = Δv_o。"""
        layout = _full_rank_layout()
        dv = np.array([1.0, 0, 0])
        V = solve_joint_control(layout.E, layout.E_r, dv, delta_m=[0, 0, 0], mass_kg=1000.0)
        np.testing.assert_allclose(layout.E @ V, dv, atol=1e-10)

    def test_joint_constraints_satisfied(self):
        """联合控制：同时满足轨道和角动量约束。"""
        layout = _full_rank_layout()
        dv = np.array([0.5, -0.3, 0.1])
        dm = np.array([10, -5, 20.0])
        V = solve_joint_control(layout.E, layout.E_r, dv, delta_m=dm, mass_kg=800.0)
        np.testing.assert_allclose(layout.E @ V, dv, atol=1e-10)
        np.testing.assert_allclose(800.0 * layout.E_r @ V, dm, atol=1e-10)

    def test_zero_demands(self):
        layout = _full_rank_layout()
        V = solve_joint_control(layout.E, layout.E_r, [0, 0, 0], [0, 0, 0], 1000.0)
        np.testing.assert_allclose(V, 0.0, atol=1e-14)
