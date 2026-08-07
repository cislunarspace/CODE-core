"""ForceModel Python STM 路径 ∂a/∂v 同步测试（issue #317 第 2.1 项，ADR 0018）。

验证 ``ForceModel`` 的 Python 回退 STM 路径不再把 ``∂a/∂v`` 静默置零：

- ``_compute_total_jacobian`` 返回 ``(∂a/∂r, ∂a/∂v)`` 二元组；速度依赖力
  （阻尼型）的 ``∂a/∂v`` 由有限差分给出真值。
- 整条 STM 传播把 ``∂a/∂v`` 纳入 ``A[3:,3:]``：纯阻尼力的 STM
  ``Φ[3:6, 3:6] = e^{-k·Δt}·I``（旧代码 ``A[3:,3:]=0`` 会给出 ``I``，错误）。

这些测试不需要 SPICE——测试力忽略 system，``to_rust_spec`` 返回 None 确保
走 Python 回退路径（直接覆盖 ``_eom_func_with_stm``）。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.physical_model import PhysicalModel


class _FakeSystem:
    """最小 system 桩：提供 coordinate_system 占位即可（测试力不查 system）。"""

    def __init__(self) -> None:
        self.coordinate_system = object()


class _DampingForce(PhysicalModel):
    """速度依赖力 ``a = -k·v``（无解析雅可比，走有限差分）。

    解析 ``∂a/∂r = 0``、``∂a/∂v = -k·I``。不覆写 ``compute_jacobian`` →
    继承基类返回 None，由 ``ForceModel`` 有限差分兜底，同时对位置与速度扰动。
    """

    def __init__(self, k: float = 0.5) -> None:
        self._k = float(k)

    def compute_acceleration(self, t, state, system):  # noqa: ANN001
        v = np.asarray(state, dtype=float)[3:6]
        return -self._k * v


class _PositionOnlyNoJacForce(PhysicalModel):
    """位置型力 ``a = -k·r``（无解析雅可比，走有限差分）。

    解析 ``∂a/∂r = -k·I``、``∂a/∂v = 0``。验证位置型力的速度 FD 仍给零，
    即重构不改变既有位置型力的 STM 行为（回归）。
    """

    def __init__(self, k: float = 1.0) -> None:
        self._k = float(k)

    def compute_acceleration(self, t, state, system):  # noqa: ANN001
        r = np.asarray(state, dtype=float)[:3]
        return -self._k * r


class _PositionOnlyAnalyticForce(PhysicalModel):
    """位置型力 ``a = -k·r``，带解析 ``∂a/∂r``。验证解析力 ``∂a/∂v = 0``。"""

    def __init__(self, k: float = 1.0) -> None:
        self._k = float(k)

    def compute_acceleration(self, t, state, system):  # noqa: ANN001
        r = np.asarray(state, dtype=float)[:3]
        return -self._k * r

    def compute_jacobian(self, t, state, system):  # noqa: ANN001
        return -self._k * np.eye(3)


# =============================================================================
# _compute_total_jacobian 二元组正确性
# =============================================================================
class TestComputeTotalJacobianDadv:
    """``_compute_total_jacobian`` 返回 (∂a/∂r, ∂a/∂v)，速度依赖力给真值。"""

    def test_velocity_dependent_force_yields_nonzero_dadv(self):
        """阻尼力 a=-k·v → ∂a/∂v=-k·I、∂a/∂r=0（有限差分路径）。"""
        k = 0.5
        fm = ForceModel(_FakeSystem(), forces=[_DampingForce(k=k)])
        state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        dadr, dadv = fm._compute_total_jacobian(0.0, state)

        assert_allclose(dadr, np.zeros((3, 3)), atol=1e-10, err_msg="阻尼力 ∂a/∂r 应为零")
        assert_allclose(
            dadv,
            -k * np.eye(3),
            atol=1e-6,
            err_msg="阻尼力 ∂a/∂v 应为 -k·I（有限差分）",
        )

    def test_position_only_fd_force_yields_zero_dadv(self):
        """位置型力（无解析雅可比）→ ∂a/∂v=0、∂a/∂r=-k·I（回归）。"""
        k = 1.0
        fm = ForceModel(_FakeSystem(), forces=[_PositionOnlyNoJacForce(k=k)])
        state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        dadr, dadv = fm._compute_total_jacobian(0.0, state)

        assert_allclose(dadr, -k * np.eye(3), atol=1e-6, err_msg="位置型力 ∂a/∂r 应为 -k·I")
        assert_allclose(
            dadv,
            np.zeros((3, 3)),
            atol=1e-10,
            err_msg="位置型力 ∂a/∂v 应严格为零（扰动速度不改变加速度）",
        )

    def test_position_only_analytic_force_yields_zero_dadv(self):
        """位置型力（解析雅可比）→ ∂a/∂v=0（解析力按契约速度块置零）。"""
        k = 1.0
        fm = ForceModel(_FakeSystem(), forces=[_PositionOnlyAnalyticForce(k=k)])
        state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        dadr, dadv = fm._compute_total_jacobian(0.0, state)

        assert_allclose(dadr, -k * np.eye(3), atol=1e-12)
        assert_allclose(dadv, np.zeros((3, 3)), atol=1e-12)

    def test_mixed_forces_dadv_superposes(self):
        """组合力 ∂a/∂v 逐力叠加（阻尼 + 位置型 = -k·I + 0）。"""
        k = 0.4
        fm = ForceModel(
            _FakeSystem(),
            forces=[_DampingForce(k=k), _PositionOnlyAnalyticForce(k=1.0)],
        )
        state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        dadr, dadv = fm._compute_total_jacobian(0.0, state)

        # ∂a/∂r：仅位置型贡献 -1·I
        assert_allclose(dadr, -1.0 * np.eye(3), atol=1e-6)
        # ∂a/∂v：仅阻尼贡献 -k·I
        assert_allclose(dadv, -k * np.eye(3), atol=1e-6)


# =============================================================================
# 整条 STM 传播：∂a/∂v 进入 A[3:,3:]
# =============================================================================
class TestStmPropagationUsesDadv:
    """速度依赖力的 STM 反映 ``A[3:,3:] = ∂a/∂v``，不再静默置零。

    纯阻尼力 ``a = -k·v`` 的解析 STM：``Φ[3:6, 3:6] = e^{-k·Δt}·I``、
    ``Φ[0:3, 3:6] = (1-e^{-k·Δt})/k · I``。旧代码 ``A[3:,3:]=0`` 会给出
    ``Φ[3:6,3:6]=I``（无阻尼），是静默错误——本测试直接区分两者。
    """

    @pytest.mark.parametrize("k", [0.25, 0.5, 1.0])
    def test_damping_stm_velocity_block_decays(self, k):
        """Φ 的速度-速度块按 e^{-k·Δt} 衰减，证明 ∂a/∂v 已纳入 A。"""
        dt = 1.0
        fm = ForceModel(_FakeSystem(), forces=[_DampingForce(k=k)])
        fm.rtol = 1e-12
        fm.atol = 1e-12

        state0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        result = fm.propagate(state0, (0.0, dt), with_stm=True, initial_step=0.05)
        stm = result["stm"][-1]

        expected_decay = np.exp(-k * dt)
        # 速度-速度块对角：Φ[3,3]=Φ[4,4]=Φ[5,5]=e^{-k·dt}；非对角为零（各轴解耦）
        assert_allclose(
            np.diag(stm)[3:6],
            expected_decay,
            atol=1e-7,
            err_msg=(
                f"Φ 速度-速度块应 = e^(-k·dt)={expected_decay:.6f}；"
                "若 ≈1 则 ∂a/∂v 未纳入 A（旧 A[3:,3:]=0 bug）"
            ),
        )
        # 位置-速度块对角（Φ[0:3, 3:6] 的对角，即 stm[i, i+3]）：
        # (1-e^{-k·dt})/k —— δv0 映射到 δr 的积分核
        expected_cross = (1.0 - expected_decay) / k
        cross_diag = np.array([stm[i, i + 3] for i in range(3)])
        assert_allclose(
            cross_diag,
            expected_cross,
            atol=1e-6,
            err_msg="Φ 位置-速度块对角应 = (1-e^{-k·dt})/k",
        )

    def test_damping_stm_velocity_block_not_identity(self):
        """显式断言 Φ[3:,3:] ≠ I：旧 bug 下会是单位阵。"""
        k = 0.5
        fm = ForceModel(_FakeSystem(), forces=[_DampingForce(k=k)])
        fm.rtol = 1e-12
        fm.atol = 1e-12

        state0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        stm = fm.propagate(state0, (0.0, 1.0), with_stm=True, initial_step=0.05)["stm"][-1]

        block_vv = stm[3:6, 3:6]
        assert not np.allclose(block_vv, np.eye(3), atol=1e-3), (
            f"Φ[3:,3:] ≈ I 意味着 ∂a/∂v 被置零（旧 bug）；got {block_vv}"
        )

    def test_position_only_stm_velocity_block_is_identity(self):
        """回归：位置型力的 Φ[3:,3:] = I（无速度阻尼），重构后仍成立。

        位置型力 ``a=-k·r`` 的解析 STM 速度-速度块 = ``cos(√k·Δt)·I`` ≠ I
        一般；这里改用恒力（``a=const``，不依赖 r 也不依赖 v）验证 ``∂a/∂v=0``
        时速度块对初始速度的映射保持单位（δv 不变）。
        """

        class _ConstantAccel(PhysicalModel):
            def __init__(self, a):
                self._a = np.asarray(a, dtype=float)

            def compute_acceleration(self, t, state, system):  # noqa: ANN001
                return self._a.copy()

        fm = ForceModel(_FakeSystem(), forces=[_ConstantAccel([0.1, 0.0, 0.0])])
        fm.rtol = 1e-12
        fm.atol = 1e-12

        state0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        stm = fm.propagate(state0, (0.0, 1.0), with_stm=True, initial_step=0.05)["stm"][-1]

        # 恒力：δv(t)=δv0（速度不演化），故 Φ[3:,3:]=I、Φ[3:,:3]=0
        assert_allclose(stm[3:6, 3:6], np.eye(3), atol=1e-9, err_msg="恒力 Φ[3:,3:] 应 = I")
        assert_allclose(stm[3:6, :3], np.zeros((3, 3)), atol=1e-9, err_msg="恒力 Φ[3:,:3] 应 = 0")
