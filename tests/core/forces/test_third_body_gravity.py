"""ThirdBodyGravity 测试（issue #182）。

验证第三体引力摄动模型：
  A. 单点加速度与 EphemerisDynamics 的第三体分支逐字一致；
  B. 力分解路径 (ForceModel) 与 EphemerisDynamics 传播同一条 cislunar
     轨道，末状态位置差自洽。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.forces import (
    ForceModel,
    PhysicalModel,
    PointMassGravity,
    ThirdBodyGravity,
)
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

pytestmark = pytest.mark.spice

# 公共 SPICE fixtures 来自 tests/conftest.py:
#   spice_manager, spice_eph_system, spice_eph_dynamics, reference_et,
#   reference_epoch


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元的 ET 秒数（与 test_ephemeris_dynamics.py 一致）。"""
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def dro_state():
    """月球距离附近的 cislunar 初值（J2000, km, km/s）。

    与 ``test_ephemeris_dynamics.py::test_propagate_dro_like_orbit`` 一致。
    """
    return np.array([384400.0, 0.0, 0.0, 0.0, -0.5, 0.0])


@pytest.fixture
def leo_state():
    """近地轨道初始状态（J2000, km, km/s）。"""
    r = 6778.0
    v = np.sqrt(398600.436 / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0])


def _third_body_contribution(dynamics, t, r_sc, body):
    """复现 EphemerisDynamics 单个摄动天体的第三体贡献（直接 + 间接）。

    用同一公式 ``-gm * (r_bsc/|r_bsc|³ + r_ob/|r_ob|³)``，作为
    ``ThirdBodyGravity`` 的独立对照。
    """
    system = dynamics.system
    gm = system.get_gm(body)
    r_ob = np.asarray(system.get_body_position(body, t), dtype=float)
    r_bsc = np.asarray(r_sc, dtype=float) - r_ob
    r_bsc_norm = max(float(np.linalg.norm(r_bsc)), dynamics.MIN_DISTANCE)
    r_ob_norm = max(float(np.linalg.norm(r_ob)), dynamics.MIN_DISTANCE)
    return -gm * (r_bsc / r_bsc_norm**3 + r_ob / r_ob_norm**3)


# =============================================================================
# 基本接口
# =============================================================================
class TestThirdBodyGravityInterface:
    """测试类的基本构造与接口。"""

    def test_is_physical_model(self):
        """ThirdBodyGravity 应是 PhysicalModel 子类。"""
        assert issubclass(ThirdBodyGravity, PhysicalModel)

    def test_body_uppercased(self):
        """body 应被存为大写。"""
        force = ThirdBodyGravity(body="moon")
        assert force.body == "MOON"

    def test_compute_acceleration_shape(
        self, spice_eph_system, reference_et, dro_state
    ):
        """加速度输出应为 (3,)。"""
        force = ThirdBodyGravity("MOON")
        acc = force.compute_acceleration(reference_et, dro_state, spice_eph_system)
        assert acc.shape == (3,)
        assert np.all(np.isfinite(acc))

    def test_no_origin_parameter_exposed(self):
        """构造函数不应暴露 origin 参数（接口约定）。"""
        import inspect

        sig = inspect.signature(ThirdBodyGravity.__init__)
        assert "origin" not in sig.parameters


# =============================================================================
# A. 单元测试：单点加速度 == EphemerisDynamics 第三体分支
# =============================================================================
class TestThirdBodyAccelMatchesEphemeris:
    """单点加速度逐字对齐 EphemerisDynamics 的第三体分支。"""

    def test_moon_single_point_matches_ephemeris_branch(
        self, spice_eph_dynamics, spice_eph_system, reference_et, dro_state
    ):
        """ThirdBodyGravity("MOON") 的单点加速度 == EphemerisDynamics 月球第三体增量。"""
        expected = _third_body_contribution(
            spice_eph_dynamics, reference_et, dro_state[:3], "MOON"
        )

        force = ThirdBodyGravity("MOON")
        acc = force.compute_acceleration(reference_et, dro_state, spice_eph_system)

        assert_allclose(acc, expected, atol=1e-12)

    def test_sun_single_point_matches_ephemeris_branch(
        self, spice_eph_dynamics, spice_eph_system, reference_et, dro_state
    ):
        """ThirdBodyGravity("SUN") 的单点加速度 == EphemerisDynamics 太阳第三体增量。"""
        expected = _third_body_contribution(
            spice_eph_dynamics, reference_et, dro_state[:3], "SUN"
        )

        force = ThirdBodyGravity("SUN")
        acc = force.compute_acceleration(reference_et, dro_state, spice_eph_system)

        assert_allclose(acc, expected, atol=1e-12)

    def test_sum_decomposition_matches_total(
        self, spice_eph_dynamics, spice_eph_system, reference_et, dro_state
    ):
        """PointMass(EARTH) + ThirdBody(MOON) + ThirdBody(SUN) == EphemerisDynamics 总加速度。

        将 EphemerisDynamics 的总加速度（地+月+日）分解为：
        地球中心引力 + 月球第三体 + 太阳第三体，验证二者一致。
        """
        # EphemerisDynamics 的总加速度（不含 STM 部分的雅可比）
        total_acc, _ = spice_eph_dynamics._compute_acc_and_jacobian(
            reference_et, dro_state[:3], need_jacobian=False
        )

        # 力分解路径：地球中心引力 + 月/日第三体
        earth = PointMassGravity("EARTH")
        moon = ThirdBodyGravity("MOON")
        sun = ThirdBodyGravity("SUN")
        decomposed = (
            earth.compute_acceleration(reference_et, dro_state, spice_eph_system)
            + moon.compute_acceleration(reference_et, dro_state, spice_eph_system)
            + sun.compute_acceleration(reference_et, dro_state, spice_eph_system)
        )

        assert_allclose(decomposed, total_acc, atol=1e-9)


# =============================================================================
# B. 自洽性主验收测试：力分解路径 vs EphemerisDynamics
# =============================================================================
class TestCislunarForceDecomposition:
    """力分解路径与 EphemerisDynamics 传播同一条 cislunar 轨道的自洽性。"""

    def test_cislunar_force_model_matches_ephemeris_dynamics(
        self,
        spice_eph_system,
        spice_eph_dynamics,
        spice_manager,
        reference_et,
        dro_state,
    ):
        """传播 ~9 天 DRO 类轨道，两条路径末状态位置差应在积分容差内一致。

        路径 1：``EphemerisDynamics.propagate``（scipy DOP853）。
        路径 2：``ForceModel``（Rust rk_step PD45）+ PointMass(EARTH)
                + ThirdBody(MOON) + ThirdBody(SUN)。

        二者积分器与步长策略不同，但物理模型应等价；末状态位置差
        反映的是数值积分差异，应远小于 1 km。
        """
        system = spice_eph_system
        # ForceModel 需要带 coordinate_system 的 system；spice_eph_system 默认未设。
        if getattr(system, "coordinate_system", None) is None:
            system.coordinate_system = CoordinateSystem(
                axes=ICRSAxes(),
                origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
            )

        t_span = (reference_et, reference_et + 9.11 * 86400.0)

        # 路径 1：EphemerisDynamics（与 test_propagate_dro_like_orbit 同配置）
        result_eph = spice_eph_dynamics.propagate(dro_state, t_span)
        final_eph = np.asarray(result_eph["states"][-1], dtype=float)

        # 路径 2：力分解
        fm = ForceModel(
            system,
            forces=[
                PointMassGravity("EARTH"),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
            ],
        )
        # 与 EphemerisDynamics 宽松测试配置对齐：rtol/atol=1e-10, max_step=600s
        fm.rtol = spice_eph_dynamics.rtol
        fm.atol = spice_eph_dynamics.atol
        fm.max_step = spice_eph_dynamics.max_step

        result_fm = fm.propagate(dro_state, t_span, max_steps=1_000_000)
        final_fm = np.asarray(result_fm["states"][-1], dtype=float)

        pos_diff_km = float(np.linalg.norm(final_eph[:3] - final_fm[:3]))

        # 实测：两条路径末状态位置差 ~1e-6 km（亚毫米级）。这是因为两条路径
        # 的物理模型逐字等价，差异仅来自不同积分器（scipy DOP853 vs Rust
        # rk_step PD45）的数值积分误差。判据设为 1e-3 km（1 m），约为实测
        # 值的 3 个数量级余量，足以抵御不同 SPICE 内核版本（DE440/DE440s）
        # 带来的极小星历差异，同时远严于 issue #182 的 1 km 上界。
        assert pos_diff_km < 1e-3, (
            f"cislunar force-decomposition path diverged from EphemerisDynamics: "
            f"|Δr|={pos_diff_km:.4e} km (threshold 1e-3 km)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
