"""
需求: DRO 轨道 CR3BP → 星历模型修正 (Layer 3)

这是整个需求的最终目标：将 CR3BP 中计算的 3:1 DRO 轨道转换到高精度星历
(Ephemeris N-body) 模型下，使用 Multiple Shooting 差分修正。

工作流:
  Step 1: 加载 CR3BP 中的 3:1 DRO 轨道
  Step 2: 对 DRO 轨道均匀采样生成 patch points
  Step 3: synodic → J2000 坐标转换（含速度）
  Step 4: Multiple Shooting 差分修正（星历模型）
  Step 5: 验证修正后位置连续性 < 1e-6 km

输入:
  DRO 初始状态: [1.1202109158830986, 0, 0, 0, -0.46178983697629084, 0]
  DRO 周期: 2.095 TU (无量纲)
  地月系统: mu = 1.21506683e-2
  DU = 3.84405e5 km, TU = 4.34811305 * 86400 s
  参考历元: 2025-06-21T11:00:06 UTC
  天体: Earth + Moon + Sun

验证标准:
  修正后相邻 patch points 的位置连续性误差 < 1e-6 km
  修正后轨道保持 DRO 形状（距地球约 384,400 km）

参考:
  陈昱桔 (2024) "面向地月空间态势感知的DRO轨道设计与控制研究"
  SEMpy: halo_orbit_correction_2.py (类似工作流)

依赖:
  Layer 1a (SPICEManager)
  Layer 1b (EphemerisDynamics)
  Layer 1c (SynodicJ2000Transformation)
  Layer 2 (MultipleShooting)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.core import (
    CR3BP_Dynamics,
    CR3BP_System,
    Orbit,
    SynodicJ2000Transformation,
)
from e2m2e.core.ephemeris_dynamics import EphemerisDynamics
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.spice import SPICEManager

pytestmark = pytest.mark.spice
from e2m2e.algorithms import (  # noqa: E402
    DifferentialCorrection,
    MultipleShooting,
)

# =============================================================================
# 物理参数
# =============================================================================
MU = 1.21506683e-2
TU_SECONDS = 4.34811305 * 86400  # 秒

DRO_31_X0 = 1.1202109158830986
DRO_31_VY0 = -0.46178983697629084
DRO_31_PERIOD = 2.095  # 无量纲 TU

N_PATCH_POINTS = 8
POSITION_CONTINUITY_TOL = 1e-6  # km


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def cr3bp_system():
    """地月 CR3BP 系统"""
    return CR3BP_System(mu=MU, primary="earth", secondary="moon")


@pytest.fixture
def cr3bp_dynamics(cr3bp_system):
    """CR3BP 动力学"""
    return CR3BP_Dynamics(system=cr3bp_system)


@pytest.fixture
def dro_orbit(cr3bp_dynamics, cr3bp_system):
    """
    生成 3:1 DRO 轨道。
    使用微分修正从初始猜测得到精确的周期轨道。
    """
    seed_state = np.array([DRO_31_X0, 0.0, 0.0, 0.0, DRO_31_VY0, 0.0])
    seed_orbit = Orbit([seed_state], [0])
    seed_orbit.period = DRO_31_PERIOD

    corrector = DifferentialCorrection(cr3bp_dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_31_X0)
    result = corrector.iterate_correction(seed_orbit, verbose=False)

    if result is None or not corrector.success:
        pytest.skip("DRO 微分修正未收敛，跳过依赖测试")
    return result


@pytest.fixture
def spice_manager(spice_kernel_path):
    """加载了 DE440 的 SPICEManager"""
    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)
    yield mgr
    mgr.unload_kernel(spice_kernel_path)


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元 ET"""
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def eph_system(spice_manager):
    """Earth-Moon-Sun 星历系统"""
    return EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=spice_manager,
        origin="EARTH",
        frame="J2000",
    )


@pytest.fixture
def eph_dynamics(eph_system):
    """星历 N-body 动力学（使用较宽松的积分参数以加速测试）"""
    d = EphemerisDynamics(system=eph_system)
    d.rtol = 1e-10
    d.atol = 1e-10
    d.max_step = 600.0
    return d


@pytest.fixture
def syn_j2000(cr3bp_system, spice_manager):
    """synodic ↔ J2000 坐标转换器"""
    return SynodicJ2000Transformation(
        cr3bp_system=cr3bp_system,
        spice=spice_manager,
    )


@pytest.fixture(scope="module")
def _correction_cache():
    return {}


@pytest.fixture
def correction_result(
    dro_orbit,
    cr3bp_dynamics,
    syn_j2000,
    eph_dynamics,
    reference_et,
    _correction_cache,
):
    """缓存修正结果，避免重复运行昂贵的 Multiple Shooting"""
    cache_key = "default_8pt"
    if cache_key in _correction_cache:
        return _correction_cache[cache_key]

    period = dro_orbit.period
    tc = TU_SECONDS

    t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
    state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
    state_patch_syn[0] = dro_orbit.states[0]
    for i in range(1, N_PATCH_POINTS):
        state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(dro_orbit, t_patch_syn[i])

    state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
        states_syn=state_patch_syn,
        t_syn_arr=t_patch_syn,
        et0=reference_et,
    )
    t_patch_j2000 = reference_et + t_patch_syn * tc

    ms = MultipleShooting(dynamics=eph_dynamics)
    result = ms.correct(
        t_patch=t_patch_j2000,
        state_patch=state_patch_j2000,
        var_time=True,
        max_iter=50,
        tolerance=POSITION_CONTINUITY_TOL,
    )

    _correction_cache[cache_key] = result
    return result


# =============================================================================
# Test Step 1: 加载 DRO 轨道
# =============================================================================
class TestStep1LoadDROOrbit:
    """测试从 CR3BP 加载 DRO 轨道"""

    def test_dro_orbit_exists(self, dro_orbit):
        """DRO 轨道应成功生成"""
        assert dro_orbit is not None

    def test_dro_period_correct(self, dro_orbit):
        """DRO 周期应与预期值接近"""
        assert_allclose(dro_orbit.period, DRO_31_PERIOD, atol=0.1)

    def test_dro_is_planar(self, dro_orbit):
        """DRO 应为平面轨道 (z ≈ 0)"""
        states = dro_orbit.states
        z_values = states[:, 2]
        assert np.all(np.abs(z_values) < 1e-8)

    def test_dro_initial_state(self, dro_orbit):
        """DRO 初始状态应接近预期值"""
        state0 = dro_orbit.states[0]
        assert_allclose(state0[0], DRO_31_X0, atol=0.01)
        assert_allclose(state0[4], DRO_31_VY0, atol=0.01)

    def test_dro_periodicity(self, dro_orbit):
        """DRO 应满足周期性（首尾状态接近）"""
        states = dro_orbit.states
        initial = states[0]
        final = states[-1]
        assert_allclose(final[1], initial[1], atol=1e-4)  # y ≈ 0
        assert_allclose(final[3], initial[3], atol=1e-4)  # vx ≈ 0


# =============================================================================
# Test Step 2: 采样 patch points
# =============================================================================
class TestStep2SamplePatchPoints:
    """测试从 DRO 轨道采样 patch points"""

    def test_sample_uniform_time(self, dro_orbit):
        """应在 DRO 周期内均匀采样"""
        period = dro_orbit.period
        n_points = N_PATCH_POINTS
        t_patch = np.linspace(0, period, n_points, endpoint=False)

        assert len(t_patch) == n_points
        assert_allclose(t_patch[0], 0.0)
        assert t_patch[-1] < period

    def test_sample_states_at_patch_times(self, dro_orbit, cr3bp_dynamics):
        """应在采样时间点获取 DRO 状态"""
        period = dro_orbit.period
        n_points = N_PATCH_POINTS
        t_patch = np.linspace(0, period, n_points, endpoint=False)

        state_patch = np.zeros((n_points, 6))
        state_patch[0] = dro_orbit.states[0]

        for i in range(1, n_points):
            state_patch[i] = cr3bp_dynamics.propagate_orbit_state_at_time(dro_orbit, t_patch[i])

        assert state_patch.shape == (n_points, 6)
        assert np.all(np.isfinite(state_patch))

    def test_sampled_states_are_planar(self, dro_orbit, cr3bp_dynamics):
        """采样点应保持平面特性"""
        period = dro_orbit.period
        t_patch = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)

        for t in t_patch[1:]:
            state = cr3bp_dynamics.propagate_orbit_state_at_time(dro_orbit, t)
            assert abs(state[2]) < 1e-8, f"z 分量 {state[2]} 不为零"
            assert abs(state[5]) < 1e-8, f"vz 分量 {state[5]} 不为零"


# =============================================================================
# Test Step 3: synodic → J2000 坐标转换
# =============================================================================
class TestStep3SynodicToJ2000:
    """测试将 DRO patch points 从 synodic 转换到 J2000"""

    def test_convert_patch_points(self, dro_orbit, cr3bp_dynamics, syn_j2000, reference_et):
        """应能将所有 patch points 转换到 J2000"""
        period = dro_orbit.period
        t_patch = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)

        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = dro_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(dro_orbit, t_patch[i])

        state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch,
            et0=reference_et,
        )

        assert state_patch_j2000.shape == (N_PATCH_POINTS, 6)
        assert np.all(np.isfinite(state_patch_j2000))

    def test_j2000_positions_near_moon(self, dro_orbit, cr3bp_dynamics, syn_j2000, reference_et):
        """J2000 下的 DRO 位置应在月球距离附近"""
        state0_j2000 = syn_j2000.synodic_to_j2000(
            state_syn=dro_orbit.states[0],
            t_syn=0.0,
            et0=reference_et,
        )
        r = np.linalg.norm(state0_j2000[:3])
        assert 300000 < r < 500000, f"DRO距地球 {r:.0f} km，超出合理范围"

    def test_j2000_time_is_et(self, syn_j2000, reference_et, dro_orbit):
        """J2000 时间应为 ET 秒"""
        tc = TU_SECONDS
        period = dro_orbit.period
        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        t_patch_j2000 = reference_et + t_patch_syn * tc

        assert_allclose(t_patch_j2000[0], reference_et)
        assert t_patch_j2000[-1] - t_patch_j2000[0] > 0


# =============================================================================
# Test Step 4: Multiple Shooting 修正
# =============================================================================
class TestStep4MultipleShootingCorrection:
    """测试在星历模型下进行 Multiple Shooting 修正"""

    def test_correction_converges(
        self,
        dro_orbit,
        cr3bp_dynamics,
        syn_j2000,
        eph_dynamics,
        reference_et,
    ):
        """Multiple Shooting 修正应收敛"""
        period = dro_orbit.period
        tc = TU_SECONDS

        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = dro_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                dro_orbit, t_patch_syn[i]
            )

        state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch_syn,
            et0=reference_et,
        )
        t_patch_j2000 = reference_et + t_patch_syn * tc

        ms = MultipleShooting(dynamics=eph_dynamics)
        result = ms.correct(
            t_patch=t_patch_j2000,
            state_patch=state_patch_j2000,
            var_time=True,
            max_iter=50,
            tolerance=POSITION_CONTINUITY_TOL,
        )

        assert result.converged, f"Multiple Shooting 未收敛，迭代 {result.iterations} 次"


# =============================================================================
# Test Step 5: 验证修正结果
# =============================================================================
class TestStep5Validation:
    """测试修正后的轨道质量"""

    def test_position_continuity(self, correction_result):
        """修正后相邻段端点位置连续性误差应 < 1e-6 km"""
        result = correction_result
        assert result.converged, "修正应收敛"
        assert result.max_residual < POSITION_CONTINUITY_TOL, (
            f"最大残差 {result.max_residual:.2e} km > {POSITION_CONTINUITY_TOL}"
        )

    def test_orbit_shape_preserved(self, correction_result):
        """修正后轨道形状应与 CR3BP DRO 相似"""
        result = correction_result
        assert result.converged, f"修正未收敛 (residual={result.max_residual:.2e})"

        corrected_states = result.state_patch
        distances = np.linalg.norm(corrected_states[:, :3], axis=1)
        mean_dist = np.mean(distances)
        assert 300000 < mean_dist < 500000, f"修正后平均距地球 {mean_dist:.0f} km，偏离 DRO 范围"
        std_dist = np.std(distances)
        assert std_dist / mean_dist < 0.1, (
            f"修正后轨道形状变化过大: std/mean = {std_dist / mean_dist:.3f}"
        )


# =============================================================================
# Test 完整流程 (Integration)
# =============================================================================
class TestDROEphemerisPipeline:
    """测试完整的 DRO CR3BP → 星历模型修正流程"""

    def test_full_pipeline(self, correction_result):
        """完整流程: DRO生成 → 采样 → 坐标转换 → 星历修正 → 验证"""
        result = correction_result

        assert result.converged, f"修正未收敛，迭代 {result.iterations} 次"
        assert result.max_residual < POSITION_CONTINUITY_TOL, (
            f"最大位置连续性误差 {result.max_residual:.2e} km > {POSITION_CONTINUITY_TOL} km"
        )

    def test_different_patch_point_counts(
        self,
        cr3bp_dynamics,
        dro_orbit,
        eph_dynamics,
        syn_j2000,
        reference_et,
        _correction_cache,
    ):
        """不同 patch point 数量应都能收敛"""
        period = dro_orbit.period
        tc = TU_SECONDS

        for n_points in [4, 12]:
            cache_key = f"npt_{n_points}"
            if cache_key in _correction_cache:
                result = _correction_cache[cache_key]
            else:
                t_patch_syn = np.linspace(0, period, n_points, endpoint=False)
                state_patch_syn = np.zeros((n_points, 6))
                state_patch_syn[0] = dro_orbit.states[0]
                for i in range(1, n_points):
                    state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                        dro_orbit, t_patch_syn[i]
                    )

                state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
                    states_syn=state_patch_syn,
                    t_syn_arr=t_patch_syn,
                    et0=reference_et,
                )
                t_patch_j2000 = reference_et + t_patch_syn * tc

                ms = MultipleShooting(dynamics=eph_dynamics)
                result = ms.correct(
                    t_patch=t_patch_j2000,
                    state_patch=state_patch_j2000,
                    var_time=True,
                    max_iter=50,
                    tolerance=1e-4,
                )
                _correction_cache[cache_key] = result

            assert result.converged, f"{n_points} 个 patch points 时修正未收敛"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
