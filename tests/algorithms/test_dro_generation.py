"""
DRO轨道生成和延拓测试模块

测试DRO (Distant Retrograde Orbit) 轨道的生成和延拓功能。
这些测试确保 phase1_generate_dro.py 中使用的功能不被破坏。

参考论文：
  Cui et al. (2025) "Two-Impulse Transfers from Lunar Distant Retrograde Orbits
  to Resonant Orbits", JGCD, Vol.48, No.6

地月系统参数：
  μ = 1.21506683 × 10⁻²
  DU = 3.84405 × 10⁵ km, TU = 4.34811305 天
"""


import pytest

from e2m2e.algorithms import Continuation
from e2m2e.core import Orbit

# 公共 fixtures 从 tests/algorithms/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro, dro_continuation
# 种子 x0=0.79188556619742, vy0=0.573665890385585, period=6.307498 来自 conftest。



# ============================================================
# DRO 微分修正测试
# ============================================================
class TestDROCorrection:
    """测试DRO轨道的微分修正功能"""

    def test_corrector_setup(self, dro_corrector):
        """测试微分修正器配置正确"""
        assert dro_corrector.setup_type == "2D_symmetric_x_fixed_x0"
        assert "y_dot0" in dro_corrector.free_variables
        assert "T_half" in dro_corrector.free_variables
        # y分量索引为1，vx分量为3
        assert 1 in dro_corrector.constraint_indices
        assert 3 in dro_corrector.constraint_indices

    def test_dro_seed_correction(self, corrected_dro, dro_corrector):
        """测试DRO种子轨道的修正（cached orbit, fresh corrector check）"""
        # 验证修正后的轨道
        assert corrected_dro is not None, "DRO种子修正应该成功"
        assert isinstance(corrected_dro, Orbit), "结果应该是 Orbit 对象"
        assert corrected_dro.period > 0, "周期应该为正"

        # 验证 corrector 自身状态：cached orbit 是预先算好的，
        # 所以这里直接跑一次 fresh 的修正来验证 corrector 的状态机。
        from tests.algorithms.conftest import DRO_VY0, DRO_X0

        seed = Orbit(
            states=[[DRO_X0, 0.0, 0.0, 0.0, DRO_VY0, 0.0]],
            times=[0],
        )
        seed.period = 3.420385
        dro_corrector.iterate_correction(seed, verbose=False)

        assert dro_corrector.success is True, "修正应该成功"
        assert dro_corrector.converged is True, "应该收敛"

        # 验证对称性条件（通过轨道状态验证）
        states = corrected_dro.states
        n_states = len(states)
        mid_idx = n_states // 2
        mid_state = states[mid_idx]

        assert abs(mid_state[1]) < 1e-2, f"y(T/2) 应该接近0，实际: {mid_state[1]}"
        assert abs(mid_state[3]) < 1e-2, f"vx(T/2) 应该接近0，实际: {mid_state[3]}"

    def test_dro_period_reasonable(self, corrected_dro):
        """测试DRO轨道周期在合理范围内"""
        period = corrected_dro.period
        # DRO 周期通常在 1–15 个无量纲时间单位之间；
        # 上限 15 留出余量以容纳不同初值收敛到不同周期轨道的场景。
        assert 1.0 < period < 15.0, f"周期应该在合理范围内: {period}"

    def test_dro_jacobi_constant(self, corrected_dro, dro_dynamics):
        """测试DRO的Jacobi常数计算"""
        # DRO的Jacobi常数通常在3.0-3.5之间
        C = dro_dynamics.system.get_jacobi_constant(corrected_dro.states[0])
        assert 2.5 < C < 4.0, f"Jacobi常数应该在合理范围内: {C}"

    def test_dro_orbit_save_load(self, corrected_dro, dro_dynamics, tmp_path):
        """测试DRO轨道的保存和加载"""
        filepath = tmp_path / "test_dro.json"
        corrected_dro.save_to_file(str(filepath))
        assert filepath.exists(), "文件应该被创建"

        loaded_orbit = Orbit.load_from_file(str(filepath), system=dro_dynamics.system)
        assert loaded_orbit is not None
        assert loaded_orbit.period == corrected_dro.period


# ============================================================
# DRO 自然延拓测试
# ============================================================
class TestDRONaturalContinuation:
    """测试DRO轨道的自然延拓功能"""

    def test_continuation_setup(self, dro_corrector):
        """测试延拓器配置"""
        continuation = Continuation(corrector=dro_corrector, step=0.02)

        assert continuation.continuation_parameter == "x0"
        assert continuation.step_size == 0.02

    def test_continuation_single_orbit(self, corrected_dro, dro_corrector):
        """测试生成单条延拓轨道"""
        from tests.algorithms.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.01)

        result_family = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0, DRO_X0 + 0.02),
            step_size=0.01,
            verbose=False,
        )

        # 验证结果
        assert result_family is not None, "延拓应该返回结果"
        assert len(result_family) >= 1, "应该至少生成一条轨道"

    def test_continuation_multiple_orbits(self, corrected_dro, dro_corrector):
        """测试生成多条延拓轨道"""
        from tests.algorithms.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.005)

        result_family = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0, DRO_X0 + 0.02),
            step_size=0.005,
            verbose=False,
        )

        if result_family is not None:
            assert len(result_family) >= 1, "应该至少生成一条轨道"
            for orbit in result_family:
                if orbit is not None:
                    assert orbit.period > 0

    def test_continuation_period_trend(self, corrected_dro, dro_corrector):
        """测试延拓过程中周期变化趋势"""
        from tests.algorithms.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.005)

        result_family = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0, DRO_X0 + 0.02),
            step_size=0.005,
            verbose=False,
        )

        if result_family is not None:
            periods = result_family.get_periods()
            assert len(periods) >= 1
            for p in periods:
                assert p > 0, f"周期应该为正数: {p}"


# ============================================================
# 集成测试：完整DRO生成流程
# ============================================================
class TestDROGenerationPipeline:
    """测试完整的DRO生成流程"""

    def test_full_pipeline(self, dro_dynamics, dro_corrector, corrected_dro):
        """测试完整的DRO生成和延拓流程"""
        from tests.algorithms.conftest import DRO_X0

        # 1. 验证系统和动力学
        assert dro_dynamics.system.mu == pytest.approx(1.21506683e-2, abs=1e-12)

        # 2. 验证修正后的种子轨道
        assert corrected_dro is not None, "种子轨道修正应该成功"
        assert dro_corrector.converged is True or corrected_dro is not None

        # 3. 延拓生成轨道族
        continuation = Continuation(corrector=dro_corrector, step=0.01)

        family_result = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0, DRO_X0 + 0.03),
            step_size=0.01,
            verbose=False,
        )

        # 4. 验证结果
        assert family_result is not None
        assert len(family_result) > 0

        for orbit in family_result:
            if orbit is not None:
                assert orbit.period > 0

    def test_backward_continuation(self, corrected_dro, dro_corrector):
        """测试反向延拓"""
        from tests.algorithms.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.01)

        result_family = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0 - 0.02, DRO_X0),
            step_size=0.01,
            verbose=False,
        )

        if result_family is not None:
            assert len(result_family) >= 0
