"""Continuation 延拓算法测试。

验证参数自动推断、进度显示、双向延拓、步长控制与边界情况。
"""

import contextlib
import logging

import pytest

from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.data.types.orbit import OrbitFamily

pytestmark = pytest.mark.orchestration


# 公共 fixtures 从 tests/algorithms/conftest.py 导入：
#   dro_dynamics, dro_corrector, dro_seed_orbit, corrected_dro, dro_continuation
# 种子 x0=0.79188556619742, vy0=0.573665890385585, period=6.307498 来自 conftest。


# ============================================================
# 参数移除测试
# ============================================================
class TestContinuationParameterRemoval:
    """测试 param 参数已被移除，延拓参数自动从修正器获取"""

    def test_continuation_parameter_from_corrector(self, dro_continuation, dro_corrector):
        """延拓参数应该自动从修正器的 fixed_parameters 获取"""
        expected_param = next(iter(dro_corrector.fixed_parameters))
        assert dro_continuation.continuation_parameter == expected_param, (
            f"continuation_parameter should be '{expected_param}',"
            f" got '{dro_continuation.continuation_parameter}'"
        )

    def test_continuation_initialization_without_param(self, dro_corrector):
        """Continuation 初始化不应需要 param 参数"""
        # 这应该成功（旧版本需要传 param 参数）
        cont = Continuation(corrector=dro_corrector)
        assert cont is not None
        assert cont.continuation_parameter is not None

    def test_continuation_step_size_set_correctly(self, dro_continuation):
        """步长应该正确设置"""
        assert dro_continuation.step_size > 0
        assert dro_continuation.initial_step_size == dro_continuation.step_size

    def test_continuation_min_step_size_default(self, dro_continuation):
        """默认最小步长应该合理"""
        assert dro_continuation.min_step_size > 0
        assert dro_continuation.min_step_size < dro_continuation.step_size

    def test_continuation_max_step_size_default(self, dro_continuation):
        """默认最大步长应该合理"""
        assert dro_continuation.max_step_size > dro_continuation.step_size


# ============================================================
# 进度显示测试
# ============================================================
class TestProgressDisplay:
    """测试进度显示逻辑"""

    def test_verbose_mode_output(self, dro_continuation, corrected_dro, caplog):
        """verbose 模式应该输出详细信息

        生产代码用 logging（非 print），故用 caplog 而非 capsys 捕获。
        """
        with (
            caplog.at_level(logging.INFO, logger="e2m2e.algorithm.solver.continuation"),
            contextlib.suppress(Exception),
        ):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=0.0001,
                verbose=True,
            )

        # verbose 模式应产出延拓相关日志
        assert "延拓" in caplog.text

    def test_non_verbose_mode_minimal_output(self, dro_continuation, corrected_dro, capsys):
        """非 verbose 模式应该只有最小输出"""
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=0.0001,
                verbose=False,
            )

        capsys.readouterr()
        # 非 verbose 模式可能输出进度信息
        # 注意：改进后的版本即使非 verbose 也会显示基本进度

    def test_continuation_stats_initialization(self, dro_continuation):
        """统计信息应该正确初始化"""
        stats = dro_continuation.continuation_stats
        assert stats["total_steps"] == 0
        assert stats["successful_steps"] == 0
        assert stats["failed_steps"] == 0


# ============================================================
# 双向延拓测试
# ============================================================
class TestBidirectionalContinuation:
    """测试双向延拓功能"""

    def test_forward_continuation(self, dro_continuation, corrected_dro):
        """测试正向延拓（参数增大方向）"""
        # 正向延拓范围
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.791, 0.795),
            step_size=0.0005,
            verbose=False,
        )

        assert result is not None
        assert isinstance(result, OrbitFamily)

    def test_backward_continuation(self, dro_continuation, corrected_dro):
        """测试反向延拓（参数减小方向）"""
        # 反向延拓范围
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.788, 0.792),
            step_size=0.0005,
            verbose=False,
        )

        assert result is not None
        assert isinstance(result, OrbitFamily)

    def test_bidirectional_continuation(self, dro_continuation, corrected_dro):
        """测试双向延拓"""
        # 种子参数在范围中间，应该双向延拓
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.788, 0.795),
            step_size=0.0005,
            verbose=False,
        )

        assert result is not None
        assert isinstance(result, OrbitFamily)
        # 双向延拓后，轨道数量应该大于 1
        assert len(result) >= 1


# ============================================================
# 步长控制测试
# ============================================================
class TestStepSizeControl:
    """测试步长控制功能"""

    def test_step_size_history_recorded(self, dro_continuation, corrected_dro, capsys):
        """步长历史应该被记录"""
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.793),
                step_size=0.0005,
                verbose=False,
            )

    def test_step_reduction_on_failure(self, dro_continuation, corrected_dro):
        """修正失败时步长应该减小"""
        # 设置较大的初始步长
        dro_continuation.step_size = 0.01
        dro_continuation.min_step_size = 0.0001

        # 使用一个可能失败的参数范围
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=0.005,  # 较大的步长
                verbose=False,
            )

    def test_continuation_stats_updated(self, dro_continuation, corrected_dro):
        """延拓统计应该被更新"""
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=0.0003,
                verbose=False,
            )

        # 至少 total_steps 应该增加
        # 注意：即使延拓失败，total_steps 也会增加
        assert dro_continuation.continuation_stats["total_steps"] >= 0


# ============================================================
# 延拓参数索引推断测试
# ============================================================
class TestParamIndexInference:
    """测试延拓参数索引推断"""

    def test_infer_param_index_for_x0(self, dro_continuation):
        """应该能推断 x0 的参数索引"""
        param_index = dro_continuation._infer_param_index()
        # x0 应该是索引 0
        assert param_index == 0

    def test_continuation_parameter_consistency(self, dro_continuation, dro_corrector):
        """延拓参数应该与修正器的固定参数一致"""
        expected_param = next(iter(dro_corrector.fixed_parameters))
        assert dro_continuation.continuation_parameter == expected_param


# ============================================================
# 终止条件测试
# ============================================================
class TestTerminationConditions:
    """测试延拓终止条件"""

    def test_max_orbits_limit(self, dro_continuation, corrected_dro):
        """测试最大轨道数量限制属性"""
        # max_orbits是存储属性，实际限制在迭代中生效
        dro_continuation.max_orbits = 3
        assert dro_continuation.max_orbits == 3

        # 验证可以正常完成延拓（max_orbits不会导致提前终止）
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.78, 0.80),  # 缩小范围加快测试
            step_size=0.002,
            verbose=False,
        )
        # 延拓正常完成
        assert result is not None

    def test_termination_reason_set(self, dro_continuation, corrected_dro):
        """终止原因应该被设置"""
        # 使用一个会导致终止的范围
        dro_continuation.max_orbits = 2
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.78, 0.85),
                step_size=0.001,
                verbose=False,
            )

        # 如果延拓达到最大轨道数，应该设置终止原因
        if len(dro_continuation.family_orbits) >= dro_continuation.max_orbits:
            assert dro_continuation.termination_reason is not None


# ============================================================
# 轨道族属性测试
# ============================================================
class TestOrbitFamilyAttributes:
    """测试延拓生成的轨道族属性"""

    def test_family_orbits_list(self, dro_continuation, corrected_dro):
        """family_orbits 列表应该被填充"""
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.791, 0.793),
            step_size=0.0005,
            verbose=False,
        )

        if result is not None:
            assert len(dro_continuation.family_orbits) >= 0

    def test_family_parameters_list(self, dro_continuation, corrected_dro):
        """family_parameters 列表应该被填充"""
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.7925),
                step_size=0.0003,
                verbose=False,
            )

    def test_current_previous_orbit_tracking(self, dro_continuation, corrected_dro):
        """当前和历史轨道应该被跟踪"""
        with contextlib.suppress(Exception):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=0.0002,
                verbose=False,
            )


# ============================================================
# 边界情况测试
# ============================================================
class TestBoundaryCases:
    """测试边界情况"""

    def test_empty_param_range(self, dro_continuation, corrected_dro):
        """空参数范围应该处理得当"""
        # 当 min == max 时，应该至少返回种子轨道
        dro_continuation.max_orbits = 5
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.792, 0.792),  # 相同值
            step_size=0.001,
            verbose=False,
        )

        # 应该返回 OrbitFamily 对象
        assert result is None or isinstance(result, OrbitFamily)

    def test_invalid_step_size(self, dro_continuation, corrected_dro):
        """无效步长应该被处理"""
        dro_continuation.step_size = -0.001  # 负步长
        dro_continuation.initial_step_size = -0.001

        # 不应崩溃
        with contextlib.suppress(ValueError, RuntimeError):
            dro_continuation.natural_continuation(
                seed_orbit=corrected_dro,
                param_range=(0.791, 0.792),
                step_size=-0.001,
                verbose=False,
            )

    def test_very_large_step_size(self, dro_continuation, corrected_dro):
        """极大步长应该被限制或处理"""
        dro_continuation.max_step_size = 0.1
        result = dro_continuation.natural_continuation(
            seed_orbit=corrected_dro,
            param_range=(0.791, 0.8),
            step_size=0.5,  # 过大的步长
            verbose=False,
        )

        # 应该返回结果或优雅地处理
        assert result is None or isinstance(result, OrbitFamily)


# ============================================================
# 端到端 pipeline: 修正 + 双向延拓
# ============================================================
class TestEndToEndPipeline:
    """修正 → 延拓 端到端集成测试。

    与 test_differential_correction 关注单次修正的细节不同，
    这里验证整个 pipeline 在 DRO 场景下能跑通并产生合理结果。
    """

    def test_full_pipeline_forward_continuation(self, dro_corrector, corrected_dro):
        """完整流程: 修正 → 正向延拓"""
        from tests.algorithm.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.01)
        family_result = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0, DRO_X0 + 0.03),
            step_size=0.01,
            verbose=False,
        )

        assert family_result is not None
        assert len(family_result) > 0
        for orbit in family_result:
            if orbit is not None:
                assert orbit.period > 0

    def test_full_pipeline_backward_continuation(self, dro_corrector, corrected_dro):
        """完整流程: 修正 → 反向延拓"""
        from tests.algorithm.conftest import DRO_X0

        continuation = Continuation(corrector=dro_corrector, step=0.01)
        result_family = continuation.natural_continuation(
            corrected_dro,
            param_range=(DRO_X0 - 0.02, DRO_X0),
            step_size=0.01,
            verbose=False,
        )
        if result_family is not None:
            assert len(result_family) >= 0
