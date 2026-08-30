"""Continuation 延拓算法测试——契约层。

只保留不驱动真实 CR3BP 修正+延拓的轻量覆盖：延拓参数自动推断、步长
配置契约、参数索引推断与统计初始化。真实延拓行为（双向延拓、步长控制、
端到端 pipeline）已按 ADR 0037 预算决策移出默认套件，由
tests/algorithm/design/continuation/test_continuation_per_family.py 的
5 族延拓链承担最小覆盖。
"""

import pytest

from e2m2e.algorithm.solver.continuation import Continuation

pytestmark = pytest.mark.orchestration


# 公共 fixtures 从 tests/algorithm/conftest.py 导入：
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
        # 这应该成功
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
# 统计信息契约
# ============================================================
class TestContinuationStats:
    """测试统计信息初始化"""

    def test_continuation_stats_initialization(self, dro_continuation):
        """统计信息应该正确初始化"""
        stats = dro_continuation.continuation_stats
        assert stats["total_steps"] == 0
        assert stats["successful_steps"] == 0
        assert stats["failed_steps"] == 0


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
