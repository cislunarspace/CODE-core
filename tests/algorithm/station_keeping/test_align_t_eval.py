"""RustPropagator._align_t_eval 首点对齐回归测试。

验证：t_eval 首点 != t0 时前置补 t0，返回 (对齐后数组, True)；
首点 == t0 时原样返回 (原数组, False)。
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from e2m2e.algorithm.station_keeping.monte_carlo import RustPropagator

pytestmark = pytest.mark.orchestration


class TestAlignTEval:
    """RustPropagator._align_t_eval 首点对齐。"""

    def test_already_aligned(self):
        """t_eval[0] == t0 → 原样返回，prepended=False。"""
        t0 = 100.0
        t_eval = np.array([100.0, 200.0, 300.0])
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert not prepended
        assert_array_equal(result, t_eval)

    def test_prepends_t0(self):
        """t_eval[0] != t0 → 前置 t0，prepended=True。"""
        t0 = 100.0
        t_eval = np.array([200.0, 300.0, 400.0])
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert prepended
        assert result[0] == t0
        assert_array_equal(result, np.array([100.0, 200.0, 300.0, 400.0]))

    def test_empty_t_eval(self):
        """空 t_eval → 原样返回，prepended=False。"""
        t0 = 0.0
        t_eval = np.array([])
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert not prepended
        assert len(result) == 0

    def test_single_point_not_t0(self):
        """单点 t_eval != t0 → 补 t0，返回 2 元素数组。"""
        t0 = 0.0
        t_eval = np.array([500.0])
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert prepended
        assert_array_equal(result, np.array([0.0, 500.0]))

    def test_single_point_is_t0(self):
        """单点 t_eval == t0 → 原样返回，prepended=False。"""
        t0 = 500.0
        t_eval = np.array([500.0])
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert not prepended
        assert_array_equal(result, t_eval)

    def test_list_input(self):
        """list 输入（非 ndarray）也能正确处理。"""
        t0 = 0.0
        t_eval = [10.0, 20.0]
        result, prepended = RustPropagator._align_t_eval(t0, t_eval)
        assert prepended
        assert result[0] == t0
        assert len(result) == 3
