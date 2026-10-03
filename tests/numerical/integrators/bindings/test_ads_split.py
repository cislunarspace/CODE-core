"""dace-rs ADS 自动域分裂驱动器的行为测试（issue #787）。

覆盖六类事实：sin 大盒的叶覆盖、采样包含与确定性；近极点有理函数的
自适应加密；预算与方向上限触顶的显式未达标；收紧容差单调收紧包围；
Jacobi 常数端到端冒烟；参数校验与回调异常透传。容差按计划预留的数值
调整权标定：sin 与有理函数的全达标叶数按 ∫|f'|dx/tol 估计，选在
max_leaves 预算内全达标的紧容差。DA 上下文是进程级全局状态，每个用例
开头各自 da_init_py，不跨用例持对象。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.data.constants import Datum
from e2m2e.integrators import AdsLeaf, AdsResult, da_ads_split, da_init_py

pytestmark = [
    pytest.mark.theory,
    requires_native_symbols("da_ads_split", "AdsLeaf", "AdsResult", "Da", "da_init_py"),
]


def test_sin_big_box_enclosure_coverage_determinism():
    """sin 大盒：全达标、无缝拼接、采样真值逐叶包含、逐叶确定性。"""
    da_init_py(8, 1)
    func = lambda xs: [xs[0].sin()]  # noqa: E731
    result = da_ads_split(func, [(-1.2, 1.2)], [1e-3], max_leaves=4096)

    assert isinstance(result, AdsResult)
    assert all(isinstance(leaf, AdsLeaf) for leaf in result.leaves)
    assert result.met_leaves == len(result.leaves)
    assert all(leaf.met for leaf in result.leaves)
    assert len(result.leaves) >= 2

    leaves = sorted(result.leaves, key=lambda leaf: leaf.center[0])
    widths = [2.0 * leaf.half_width[0] for leaf in leaves]
    assert math.isclose(math.fsum(widths), 2.4, rel_tol=1e-12)
    for lower, upper in zip(leaves, leaves[1:], strict=False):
        seam_lo = lower.center[0] + lower.half_width[0]
        seam_hi = upper.center[0] - upper.half_width[0]
        assert abs(seam_lo - seam_hi) <= 1e-15

    for leaf in leaves:
        lo, hi = leaf.bounds[0]
        assert hi - lo <= 1e-3
        for frac in (-0.8, -0.4, 0.0, 0.4, 0.8):
            point = leaf.center[0] + frac * leaf.half_width[0]
            assert lo - 1e-12 <= math.sin(point) <= hi + 1e-12

    repeat = da_ads_split(func, [(-1.2, 1.2)], [1e-3], max_leaves=4096)
    signature = lambda r: [  # noqa: E731
        (leaf.center, leaf.half_width, leaf.met) for leaf in r.leaves
    ]
    assert signature(result) == signature(repeat)
    assert result.splits_per_var == repeat.splits_per_var


def test_rational_near_pole_no_false_convergence():
    """极点在盒外 0.05 的有理函数：全达标、真值在界内、近极点确实加密。"""
    da_init_py(8, 1)
    func = lambda xs: [1.0 / (xs[0] - 1.05)]  # noqa: E731
    result = da_ads_split(func, [(0.0, 1.0)], [1e-2], max_leaves=8192)

    assert result.met_leaves == len(result.leaves)
    assert result.splits_per_var[0] >= 1
    leaves = sorted(result.leaves, key=lambda leaf: leaf.center[0])
    for leaf in leaves:
        lo, hi = leaf.bounds[0]
        assert hi - lo <= 1e-2
        for frac in (-0.8, -0.4, 0.0, 0.4, 0.8):
            point = leaf.center[0] + frac * leaf.half_width[0]
            reference = 1.0 / (point - 1.05)
            assert lo <= reference <= hi
    assert leaves[-1].half_width[0] < 0.5


def test_budget_and_direction_caps_flag_unmet():
    """预算触顶与方向上限触顶：显式 met=False，不谎报达标。"""
    da_init_py(4, 1)
    func = lambda xs: [xs[0].sin()]  # noqa: E731

    capped = da_ads_split(func, [(-1.2, 1.2)], [1e-10], max_leaves=3)
    assert len(capped.leaves) <= 3
    assert all(leaf.met is False for leaf in capped.leaves)
    assert capped.met_leaves == 0

    frozen = da_ads_split(func, [(-1.2, 1.2)], [1e-10], max_splits_per_var=0)
    assert len(frozen.leaves) == 1
    assert frozen.leaves[0].met is False
    assert frozen.splits_per_var == [0]


def test_tolerance_tightening_monotone():
    """收紧容差：全达标下全体叶子包围宽度之和非增（确定性分裂，容 1e-9）。"""
    da_init_py(8, 1)
    func = lambda xs: [xs[0].sin()]  # noqa: E731
    loose = da_ads_split(func, [(-1.2, 1.2)], [1e-2], max_leaves=8192)
    tight = da_ads_split(func, [(-1.2, 1.2)], [1e-3], max_leaves=8192)
    assert loose.met_leaves == len(loose.leaves)
    assert tight.met_leaves == len(tight.leaves)
    assert len(tight.leaves) > len(loose.leaves)
    width_sum = lambda r: math.fsum(  # noqa: E731
        leaf.bounds[0][1] - leaf.bounds[0][0] for leaf in r.leaves
    )
    assert width_sum(tight) <= width_sum(loose) + 1e-9


def test_jacobi_constant_end_to_end_smoke():
    """平面 Jacobi 常数端到端：叶面积守恒、采样真值逐叶包含、重展开可求值。"""
    da_init_py(6, 2)
    mu = Datum.DE421.mu

    def jacobi(xs):
        x, y = xs
        r1 = ((x + mu) * (x + mu) + y * y).sqrt()
        r2 = ((x - (1 - mu)) * (x - (1 - mu)) + y * y).sqrt()
        return [2.0 * (0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2)]

    def reference(px, py):
        r1 = math.sqrt((px + mu) ** 2 + py**2)
        r2 = math.sqrt((px - (1 - mu)) ** 2 + py**2)
        return px * px + py * py + 2.0 * (1 - mu) / r1 + 2.0 * mu / r2

    result = da_ads_split(jacobi, [(0.80, 0.90), (-0.05, 0.05)], [5e-3])
    assert result.met_leaves == len(result.leaves)
    assert len(result.leaves) >= 2
    assert result.splits_per_var[0] >= 1 and result.splits_per_var[1] >= 1
    area = math.fsum(4.0 * leaf.half_width[0] * leaf.half_width[1] for leaf in result.leaves)
    assert math.isclose(area, 0.01, rel_tol=1e-9)

    rng = np.random.default_rng(20260101)
    points = rng.uniform([0.80, -0.05], [0.90, 0.05], size=(50, 2))
    for px, py in points:
        owners = [
            leaf
            for leaf in result.leaves
            if abs(px - leaf.center[0]) <= leaf.half_width[0] + 1e-12
            and abs(py - leaf.center[1]) <= leaf.half_width[1] + 1e-12
        ]
        assert len(owners) == 1
        leaf = owners[0]
        lo, hi = leaf.bounds[0]
        value = reference(px, py)
        assert lo - 1e-9 <= value <= hi + 1e-9
        unit = [
            (px - leaf.center[0]) / leaf.half_width[0],
            (py - leaf.center[1]) / leaf.half_width[1],
        ]
        assert abs(leaf.values[0].eval(unit) - value) <= (hi - lo) + 1e-9


def test_validation_errors():
    """参数预校验：六类非法输入逐项 ValueError，信息含具体原因关键词。"""
    da_init_py(4, 1)
    func = lambda xs: [xs[0].sin()]  # noqa: E731

    with pytest.raises(ValueError, match="domain"):
        da_ads_split(func, [], [1e-6])
    with pytest.raises(ValueError, match="区间"):
        da_ads_split(func, [(1.0, 0.0)], [1e-6])
    with pytest.raises(ValueError, match="容差"):
        da_ads_split(func, [(0.0, 1.0)], [0.0])
    with pytest.raises(ValueError, match="tolerance_kind"):
        da_ads_split(func, [(0.0, 1.0)], [1e-6], tolerance_kind="rel")
    with pytest.raises(ValueError, match="max_leaves"):
        da_ads_split(func, [(0.0, 1.0)], [1e-6], max_leaves=0)
    with pytest.raises(ValueError, match="targets"):
        da_ads_split(func, [(0.0, 1.0)], [1e-6], targets=[0, 1])

    # 空列表等价 None：按全部分量对齐，由上游按实际输出数校验。
    empty_targets = da_ads_split(func, [(0.0, 1.0)], [1e-2], targets=[])
    assert empty_targets.met_leaves == len(empty_targets.leaves) >= 1


def test_callback_errors():
    """回调异常：Python 异常原样穿透；空返回经 panic 边界映射 RuntimeError 650。"""
    da_init_py(4, 1)

    def boom(xs):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        da_ads_split(boom, [(0.0, 1.0)], [1e-3], max_leaves=8)

    with pytest.raises(RuntimeError, match=r"DA 运算失败.*650"):
        da_ads_split(lambda xs: [], [(0.0, 1.0)], [1e-6])
