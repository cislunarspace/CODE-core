"""Axial 分岔种子备忘的语义测试（#801）。

真实扫描固有 ~1 min（ADR 0037 移出默认套件），这里替换扫描函数只验证
备忘层行为：命中不重算、返回副本、按键分隔、清空后重算。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family import axial_initial_guess as mod
from e2m2e.algorithm.family.axial_initial_guess import (
    clear_axial_bifurcation_cache,
    compute_axial_initial_guess,
)

pytestmark = pytest.mark.orchestration

_MU = 1.215058560962404e-2


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_axial_bifurcation_cache()
    yield
    clear_axial_bifurcation_cache()


@pytest.fixture
def dynamics():
    return CR3BP_Dynamics(CR3BP_System(mu=_MU, primary="Earth", secondary="Moon"))


@pytest.fixture
def scan_calls(monkeypatch):
    """替换真实扫描，记录每次重算的平动点编号。"""
    calls: list[int] = []

    def scan(_dynamics, libration_point):
        calls.append(libration_point)
        return np.array([0.8, 0.0, 0.0, 0.0, 0.3, 0.0]), 2.5 + 0.1 * libration_point

    monkeypatch.setattr(mod, "_scan_axial_bifurcation_seed", scan)
    return calls


def test_memo_skips_recompute_and_returns_copy(dynamics, scan_calls):
    """同键第二次调用不重算，且备忘层返回的是副本而非缓存内数组。

    直接调 ``_find_axial_bifurcation_seed``：公共入口 ``compute_axial_initial_guess``
    自身还会复制一次，经它会掩蔽备忘层的副本语义。
    """
    state1, period1 = mod._find_axial_bifurcation_seed(dynamics, 1)
    state1[0] = -99.0  # 调用方就地改写
    state2, period2 = mod._find_axial_bifurcation_seed(dynamics, 1)

    assert scan_calls == [1]
    assert state2[0] == pytest.approx(0.8)  # 缓存值未被污染
    assert state2 is not state1
    assert period1 == period2 == pytest.approx(2.6)


def test_memo_separates_keys(dynamics, scan_calls):
    """(mu, 平动点) 不同的键各自重算。"""
    compute_axial_initial_guess(dynamics, 1, 0.05)
    compute_axial_initial_guess(dynamics, 2, 0.05)
    compute_axial_initial_guess(dynamics, 1, 0.05)

    assert scan_calls == [1, 2]


def test_clear_forces_recompute(dynamics, scan_calls):
    """清空备忘后重算。"""
    compute_axial_initial_guess(dynamics, 1, 0.05)
    clear_axial_bifurcation_cache()
    compute_axial_initial_guess(dynamics, 1, 0.05)

    assert scan_calls == [1, 1]
