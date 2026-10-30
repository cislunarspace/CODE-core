"""DPO 传播时长按圈数预算解析（#841）。

``design_orbit`` 对 DPO 不再取一年的固定时长：主族单圈成员的周期随振幅
从约 1.8 小时变到约 8 天，segmented 修正的规模随圈数立方增长。默认取
两圈，超过上限的请求在参数层确定性拒绝。周期取 20000 km 成员（0.766 TU，
约 3.33 天）。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.design.design_orbit import _resolve_dpo_duration
from e2m2e.data.constants import SECONDS_PER_DAY

pytestmark = pytest.mark.orchestration

#: 20000 km 主族单圈成员的 CR3BP 周期（TU）。
PERIOD_TU = 0.766


def test_default_covers_two_revolutions(earth_moon_system):
    """默认时长等于两圈，不随固定一年展开成上百圈。"""
    t_c = float(earth_moon_system.characteristic_time)
    duration = _resolve_dpo_duration(PERIOD_TU, t_c, None)
    assert duration == pytest.approx(2.0 * PERIOD_TU * t_c)
    assert 6.0 < duration / SECONDS_PER_DAY < 7.0


def test_accepts_request_at_rev_budget(earth_moon_system):
    t_c = float(earth_moon_system.characteristic_time)
    requested = 2.0 * PERIOD_TU * t_c
    assert _resolve_dpo_duration(PERIOD_TU, t_c, requested) == pytest.approx(requested)


def test_rejects_one_year_request(earth_moon_system):
    """一年时长在默认振幅下对应约 110 圈，实测无法完成，参数层拒绝。"""
    t_c = float(earth_moon_system.characteristic_time)
    with pytest.raises(ValueError, match="最多覆盖 2 圈"):
        _resolve_dpo_duration(PERIOD_TU, t_c, 31557600.0)
