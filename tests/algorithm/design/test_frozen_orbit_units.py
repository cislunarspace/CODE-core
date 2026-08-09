"""冻结轨道设计纯函数单元测试（L1，无 SPICE 依赖）。"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.orchestration

MU_MOON = 4902.799967088639


def test_oe_cart_roundtrip():
    """给定一组根数，转笛卡尔再转回，根数应一致（数值精度内）。"""
    from e2m2e.algorithm.design.frozen_orbit import _cart2oe, _oe2cart

    cases = [
        (3000, 0.354, 75.0, 0.0, 270.0, 0.0),
        (5000, 0.612, 75.0, 30.0, 270.0, 45.0),
        (15000, 0.871, 75.0, 90.0, 90.0, 180.0),
    ]
    for a, e, i, raan, aop, nu in cases:
        state = _oe2cart(a, e, i, raan, aop, nu, MU_MOON)
        oe = _cart2oe(state, MU_MOON)
        assert abs(oe["a"] - a) < 1e-6, f"a={a}: got {oe['a']}"
        assert abs(oe["e"] - e) < 1e-10, f"e={e}: got {oe['e']}"
        assert abs(oe["i"] - i) < 1e-8, f"i={i}: got {oe['i']}"
        assert abs(oe["rp"] - a * (1 - e)) < 1e-6


def test_perilune_constraint():
    """rp = a(1-e) 应等于 R_MOON + perilune_height。"""
    from e2m2e.algorithm.design.frozen_orbit import R_MOON, _cart2oe, _oe2cart

    rp_target = R_MOON + 200.0  # 1938 km
    for a in [3000, 5000, 15000]:
        e = 1.0 - rp_target / a
        state = _oe2cart(a, e, 75.0, 0.0, 270.0, 0.0, MU_MOON)
        oe = _cart2oe(state, MU_MOON)
        assert abs(oe["rp"] - rp_target) < 1e-6, f"a={a}: rp={oe['rp']}"


def test_arg_of_pericenter_south_pole():
    """ω=270° 时近月点位置向量的 z 分量应为负（南极方向）。"""
    from e2m2e.algorithm.design.frozen_orbit import _oe2cart

    state = _oe2cart(5000, 0.612, 75.0, 0.0, 270.0, 0.0, MU_MOON)
    assert state[2] < 0, f"ω=270° 近月点 z={state[2]:.1f} km，应在南极方向（z<0）"


def test_compute_drift_wraparound():
    """ω 从 269° 到 271° 经 wraparound，漂移应为 +2°（非 -358°）。"""
    from e2m2e.algorithm.design.frozen_orbit import _compute_drift

    elements = {
        "a": np.full(10, 3000.0),
        "e": np.linspace(0.354, 0.335, 10),
        "i": np.full(10, 75.0),
        "raan": np.zeros(10),
        "aop": np.append(np.full(5, 269.0), np.full(5, 271.0)),
        "rp": np.linspace(1938, 1999, 10),
    }
    drift = _compute_drift(elements, output_step_sec=3600.0)
    assert abs(drift["drift_aop_deg"] - 2.0) < 0.01
    assert drift["drift_e"] < 0
    assert drift["drift_rp_km"] > 0


def test_compute_drift_secular_rate():
    """secular ω 漂移率：线性拟合 + 单位转换。"""
    from e2m2e.algorithm.design.frozen_orbit import _compute_drift

    n = 100
    # ω 以 0.1°/点 线性增长
    aop = np.arange(n, dtype=float) * 0.1
    elements = {
        "a": np.full(n, 3000.0),
        "e": np.full(n, 0.354),
        "i": np.full(n, 75.0),
        "raan": np.zeros(n),
        "aop": aop,
        "rp": np.full(n, 1938.0),
    }
    drift = _compute_drift(elements, output_step_sec=3600.0)
    # 0.1°/点 × (365.25*86400 / 3600) 点/年 ≈ 0.1 × 8766 = 876.6°/年
    expected_rate = 0.1 * 365.25 * 86400 / 3600
    assert abs(drift["secular_aop_rate_deg_per_year"] - expected_rate) < 0.1


def test_compute_drift_no_time_info():
    """无时间信息时 secular 率为 None。"""
    from e2m2e.algorithm.design.frozen_orbit import _compute_drift

    elements = {
        "a": np.full(5, 3000.0),
        "e": np.full(5, 0.354),
        "i": np.full(5, 75.0),
        "raan": np.zeros(5),
        "aop": np.array([270.0, 271.0, 272.0, 273.0, 274.0]),
        "rp": np.full(5, 1938.0),
    }
    drift = _compute_drift(elements)
    assert drift["secular_aop_rate_deg_per_year"] is None
