"""CR3BP 多项式流（DA 传播）绑定的行为测试（issue #786、#831）。

覆盖六类事实：返回契约与零偏差求值一致；一阶截断的线性映射与既有 STM
一致；零推力调用与改动前实现的位模式一致；缺省推力与显式零向量逐位
一致；常值推力的名义状态与同推力 f64 RK4 参照一致；前置校验错误路径抛
ValueError。DA 上下文是进程级全局状态，每个用例开头各自 da_init_py
（同时重置线程截断阶），不跨用例持对象。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.integrators import (
    da_init_py,
    propagate_cr3bp_da_py,
    propagate_cr3bp_stm_py,
)

pytestmark = [
    pytest.mark.integrator,
    requires_native_symbols("propagate_cr3bp_da_py", "da_init_py"),
]

# DE421 校准地月质量比与北族 L2 NRHO 折叠种子，出处同
# crates/e2m2e-integrators/src/family_generation/nrho.rs（种子周期在该 μ 下标定）。
MU = 0.012150585350562453
SEED = [1.128103754424342, 0.0, 0.17883236940616654, 0.0, -0.22553424464298827, 0.0]

# 改动前 master 构建上 propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 1, 1e-3)
# 的位级结果（浮点用 float.hex() 记录，含 -0.0 的符号位），用于钉住零推力路径
# 与改动前实现逐位一致。线性系数按 x、y、z、vx、vy、vz 偏差顺序排列。
_FROZEN_TIMES = ("0x1.0000000000000p-1",)

_FROZEN_N_STEPS = 500

_FROZEN_STATES = (
    (
        "0x1.1ac558bf847b6p+0",
        "-0x1.9ea560a7d361dp-4",
        "0x1.22138cb597cd5p-3",
        "-0x1.680ad5a97ea65p-4",
        "-0x1.40225552a80e3p-3",
        "-0x1.2e051f91b84e8p-3",
    ),
)

_FROZEN_LINEAR = (
    (
        "0x1.482e388da331fp+0",
        "-0x1.2a2abe70c8c41p-4",
        "0x1.a52073edbb6ccp-3",
        "0x1.e01750a9fc779p-2",
        "0x1.d0804e4e3e355p-3",
        "0x1.16e6d2764fd81p-5",
    ),
    (
        "-0x1.373a59978cae6p-3",
        "0x1.e4ece5f15083ap-1",
        "-0x1.03e618ddb5186p-3",
        "-0x1.020a53441908fp-2",
        "0x1.a172be6802201p-2",
        "-0x1.6e9b520a3fe5ap-6",
    ),
    (
        "0x1.e5b772b0cb840p-3",
        "-0x1.b4076892a73bdp-5",
        "0x1.0999d163263d8p+0",
        "0x1.4beedd04e62aep-5",
        "-0x1.b0cdb65c4262ap-9",
        "0x1.024b2e94add89p-1",
    ),
    (
        "0x1.0fce8041cbb2bp+0",
        "-0x1.b3edc551f1204p-2",
        "0x1.7deb8079b7021p-1",
        "0x1.a50f8afe66485p-1",
        "0x1.a218dfb1946cfp-1",
        "0x1.85f6a226cc01ep-3",
    ),
    (
        "-0x1.df5e4cf8c0cc9p-1",
        "-0x1.438b2806b7183p-4",
        "-0x1.8735b62f6f80dp-1",
        "-0x1.0668d69e7b021p+0",
        "0x1.dd4366452c4cbp-2",
        "-0x1.6f9c122efd567p-3",
    ),
    (
        "0x1.feb5c4f1a0450p-1",
        "-0x1.4d4a2af900437p-2",
        "0x1.3da40539727d1p-3",
        "0x1.0547f19943bb9p-2",
        "-0x1.a05a92f90d733p-6",
        "0x1.051e0a4f191c1p+0",
    ),
)


def _thrust_rhs(mu: float, state: np.ndarray, thrust: np.ndarray) -> np.ndarray:
    """CR3BP 运动方程右端项，会合系常值推力加速度直接加到加速度三分量。"""
    x, y, z, vx, vy, vz = state
    dx1 = x + mu
    dx2 = x + mu - 1.0
    r1 = np.sqrt(dx1 * dx1 + y * y + z * z)
    r2 = np.sqrt(dx2 * dx2 + y * y + z * z)
    ax = 2.0 * vy + x - (1.0 - mu) * dx1 / r1**3 - mu * dx2 / r2**3 + thrust[0]
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1**3 - mu * y / r2**3 + thrust[1]
    az = -(1.0 - mu) * z / r1**3 - mu * z / r2**3 + thrust[2]
    return np.array([vx, vy, vz, ax, ay, az])


def _rk4_reference(
    mu: float, arc: float, state0: list[float], thrust: list[float], step: float
) -> np.ndarray:
    """同推力 f64 固定步长 RK4 参照，步长与末步对齐逻辑同内核。"""
    y = np.asarray(state0, dtype=float)
    thrust_arr = np.asarray(thrust, dtype=float)
    t = 0.0
    direction = 1.0 if arc > 0.0 else -1.0
    while abs(arc - t) > 1e-12 * (1.0 + abs(arc)):
        h = min(step, abs(arc - t)) * direction
        k1 = _thrust_rhs(mu, y, thrust_arr)
        k2 = _thrust_rhs(mu, y + 0.5 * h * k1, thrust_arr)
        k3 = _thrust_rhs(mu, y + 0.5 * h * k2, thrust_arr)
        k4 = _thrust_rhs(mu, y + h * k3, thrust_arr)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


def test_result_contract_and_zero_deviation():
    """dict 四键齐全；flows 为 6 个 Da；零偏差求值等于名义状态。"""
    da_init_py(2, 6)
    result = propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3)
    assert set(result) == {"times", "states", "flows", "n_steps"}
    assert result["times"] == [0.5]
    assert isinstance(result["n_steps"], int) and result["n_steps"] == 500
    flows = result["flows"][-1]
    assert len(flows) == 6
    for poly, nominal in zip(flows, result["states"][-1], strict=True):
        assert poly.eval([0.0] * 6) == nominal


def test_first_order_matches_stm():
    """一阶截断的线性映射与既有 STM 传播一致（筛选级 1e-8）。"""
    da_init_py(1, 6)
    arc = 0.5
    result = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 1, 1e-3)
    stm = propagate_cr3bp_stm_py(MU, (0.0, arc), [arc], SEED, 1e-12, 1e-12, None, None)
    linear = np.array([poly.linear() for poly in result["flows"][-1]])
    reference = np.array(stm["stm"][-1]).reshape(6, 6)
    rel_err = np.abs(linear - reference) / np.maximum(np.abs(reference), 1.0)
    assert rel_err.max() < 1e-8, f"一阶线性映射对 STM 的最大相对误差 {rel_err.max():.3e} 超过 1e-8"


def test_zero_thrust_matches_prechange_baseline():
    """零推力调用与改动前实现的位模式逐位一致（基准取自 master 构建）。"""
    da_init_py(1, 6)
    result = propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 1, 1e-3)
    assert tuple(t.hex() for t in result["times"]) == _FROZEN_TIMES
    assert result["n_steps"] == _FROZEN_N_STEPS
    assert tuple(tuple(x.hex() for x in state) for state in result["states"]) == _FROZEN_STATES
    linear = tuple(tuple(c.hex() for c in poly.linear()) for poly in result["flows"][-1])
    assert linear == _FROZEN_LINEAR, "零推力路径的线性系数位模式与改动前实现不一致"


def test_thrust_default_equals_explicit_zero():
    """缺省推力与显式零向量逐位一致：时间、名义状态、步数与多项式映射。"""
    da_init_py(2, 6)
    arc = 0.5
    default = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 2, 1e-3)
    explicit = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 2, 1e-3, thrust=[0.0, 0.0, 0.0])
    assert default["times"] == explicit["times"]
    assert default["states"] == explicit["states"]
    assert default["n_steps"] == explicit["n_steps"]
    for flow_default, flow_explicit in zip(default["flows"], explicit["flows"], strict=True):
        for poly_default, poly_explicit in zip(flow_default, flow_explicit, strict=True):
            assert poly_default.cons() == poly_explicit.cons()
            assert poly_default.linear() == poly_explicit.linear()
            assert poly_default.monomials() == poly_explicit.monomials()


def test_constant_thrust_nominal_matches_f64_rk4():
    """常值推力的名义终态与同推力 f64 RK4 参照一致（同时识别分量顺序）。"""
    da_init_py(2, 6)
    arc = 0.5
    thrust = [1.0e-4, -2.0e-4, 5.0e-5]
    result = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 2, 1e-3, thrust=thrust)
    reference = _rk4_reference(MU, arc, SEED, thrust, 1e-3)
    got = np.asarray(result["states"][-1], dtype=float)
    err = np.max(np.abs(got - reference) / np.maximum(np.abs(reference), 1.0))
    assert err < 1e-10, f"常推力名义终态对 f64 RK4 参照的最大相对误差 {err:.3e} 超过 1e-10"


def test_constant_thrust_changes_nominal_and_linear():
    """常值推力改变名义终态与一阶线性映射，状态增量与同推力 f64 RK4 参照一致。"""
    da_init_py(2, 6)
    arc = 0.5
    thrust = [1.0e-4, -2.0e-4, 5.0e-5]
    zero = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 2, 1e-3)
    pushed = propagate_cr3bp_da_py(MU, (0.0, arc), [arc], SEED, 2, 1e-3, thrust=thrust)
    dv = np.asarray(pushed["states"][-1], dtype=float) - np.asarray(zero["states"][-1], dtype=float)
    assert np.abs(dv).max() > 0.0, "常值推力应改变名义终态"
    reference = _rk4_reference(MU, arc, SEED, thrust, 1e-3) - _rk4_reference(
        MU, arc, SEED, [0.0, 0.0, 0.0], 1e-3
    )
    err = np.max(np.abs(dv - reference))
    assert err < 1e-9, f"常推力状态增量对 f64 RK4 参照的最大绝对误差 {err:.3e} 超过 1e-9"
    zero_linear = np.array([poly.linear() for poly in zero["flows"][-1]])
    pushed_linear = np.array([poly.linear() for poly in pushed["flows"][-1]])
    assert np.abs(pushed_linear - zero_linear).max() > 1e-8, "常值推力应改变一阶线性映射"


def test_thrust_validation_errors():
    """thrust 的长度与有限性校验统一抛 ValueError 并提到 thrust。"""
    da_init_py(2, 6)
    with pytest.raises(ValueError, match="thrust"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3, thrust=[0.0, 0.0])
    with pytest.raises(ValueError, match="thrust"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3, thrust=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="thrust"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3, thrust=[float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="thrust"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3, thrust=[float("inf"), 0.0, 0.0])


def test_validation_errors():
    """前置校验错误路径统一抛 ValueError 并给出修复指引。"""
    da_init_py(2, 6)
    with pytest.raises(ValueError, match="initial_state"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED[:5], 2, 1e-3)
    with pytest.raises(ValueError, match="t_eval"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [], SEED, 2, 1e-3)
    with pytest.raises(ValueError, match="截断阶"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 3, 1e-3)
    da_init_py(2, 3)
    with pytest.raises(ValueError, match="变量数"):
        propagate_cr3bp_da_py(MU, (0.0, 0.5), [0.5], SEED, 2, 1e-3)
