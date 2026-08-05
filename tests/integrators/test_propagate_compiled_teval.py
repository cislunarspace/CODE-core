"""白盒回归测试：propagate_compiled 的 ``t_eval[0] > t0`` 行为。

回归目标（commit c3685e7 的同源 bug，补齐 c3685e7 未覆盖的传播函数）：
segmented 分段打靶逐段积分时，每段 ``seg_t0``（patch point 时刻，非整数
小时）与 ``t_eval_seg[0]``（et_grid 整数小时点）不严格相等。``propagate_compiled``
及其同族函数（``propagate_compiled_lowthrust``、
``propagate_compiled_lowthrust_sensitivity``、``propagate_compiled_stm``）
原硬编码 ``eval_idx = 1``、把 ``t0`` 当作首个输出点，导致 ``t_eval[0] > t0``
时首个输出点错置为初值、``t_eval[0]`` 处的真实状态被跳过。

本测试用纯二体力模型（地球点质量，无需 SPICE 内核）直接构造
``t_eval[0] > t0`` 场景，经 ``ForceModel.propagate`` 走 Rust
``propagate_compiled`` 快速路径，校验：
- ``time[0]`` 等于 ``t_eval[0]``（而非 ``t0``）
- ``states[0]`` 不等于初值 ``y0``（首点不是错置的初值）

应在 1 秒内跑完。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.point_mass_gravity import PointMassGravity

pytestmark = pytest.mark.spice

# 地球引力参数 (km³/s²)
EARTH_MU = 398600.4418


class _FakeSystem:
    """最小 System 桩，仅供 PointMassGravity.to_rust_spec 解析 mu。

    与 tests/integrators/test_h_init_step_cap.py 的桩一致；无 ``origin``
    属性，``_propagate_via_rust`` 会回退默认 "EARTH"。
    """

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return EARTH_MU


def _build_force_model() -> ForceModel:
    system = _FakeSystem()
    pm = PointMassGravity("EARTH", mu=EARTH_MU)
    fm = ForceModel(system, forces=[pm])
    fm.max_step = 3600.0
    return fm


def test_propagate_compiled_teval0_greater_than_t0():
    """``t_eval[0] > t0`` 时首个输出点跟随 ``t_eval[0]``，而非错置为初值。

    回归 bug：硬编码 ``vec![t0] + eval_idx=1`` 假设 ``t_eval[0]==t0``，
    ``t_eval[0]>t0`` 时首个输出点状态错置为初值（位置停滞），后续点错位。
    """
    fm = _build_force_model()

    # LEO 圆轨道初始状态 (km, km/s)
    y0 = np.array([6678.0, 0.0, 0.0, 0.0, 7.726, 0.0])
    # t0 取非整数小时值（patch point 时刻的常态），t_eval[0] 严格大于 t0
    t0 = 100.5
    tf = 300.0
    t_eval = np.array([200.0, 300.0])

    result = fm.propagate(y0, (t0, tf), t_eval=t_eval)

    times = np.asarray(result["time"])
    states = np.asarray(result["states"])

    # 首个输出时刻应等于 t_eval[0]（200.0），而非 t0（100.5）
    assert times[0] == pytest.approx(200.0, abs=1e-9), (
        f"time[0]={times[0]} 应等于 t_eval[0]=200.0，而非 t0={t0}; "
        "疑似 propagate_compiled 的 eval_idx 初始化回归（t_eval[0]≠t0）"
    )
    # 首点状态不应是错置的初值：LEO 在 ~200 s 内位置明显移动
    pos0 = states[0, :3]
    assert not np.allclose(pos0, y0[:3], atol=1e-6), (
        f"states[0] 位置 {pos0} 等于初值 {y0[:3]}，疑似首点被错置为初值"
    )


def test_propagate_compiled_teval_first_equals_t0():
    """对照基准：``t_eval[0] == t0`` 时首点应记录初值（回归不应破坏此路径）。

    与上一测试同函数、互补：确保动态判定的 ``t_eval[0]==t0`` 分支仍记录初值，
    防止修复过度（漏记初值会导致输出少一个点）。
    """
    fm = _build_force_model()

    y0 = np.array([6678.0, 0.0, 0.0, 0.0, 7.726, 0.0])
    t0 = 100.5
    tf = 300.0
    t_eval = np.array([100.5, 200.0, 300.0])  # t_eval[0] == t0

    result = fm.propagate(y0, (t0, tf), t_eval=t_eval)

    times = np.asarray(result["time"])
    states = np.asarray(result["states"])

    assert times[0] == pytest.approx(t0, abs=1e-9), (
        f"t_eval[0]==t0 时 time[0]={times[0]} 应等于 t0={t0}"
    )
    assert np.allclose(states[0, :3], y0[:3], atol=1e-9), (
        f"t_eval[0]==t0 时 states[0] 应等于初值，得到 {states[0, :3]}"
    )
