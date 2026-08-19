"""``propagate_compiled`` 绑定契约与回归测试。

合并原 test_propagate_compiled_gil.py（#318：主积分循环释 GIL）、
test_propagate_compiled_teval.py（c3685e7 同源 bug：``t_eval[0] > t0`` 首点错置）
与 test_h_init_step_cap.py（#279：h_init 步长上限缺失导致稀疏 t_eval 漂移），
并补充绑定层参数校验（y0 长度、h_init、空 t_eval）。

GIL 测试用纯二体 ``point_mass`` 力模型（无 SPICE 内核依赖），经 Rust
``propagate_compiled`` 直连快路径；teval / h_init 测试经 ``ForceModel.propagate``
走同一路径（挂 spice 门控，见各测试）。
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.point_mass_gravity import PointMassGravity
from e2m2e.data.constants import Datum
from e2m2e.integrators import RkMethod, propagate_compiled

pytestmark = pytest.mark.integrator


if propagate_compiled is None:
    pytest.skip("propagate_compiled 需要 spice-feature 构建", allow_module_level=True)

# 地球引力参数：WGS-84 基准（Datum.WGS84）。
EARTH_MU = Datum.WGS84.earth_gm


def _leo_y0() -> list[float]:
    """6778 km 圆轨道初始状态 (km, km/s)。"""
    r = 6778.0
    v = float(np.sqrt(EARTH_MU / r))
    return [r, 0.0, 0.0, 0.0, v, 0.0]


class _FakeSystem:
    """最小 System 桩，仅供 PointMassGravity.to_rust_spec 解析 mu。

    无 ``origin`` 属性，``_propagate_via_rust`` 会回退默认 "EARTH"。
    """

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.data.templates.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.data.templates.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return EARTH_MU


def _build_force_model() -> ForceModel:
    system = _FakeSystem()
    pm = PointMassGravity("EARTH", mu=EARTH_MU)
    fm = ForceModel(system, forces=[pm])
    fm.max_step = 3600.0
    return fm


# ---------------------------------------------------------------------------
# 绑定层参数校验
# ---------------------------------------------------------------------------


def test_propagate_compiled_rejects_invalid_inputs():
    """绑定层校验：y0 长度非 6、h_init 非正、t_eval 为空均抛 ValueError。"""
    y0 = _leo_y0()
    t_eval = [0.0, 3600.0]
    forces = [("point_mass", EARTH_MU)]

    with pytest.raises(ValueError):
        propagate_compiled(RkMethod.PD45, 0.0, y0[:5], 3600.0, 1e-12, t_eval, "EARTH", forces, 1000)

    with pytest.raises(ValueError):
        propagate_compiled(RkMethod.PD45, 0.0, y0, 0.0, 1e-12, t_eval, "EARTH", forces, 1000)

    with pytest.raises(ValueError):
        propagate_compiled(RkMethod.PD45, 0.0, y0, 3600.0, 1e-12, [], "EARTH", forces, 1000)


# ---------------------------------------------------------------------------
# 回归 #318：主积分循环释放 GIL
# ---------------------------------------------------------------------------


def test_propagate_compiled_releases_gil():
    """主线程长期预报期间，心跳线程持续打点 → propagate_compiled 已释放 GIL。

    回归 #318：主积分循环漏包 ``py.allow_threads``，全程持 GIL 使心跳饿死
    （ticks ≈ 0）。修复后释 GIL，ticks 随 dt 线性增长。

    判别原理：心跳线程循环 ``ticks += 1; time.sleep(0.005)``。``time.sleep``
    释放 GIL，醒来后要执行 ``ticks += 1`` 必须重新获取 GIL——

    - 主线程已用 ``py.allow_threads`` 释放 GIL：心跳线程 reacquire 立即成功，
      按 ~5 ms 节奏持续打点，dt 秒内打点 ~dt/0.005 次。
    - 主线程持 GIL（回归）：心跳线程 reacquire 阻塞到主线程返回，期间打点 ≈ 0。

    两者相差 1~2 个数量级，阈值取 3（远低于释 GIL 的 ~48 次/0.24 s，远高于
    持 GIL 的 0~1 次），判别稳健。传播段太短（dt < 0.15 s）信号不足则 skip，
    不在快机上误判。
    """
    y0 = _leo_y0()
    # 8 天两体传播 ≈ 44 万步、~0.24 s（release 构建），稳超 0.10 s 判别下限。
    days = 8.0
    t0, tf = 0.0, days * 86400.0
    t_eval = np.linspace(t0, tf, 50)

    ticks = [0]
    stop = [False]

    def heartbeat() -> None:
        while not stop[0]:
            ticks[0] += 1
            time.sleep(0.005)

    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()
    try:
        time.sleep(0.05)  # 让心跳起步，确保它已在 sleep→reacquire 循环中
        ticks_before = ticks[0]
        t_start = time.perf_counter()
        propagate_compiled(
            RkMethod.PD45,
            t0,
            y0,
            3600.0,
            1e-12,
            [float(x) for x in t_eval],
            "EARTH",
            [("point_mass", EARTH_MU)],
            2_000_000,
        )
        dt = time.perf_counter() - t_start
    finally:
        stop[0] = True
        th.join(timeout=2.0)

    ticks_during = ticks[0] - ticks_before

    # 传播段太短 → 信号不足，快机上可能误判，跳过（罕见）。
    if dt < 0.10:
        pytest.skip(f"propagation too short ({dt:.3f}s) to assert GIL release")

    # 释 GIL 时 dt≈0.24 s → ~48 ticks；持 GIL（回归）→ 0~1 ticks。
    # 阈值 3 兼顾：远低于释 GIL 实测（数十次），远高于持 GIL（0~1 次），
    # 且对单核 / CI 调度抖动留有 ~10× 余量。
    assert ticks_during >= 3, (
        f"heartbeat ticked only {ticks_during} times during {dt:.2f}s propagation "
        f"(expected ~{dt / 0.005:.0f} if GIL released); "
        "propagate_compiled may be holding the GIL — 主循环漏了 py.allow_threads (#318)"
    )


# ---------------------------------------------------------------------------
# 回归 c3685e7 同源：t_eval[0] > t0 首点错置
# ---------------------------------------------------------------------------


@pytest.mark.spice
def test_propagate_compiled_teval0_greater_than_t0():
    """``t_eval[0] > t0`` 时首个输出点跟随 ``t_eval[0]``，而非错置为初值。

    回归 bug：硬编码 ``vec![t0] + eval_idx=1`` 假设 ``t_eval[0]==t0``，
    ``t_eval[0]>t0`` 时首个输出点状态错置为初值（位置停滞），后续点错位。
    segmented 分段打靶逐段积分时，每段 ``seg_t0``（patch point 时刻，非整数
    小时）与 ``t_eval_seg[0]``（et_grid 整数小时点）不严格相等，即此场景。
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


@pytest.mark.spice
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


# ---------------------------------------------------------------------------
# 回归 #279：h_init 步长上限
# ---------------------------------------------------------------------------


@pytest.mark.spice
def test_sparse_vs_dense_t_eval_consistency():
    """稀疏 vs 密集 t_eval 的 LEO 传播终态差 < 1 km。

    回归目标：h_init 步长上限缺失时，稀疏 t_eval 下自适应步长失控，
    导致终态差 22 万 km（实测，30 天）。加 h_init 上限后两者应一致。
    注：tol=1e-12 下有效步长 ~1.6s，1 天需 ~55k 步，受默认 max_steps
    限制，此处用 1 天验证回归（差值与持续时间成正比，1 天已足够暴露 bug）。
    """
    system = _FakeSystem()
    pm = PointMassGravity("EARTH", mu=EARTH_MU)
    fm = ForceModel(system, forces=[pm])
    # h_init 需足够大让自适应控制器自由选择步长
    fm.max_step = 3600.0

    # LEO 圆轨道初始状态 (km, km/s)
    r = 6778.0
    v = np.sqrt(EARTH_MU / r)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    days = 1.0
    t0 = 0.0
    tf = days * 86400.0  # 秒

    # 稀疏：只有起点和终点
    t_eval_sparse = np.array([t0, tf])
    # 密集：31 个等距点
    t_eval_dense = np.linspace(t0, tf, 31)

    result_sparse = fm.propagate(y0, (t0, tf), t_eval=t_eval_sparse)
    result_dense = fm.propagate(y0, (t0, tf), t_eval=t_eval_dense)

    final_sparse = np.asarray(result_sparse["states"][-1])
    final_dense = np.asarray(result_dense["states"][-1])

    # 终态位置差 < 1 km（回归阈值：无 h_init 上限时差 22 万 km/30天，
    # 1 天约为 1/30，仍远超 1 km）
    pos_diff_km = np.linalg.norm(final_sparse[:3] - final_dense[:3])
    assert pos_diff_km < 1.0, (
        f"Sparse vs dense t_eval position drift: {pos_diff_km:.1f} km > 1 km; "
        "h_init step cap may be missing"
    )
