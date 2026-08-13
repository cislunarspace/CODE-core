"""``propagate_cr3bp_stm_py`` 释放 GIL 回归测试（#313）。

CR3BP 修正段（``design_dro`` / ``design_halo`` / ``design_lissajous`` 共用
``_propagate_with_stm``）切到 Rust ``propagate_cr3bp_stm_py`` 后，积分循环仍持
GIL，下游 worker 线程跑 ``design_orbit`` 时主线程（GUI / QTimer tick）被冻结
66s+。本测试守住"积分循环包进 ``py.allow_threads``"：主线程积分期间，另一个
Python 线程应能持续推进（心跳打点）。

判别原理同 ``integrators/bindings/test_propagate_compiled.py`` 中的 GIL 测试（#318）：心跳线程循环
``ticks += 1; time.sleep(0.005)``。``time.sleep`` 释 GIL，醒来执行 ``ticks += 1``
必须重新获取 GIL——

- 主循环已用 ``py.allow_threads`` 释 GIL：心跳 reacquire 立即成功，按 ~5 ms 节奏
  持续打点，dt 秒内 ~dt/0.005 次。
- 主循环持 GIL（回归）：心跳 reacquire 阻塞到积分返回，期间打点 ≈ 0。

两者相差 1~2 个数量级，阈值取 3，判别稳健。传播段太短（dt < 0.10 s）信号不足
则 skip。

CR3BP STM 路径是无量纲纯数学（不依赖 SPICE 内核），故本测试直连 Rust 绑定，
不参与 ``pytest.mark.spice`` 门控。用 L4（三角平动点，地月 μ < Routh 临界故线性
稳定）邻近初值，``max_step`` 钳位强制 ~10⁵ 步，保证积分耗时稳超判别下限。
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.data.constants import Datum
from e2m2e.integrators import propagate_cr3bp_stm_py

pytestmark = pytest.mark.integrator


if propagate_cr3bp_stm_py is None:
    pytest.skip("propagate_cr3bp_stm_py 需要 Rust 扩展构建", allow_module_level=True)

# 地月质量参数（无量纲）：DE421 基准（Datum.DE421）。
MU_EARTH_MOON = Datum.DE421.mu


def _l4_state() -> list[float]:
    """L4 邻近初值（CR3BP 旋转系，无量纲）。

    L4 = (1/2 - μ, √3/2, 0)；x 方向偏置 0.01 激发有界天平动（线性稳定，
    长时间不发散），保证 100 无量纲时间积分数值良态。
    """
    return [0.5 - MU_EARTH_MOON + 0.01, np.sqrt(3.0) / 2.0, 0.0, 0.0, 0.0, 0.0]


def test_propagate_cr3bp_stm_releases_gil():
    """主线程 STM 积分期间，心跳线程持续打点 → propagate_cr3bp_stm_py 已释放 GIL。

    回归 #313：积分循环漏包 ``py.allow_threads``，全程持 GIL 使心跳饿死
    （ticks ≈ 0）。修复后释 GIL，ticks 随 dt 线性增长。
    """
    y0 = _l4_state()
    # max_step=1e-3 钳位 → ≥ 1e5 步（rtol 1e-12 下自适应步长远大于 1e-3 被钳），
    # 100 无量纲时间 ≈ 16 个月，L4 天平动有界。release 构建约 0.2~0.4 s。
    t0, tf = 0.0, 100.0
    t_eval = np.linspace(t0, tf, 100)

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
        propagate_cr3bp_stm_py(
            mu=MU_EARTH_MOON,
            t_span=(t0, tf),
            t_eval=[float(t) for t in t_eval],
            initial_state=y0,
            rtol=1e-12,
            atol=1e-12,
            max_step=1e-3,
        )
        dt = time.perf_counter() - t_start
    finally:
        stop[0] = True
        th.join(timeout=2.0)

    ticks_during = ticks[0] - ticks_before

    # 积分段太短 → 信号不足，快机上可能误判，跳过（罕见）。
    if dt < 0.10:
        pytest.skip(f"propagation too short ({dt:.3f}s) to assert GIL release")

    # 释 GIL 时 dt≈0.3 s → ~60 ticks；持 GIL（回归）→ 0~1 ticks。
    # 阈值 3 兼顾：远低于释 GIL 实测（数十次），远高于持 GIL（0~1 次），
    # 对单核 / CI 调度抖动留 ~10× 余量。
    assert ticks_during >= 3, (
        f"heartbeat ticked only {ticks_during} times during {dt:.2f}s STM propagation "
        f"(expected ~{dt / 0.005:.0f} if GIL released); "
        "propagate_cr3bp_stm_py may be holding the GIL — 积分循环漏了 py.allow_threads (#313)"
    )
