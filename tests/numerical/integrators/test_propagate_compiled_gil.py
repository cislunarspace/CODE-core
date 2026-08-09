"""``propagate_compiled`` 释放 GIL 回归测试（#318）。

长期预报段必须释放 GIL：主线程跑 ``propagate_compiled`` 时，另一个 Python
线程应能继续推进（心跳打点）。修复前主积分循环全程持 GIL，心跳线程
reacquire GIL 阻塞、饿死，对应 transfer-orbit-design GUI 设计 DRO 时窗口
冻死（cProfile 显示 ``propagate_compiled`` cumtime 占 96%、~225 s 连续持 GIL）。

判别原理：心跳线程循环 ``ticks += 1; time.sleep(0.005)``。``time.sleep``
释放 GIL，醒来后要执行 ``ticks += 1`` 必须重新获取 GIL——

- 主线程已用 ``py.allow_threads`` 释放 GIL：心跳线程 reacquire 立即成功，
  按 ~5 ms 节奏持续打点，dt 秒内打点 ~dt/0.005 次。
- 主线程持 GIL（回归）：心跳线程 reacquire 阻塞到主线程返回，期间打点 ≈ 0。

两者相差 1~2 个数量级，阈值取 3（远低于释 GIL 的 ~48 次/0.24 s，远高于
持 GIL 的 0~1 次），判别稳健。传播段太短（dt < 0.15 s）信号不足则 skip，
不在快机上误判。

用纯二体 ``point_mass`` 力模型（无 SPICE 内核依赖），经 Rust
``propagate_compiled`` 直连快路径触发主积分循环。
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.integrators import RkMethod, propagate_compiled

pytestmark = pytest.mark.integrator


if propagate_compiled is None:
    pytest.skip("propagate_compiled 需要 spice-feature 构建", allow_module_level=True)

# 地球引力参数 (km³/s²)
EARTH_MU = 398600.4418


def _leo_y0() -> list[float]:
    """6778 km 圆轨道初始状态 (km, km/s)。"""
    r = 6778.0
    v = float(np.sqrt(EARTH_MU / r))
    return [r, 0.0, 0.0, 0.0, v, 0.0]


def test_propagate_compiled_releases_gil():
    """主线程长期预报期间，心跳线程持续打点 → propagate_compiled 已释放 GIL。

    回归 #318：主积分循环漏包 ``py.allow_threads``，全程持 GIL 使心跳饿死
    （ticks ≈ 0）。修复后释 GIL，ticks 随 dt 线性增长。
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
