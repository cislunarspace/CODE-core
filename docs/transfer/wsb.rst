WSB Low-Energy Solar-Assisted Transfer / WSB 太阳引力辅助低能转移
==================================================================

[English](#wsb-low-energy-solar-assisted-transfer) | [简体中文](#中文)

English
-------

The WSB (Weak Stability Boundary) module implements low-energy indirect lunar
transfers under solar gravity assist, dispatched as ``"WSB"`` by the
:func:`~e2m2e.algorithm.transfer.transfer_orbit` orchestrator with ballistic grid
search + ThreeBodyLambert arrival refinement closing the loop.

Principle
~~~~~~~~~

WSB transfers exploit the Sun-Earth weak-stability-boundary region: leaving the
lunar SOI into a distant solar-perturbed heliocentric arc, solar gravity near
apogee drops Moon-relative speed below capture threshold — ballistic capture at
near-zero Δv (Belbruno & Miller 1993).

Capture uses Belbruno's two-body Kepler energy :math:`H_2`:

.. math::

   H_2 = \frac{|\mathbf{v}|^2}{2} - \frac{\mu_{\text{moon}}}{r}

:math:`H_2 < 0` at perilune means bound to the Moon (ballistic capture).

Two-step solve:

1. **Ballistic grid search**: BCR4BP forward propagation over a solar-phase ×
   departure-phase × TOF grid, screening candidates on perilune altitude and
   :math:`H_2`. Default ``backend="rust"`` — propagation, section detection,
   screening parallelized Rust-side with Rayon; Python implementation runs only
   under explicit ``backend="python"`` (equivalence comparison,
   ``ProcessPoolExecutor``), never auto-fallback.
2. **Arrival refinement**: ThreeBodyLambert shooting refines the best candidate's
   Moon-centered arrival leg.

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.transfer import transfer_orbit
   from e2m2e.algorithm.transfer.wsb import WsbSearchParams

   result = transfer_orbit(
       "WSB",
       wsb_search_params=WsbSearchParams(
           tof_range=(90.0, 150.0),   # TOF range (days)
           max_total_dv=4.0,          # total Δv cap (km/s)
       ),
       departure_state=departure_state,
       target_state=target_state,
   )

   print(f"Status: {result.status.name}")
   print(f"Total Δv: {result.delta_v:.4f} km/s")
   print(f"TOF: {result.details.tof_sec / 86400.0:.2f} days")

Use cases
~~~~~~~~~

- Low-energy Earth-Moon transfers: lunar capture at minimal Δv for fuel-limited missions
- Economical options for landers & orbital inserters
- Low-fuel backup channel for emergency orbit recovery

中文
----

WSB（Weak Stability Boundary）模块实现太阳引力辅助下的低能间接月球转移，
由 :func:`~e2m2e.algorithm.transfer.transfer_orbit` 编排器以
``"WSB"`` 路由，内部经弹道网格搜索 + ThreeBodyLambert 到达段精化闭环。

基本原理
~~~~~~~~

WSB 转移利用日地系统中的弱稳定边界区域：航天器先飞离月球影响球、进入
太阳摄动下的远端日心弧，在远地点附近太阳引力做功使其相对月球的速度
降到捕获阈值之下，从而以接近零的 Δv 被月球弹道捕获（Belbruno & Miller 1993）。

捕获判据采用 Belbruno 约定的二体 Kepler 能量 :math:`H_2`（公式见英文节）：
近月点处 :math:`H_2 < 0` 表示航天器处于月球束缚态（弹道捕获）。

求解流程分两步：弹道网格搜索（默认 Rust 后端 Rayon 并行）+ 到达段精化。
使用方法与结果读取见英文节代码；应用场景包括低能地月转移、经济型着陆器
转移方案与应急低燃料救援通道。
