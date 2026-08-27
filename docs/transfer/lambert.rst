Lambert Solver & Porkchop Scan / Lambert 求解与 porkchop 扫描
==============================================================

[English](#lambert-solver-porkchop-scan) | [简体中文](#中文)

English
-------

The Lambert problem: given endpoint positions and flight time, find the
two-body arc joining them. This page covers the two-body Lambert solver (Rust
Izzo kernel), porkchop scans, and CR3BP three-body shooting seeded by two-body
solutions.

Two-body Lambert solver
~~~~~~~~~~~~~~~~~~~~~~~

:func:`~e2m2e.algorithm.transfer.lambert.solve_lambert` solves one two-body
Lambert problem via Izzo (2015) with a Rust kernel (``e2m2e-propagation`` crate)
exposed through ``e2m2e._integrators``; Python only converts types and wraps
results.

.. code-block:: python

   from e2m2e.algorithm.transfer import solve_lambert

   # Vallado's classic case
   r0 = [5000.0, 10000.0, 2100.0]      # km
   rf = [-14600.0, 2500.0, 7000.0]     # km
   tof = 3600.0                        # s
   mu = 398600.4418                    # Earth GM, km³/s²

   sol = solve_lambert(r0, rf, tof, mu)
   print(f"Departure velocity: {sol.v0}")        # km/s
   print(f"Arrival velocity: {sol.vf}")
   print(f"Iterations: {sol.n_iter}")

``direction`` picks transfer-angle direction: ``"short"`` < π (default),
``"long"`` > π. ``revs`` sets full revolutions — multi-rev returns the right
(low-energy) branch; TOF below that revolution count's minimum raises
``ValueError``.

The returned :class:`~e2m2e.algorithm.transfer.lambert.LambertSolution` carries
``v0``/``vf`` as ``(3,)`` km/s vectors plus ``n_iter`` and ``revs``.

Batch solving
~~~~~~~~~~~~~

:func:`~e2m2e.algorithm.transfer.lambert.solve_lambert_batch` solves an N-geometry
× M-TOF grid in one Rust call:

.. code-block:: python

   import numpy as np
   from e2m2e.algorithm.transfer import solve_lambert_batch

   r0_list = np.tile(r0, (4, 1))       # (N, 3)
   rf_list = np.tile(rf, (4, 1))       # (N, 3)
   tofs = [3600.0, 7200.0, 10800.0]    # (M,)

   out = solve_lambert_batch(r0_list, rf_list, tofs, mu)
   # out shape (N, M, 2, 3): [..., 0, :] = v0; [..., 1, :] = vf

Infeasible combos (zero chord etc.) get NaN at their slots without affecting
others.

Porkchop scan
~~~~~~~~~~~~~

:func:`~e2m2e.algorithm.transfer.porkchop.porkchop` solves Lambert pointwise over a
departure-time × TOF grid producing the two-impulse ΔV mesh behind porkchop
plots. Endpoints come from :class:`~e2m2e.algorithm.transfer.terminal.TerminalCondition`
implementations (e.g., ``OrbitTerminal``, or custom ones); this function doesn't
care how states arise.

At grid point ``(t_dep, tof)``: departure state sampled at ``t_dep``, arrival at
``t_dep + tof``; impulses = transfer velocity minus terminal-orbit velocities.

.. code-block:: python

   import numpy as np
   from e2m2e.algorithm.transfer import porkchop

   t_dep = np.linspace(0.0, 3600.0, 20)        # departure-time grid, s
   tof = np.linspace(900.0, 5 * 3600.0, 30)    # TOF grid, s

   data = porkchop(dep, arr, t_dep, tof, mu=398600.4418, dynamics=None)

   print(data.dv1.shape)     # (20, 30) departure impulses, km/s
   print(data.dv2.shape)     # arrival impulses
   print(data.total.shape)   # totals dv1 + dv2

   ax = data.plot()          # total ΔV contours

``dep``/``arr`` are TerminalCondition implementations; pass ``dynamics=None``
for analytic terminals not requiring propagation.
:class:`~e2m2e.algorithm.transfer.porkchop.PorkchopData` carries grids + time axes;
``plot()`` draws matplotlib contours directly.

Three-body shooting: ThreeBodyLambert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-body Lambert ignores third-body gravity — in cislunar space it's only an
initial guess. :class:`~e2m2e.algorithm.transfer.three_body_lambert.ThreeBodyLambert`
uses the two-body solution to seed damped-Newton shooting under CR3BP dynamics,
correcting departure velocity until the end position hits the target:

1. Both physical endpoints (km, km/s) nondimensionalized via
   ``CR3BP_System.physical_to_dimensionless``
2. Two-body guess from ``solve_lambert`` with μ = 1 on nondimensional geometry
3. Newton iteration: propagate ``with_stm=True``; solve corrections from the end
   STM's Φ_rv block (``Φ[0:3, 3:6]``); halve steps when full steps grow the error;
   converged when terminal position error < 1e-8 (nondimensional)

.. code-block:: python

   from e2m2e.algorithm.transfer import StateTerminal, ThreeBodyLambert
   from e2m2e.data.templates import ConvergenceState

   shooter = ThreeBodyLambert(dynamics)   # system must have scales initialized

   # Terminal states in physical units (km, km/s); arrival position constrained, velocity for the arrival impulse
   sol = shooter.solve(
       StateTerminal(s0, 0.0),
       StateTerminal(s1, tof),
       tof,                    # flight time, s
       guess="lambert",        # guess source; "orbit" uses departure velocity directly
   )

   if sol.status == ConvergenceState.CONVERGED:
       print(f"Departure impulse: {sol.arcs[0].delta_v:.6f} km/s")
       print(f"Arrival impulse: {sol.arrival_delta_v:.6f} km/s")
       print(f"Total impulse: {sol.total_delta_v:.6f} km/s")

Returns a single-arc :class:`~e2m2e.algorithm.transfer.config.TransferSolution`
in physical units; non-convergence reports residual error in ``message``.
Convergence behavior for typical scenarios (two-phase transfers on one periodic
orbit; Lyapunov → Halo rendezvous) in
``tests/algorithm/transfer/test_three_body_lambert.py``.

Multi-impulse transfers & primer-vector check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-impulse solutions are optimal only in specific geometries.
:class:`~e2m2e.algorithm.transfer.multi_impulse.MultiImpulseTransfer` plans n-impulse
transfers between fixed endpoints
(:class:`~e2m2e.algorithm.transfer.terminal.StateTerminal` — position, velocity,
time all pinned): decision variables are mid-course nodes' times & positions
``[t_i, r_i]``, arcs between neighbors closed by Lambert (default two-body
``solve_lambert``; switchable to ``ThreeBodyLambert`` refinement); impulses are
difference of closed legs' entry/exit velocities; scipy SLSQP minimizes total ΔV.

.. code-block:: python

   import numpy as np
   from e2m2e.algorithm.transfer import MultiImpulseTransfer, StateTerminal

   MU_EARTH = 398600.4418  # km³/s²
   R1, R2 = 7000.0, 42164.0  # LEO → GEO, km
   TOF_HOHMANN = np.pi * np.sqrt(((R1 + R2) / 2) ** 3 / MU_EARTH)

   def circular(r, angle=0.0):
       """Circular state of radius r at phase angle angle (counterclockwise)."""
       v = np.sqrt(MU_EARTH / r)
       return np.array(
           [r * np.cos(angle), r * np.sin(angle), 0.0,
            -v * np.sin(angle), v * np.cos(angle), 0.0]
       )

   transfer = MultiImpulseTransfer(
       StateTerminal(circular(R1), 0.0),
       StateTerminal(circular(R2, np.pi), TOF_HOHMANN),
       mu=MU_EARTH,
   )
   sol = transfer.optimize(2)
   print(f"Total impulse: {sol.total_delta_v:.4f} km/s")   # Hohmann baseline 3.7708

``optimize(n_impulses, x0=...)`` decision variables cover only mid-course nodes
(m = n − 2); n=2 has no free variables (single arc closed directly); ``x0`` seeds
mid-node values ``[t_1, r_1, ...]``. After optimization ``transfer.legs`` refreshes
to an alternating sequence of :class:`~e2m2e.algorithm.transfer.multi_impulse.Impulse`
and :class:`~e2m2e.algorithm.transfer.multi_impulse.CoastArc`.

:meth:`~e2m2e.algorithm.transfer.multi_impulse.MultiImpulseTransfer.check_primer_vector`
performs Lawden's primer-vector test: transversality conditions set p(t0) = Δv̂₀,
p(tf) = Δv̂_f; costate carried via STMs yields p(t) curves (Prussing, *Optimal Spacecraft
Trajectories*, ch. 3–4). Necessary optimality: ``|p(t)| ≤ 1`` throughout with
``|p| = 1`` at impulses aligned with them; ``|p| > 1`` inside arcs means an inserted
mid-course impulse can cut total ΔV (Lion & Handelsman 1968). Hohmann satisfies
Lawden's conditions; a two-impulse solution over 0.5× Hohmann time violates it —
the test suggests insertion points, and three-impulse optimization seeded there
lowers total ΔV:

.. code-block:: python

   report = transfer.check_primer_vector(sol, n_samples=300)
   print(f"Lawden satisfied: {report.lawden_satisfied}")
   if not report.lawden_satisfied:
       x0 = np.concatenate(
           [[report.suggested_insertion_time],
            report.suggested_insertion_position]
       )
       sol3 = transfer.optimize(3, x0=x0)   # three impulses beat two

Full cases in ``tests/algorithm/transfer/test_multi_impulse.py``.

中文
----

Lambert 问题给定两端位置与飞行时间，求连接两点的二体轨道。本页介绍三部分：
二体 Lambert 求解器（Rust Izzo 内核）、porkchop 扫描，以及以二体解为初猜的
CR3BP 三体打靶。

**二体 Lambert 求解器**：Izzo (2015) 算法，Rust 内核，Python 只做类型转换与结果封装。
``direction="short"/"long"`` 选择转移角方向；``revs`` 指定多圈数（返回右分支低能解）。
:solve_lambert_batch 对 N 组几何 × M 个飞行时间的网格批量求解，一次调用进 Rust；
无解组合填 NaN 不影响其余。

**porkchop 扫描**：出发时间 × 飞行时间网格逐点解 Lambert，得双脉冲 ΔV 网格。
终端经 ``TerminalCondition`` 接口提取状态；解析终端可传 ``dynamics=None`` 。
返回 ``PorkchopData`` 含三个网格与时间轴，``plot()`` 直接画等值线。

``shooter.solve(...)`` 返回单弧 ``TransferSolution`` ，物理单位；未收敛时在
``message`` 说明残余误差。典型场景收敛行为见
``tests/algorithm/transfer/test_three_body_lambert.py`` 。

多脉冲转移与主矢量检验
~~~~~~~~~~~~~~~~~~~~~~~

双脉冲解只在特定几何下最优。
:class:`~e2m2e.algorithm.transfer.multi_impulse.MultiImpulseTransfer` 在固定端点
（``StateTerminal`` ，位置、速度、时刻均固定）之间规划 n 脉冲转移：决策变量为各中途脉冲节点的时刻与位置，相邻节点间的弧段由 Lambert 封闭（默认二体，可切 ThreeBodyLambert 打靶精修），scipy SLSQP 最小化总 ΔV。LEO→GEO 示例（霍曼基准 3.7708 km/s）、
决策变量口径与 ``optimize(n, x0=...)`` 用法见英文节代码。

主矢量检验：由端点横截条件定 p(t0)、p(tf)，协态经 STM 携载得 p(t) 曲线；最优性必要条件是全程 ``|p(t)| ≤ 1`` 且脉冲点 ``|p| = 1`` 共线；弧内 ``|p| > 1`` 时插入中途脉冲可降总 ΔV（Lion & Handelsman 1968）。霍曼转移满足 Lawden 条件；同一端点飞行时间取 0.5 倍时弧内 ``|p| > 1`` ，检验给出插入建议，三脉冲优化随之降低总 ΔV。

完整算例见 ``tests/algorithm/transfer/test_multi_impulse.py`` 。

.. automodule:: e2m2e.algorithm.transfer.lambert
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.transfer.porkchop
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.transfer.three_body_lambert
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.transfer.multi_impulse
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

参考模块
~~~~~~~~~

- ``algorithm.transfer.lambert`` / ``porkchop`` / ``three_body_lambert`` / ``multi_impulse`` （autodoc 见下方 API 节）
