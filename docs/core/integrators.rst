Integrator Families & Configuration / 积分器族选择与配置
=========================================================

[English](#english) | [简体中文](#中文)

English
-------

e2m2e's integrators come in three families — adaptive single-step Runge-Kutta,
fixed-step multistep Adams, fixed-step second-order Cowell — for first-order
:math:`\dot{y}=f(t,y)` systems or direct second-order :math:`\ddot{x}=a(t,x)`.
The Rust side is a workspace of six crates: ``e2m2e-propagation`` (pure-math
integrators: Butcher tables, RK/ABM/Cowell, solve_ivp), ``e2m2e-forces``
(N-body & STM variational equations, compiled force models), ``e2m2e-spice``
(CSPICE FFI; embeds CSPICE when spice feature is on), ``e2m2e-integrators``
(PyO3 bindings; maturin's sole packaging target producing the
``e2m2e._integrators`` extension), plus the level-set reachability crate
``e2m2e-levelset`` (ToolboxLS port) and HJB dynamics crate
``e2m2e-hjb-dynamics``. Python does type conversion and init helpers only.

Overview
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Family
     - Methods
     - Order
     - Steps
     - Equations
     - Typical use
   * - RK
     - PD45 / PD78 / RK89
     - 5 / 8 / 9
     - adaptive
     - first-order :math:`\dot{y}=f(t,y)`
     - general propagation; accuracy baselines
   * - Adams (multistep)
     - ABM
     - 4
     - fixed
     - first-order :math:`\dot{y}=f(t,y)`
     - smooth RHS, long arcs, known steps
   * - Cowell (double integration)
     - Störmer-Cowell
     - 8
     - fixed
     - second-order :math:`\ddot{x}=a(t,x)`
     - gravity-only orbits; avoids state doubling

Runge-Kutta family
~~~~~~~~~~~~~~~~~~

Three embedded explicit RK methods share one ``rk_step`` entry, differing only in
Butcher tables; each step computes high- and low-order solutions whose difference
estimates local error for adaptive stepping.

.. list-table::
   :header-rows: 1

   * - Enum
     - Full name
     - Main/embedded order
     - Stages
     - Source
   * - ``RkMethod.PD45``
     - Prince-Dormand 5(4)
     - 5 / 4
     - 7
     - Dormand & Prince 1980
   * - ``RkMethod.PD78``
     - Prince-Dormand 8(7)13M
     - 8 / 7
     - 13
     - Hairer-Wanner / GMAT
   * - ``RkMethod.RK89``
     - Verner 9(8)
     - 9 / 8
     - 16
     - GMAT R2026a

PD45: low-order, low-overhead default. PD78/RK89 take fewer steps at equal
tolerance (>5× step savings typical) — suited to long arcs / cross-checks.

Key parameters:

- ``tol``: relative tolerance; acceptance threshold scales as ``tol * max(1, ||y||)``
  for consistent behavior across nondimensional and physical units.
- ``h0``: initial trial step; controller converges within a few steps.
- ``result.error``: L2 difference of high vs embedded solutions (local error estimate).
- ``result.h_next``: suggested next step ``h * clamp(0.9·(tol/error)^(1/(p+1)), 0.1, 5)``.
- ``state_error_dim``: restrict step-error accounting to first N dims
  (e.g., 6 during STM-augmented propagation so 36 STM entries don't dominate).

ABM (fixed-step multistep)
~~~~~~~~~~~~~~~~~~~~~~~~~~

ABM = 4-step 4th-order PECE predictor-corrector; two RHS evaluations per step but
needs four derivative samples of history. Fixed-step (history tied to h — restart
via ``initialize_abm_history`` on change); startup by RK89;
low per-step function-evaluation cost on long arcs.

Cowell (Störmer-Cowell)
~~~~~~~~~~~~~~~~~~~~~~~

Directly integrates :math:`\ddot{x} = a(t, x)`: half the state dimensions, 8th
order acting directly on position. Position-only output (``cowell_step`` returns
``x_new``); acceleration-only RHS (no velocity-dependent forces); needs 8
acceleration + 2 position history samples; startup default ``n_startup=7`` ≥ 7.

Decision tree
~~~~~~~~~~~~~

.. code-block:: text

   RHS contains velocity terms (drag, thrust direction)?
   ├── yes → first-order methods only (RK or ABM)
   │        adaptive/unknown steps?
   │        ├── yes → RK (PD45/PD78/RK89)
   │        └── no → ABM (fixed step, low call cost)
   └── no → gravity-only
            fewest states / highest position accuracy?
            ├── yes → Cowell (8th-order double integration)
            └── no → RK or ABM

Accuracy & efficiency
~~~~~~~~~~~~~~~~~~~~~

Normalized LEO+J2 benchmark (1-day arc, tol 1e-13): PD45 ≈ 5000 steps × 7 evals;
PD78 ≈ 800 × 13; RK89 ≈ 800 × 16; ABM(h=0.002) ≈ 500 steps at 2 evals. Higher
orders cut total evaluations ~3×.

中文
----

e2m2e 的积分器分三族：自适应单步 Runge-Kutta（RK）、固定步长多步 Adams、固定步长二阶 Cowell。Rust 侧 workspace 含六个 crate：``e2m2e-propagation`` （纯数学积分器）、``e2m2e-forces`` （N 体与 STM 变分方程、编译型力模型）、``e2m2e-spice`` （CSPICE FFI）、``e2m2e-integrators`` （PyO3 绑定，唯一打包目标），以及水平集求解器 crate ``e2m2e-levelset`` 与 HJB 动力学 crate ``e2m2e-hjb-dynamics`` 。Python 层仅做类型转换与初始化辅助。

积分器概览
-----------

.. list-table::
   :header-rows: 1

   * - 族
     - 方法
     - 阶数
     - 步长
     - 适用方程
     - 典型场景
   * - RK（Runge-Kutta）
     - PD45 / PD78 / RK89
     - 5 / 8 / 9
     - 自适应
     - 一阶化 :math:`\dot{y}=f(t,y)`
     - 通用传播、高精度比较基准
   * - Adams（多步）
     - ABM（Adams-Bashforth-Moulton）
     - 4
     - 固定
     - 一阶化 :math:`\dot{y}=f(t,y)`
     - 光滑右端、长弧段、步长已知
   * - Cowell（双积分）
     - Störmer-Cowell
     - 8
     - 固定
     - 二阶 :math:`\ddot{x}=a(t,x)`
     - 纯引力轨道、避免一阶化状态膨胀

Runge-Kutta 族（自适应单步）
-----------------------------

三个嵌套式显式 RK 共享 ``rk_step`` 入口；每步同时算高低阶解作局部误差估计驱动自适应。

.. list-table::
   :header-rows: 1

   * - 枚举值
     - 全称
     - 主阶 / 嵌入阶
     - 级数
     - 来源
   * - ``RkMethod.PD45``
     - Prince-Dormand 5(4)
     - 5 / 4
     - 7
     - Dormand & Prince 1980
   * - ``RkMethod.PD78``
     - Prince-Dormand 8(7)13M
     - 8 / 7
     - 13
     - Hairer-Wanner / GMAT
   * - ``RkMethod.RK89``
     - Verner 9(8)
     - 9 / 8
     - 16
     - GMAT R2026a

基本用法：

.. code-block:: python

   import numpy as np
   from e2m2e.integrators import rk_step, RkMethod

   def rhs(t, y):
       return np.array([y[1], -y[0]])   # 谐振子

   y0 = np.array([1.0, 0.0])
   result = rk_step(RkMethod.PD78, 0.0, y0.copy(), 0.1, 1e-12, rhs)
   # StepResult(y_new=..., error=..., h_next=...)

关键参数：``tol`` 相对容差（接受阈值 ``tol*max(1,||y||)`` ）；``h0`` 初始试探步长；
``result.error`` 局部误差估计；``result.h_next`` 控制器建议步长；
``state_error_dim`` 限制步长误差统计维度（STM 增广传播传 6）。

Adams-Bashforth-Moulton（固定步长多步）
----------------------------------------

ABM 是 4 步 4 阶 PECE 预测-校正器。特点：固定步长（history 绑定 h，改变须重新
``initialize_abm_history`` ）；低函数求值开销；前 4 个样本由 RK89 启动。

.. code-block:: python

   from e2m2e.integrators import MultistepMethod, multistep_step, initialize_abm_history

   t, y, history = initialize_abm_history(0.0, y0, h, rhs, n_stages=3)
   for _ in range(100):
       result = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, rhs, history)
       y = np.asarray(result.y_new, dtype=float)
       t += h
       history = result.history   # 滚动 history

Cowell（Störmer-Cowell 双积分）
--------------------------------

直接积二阶 ODE，状态维度减半、8 阶精度直接作用于位置。
位置-only 输出、加速度-only 右端、固定步长（8 加速度 + 2 位置样本 history，
启动 ``n_startup`` 默认 7 ≥ 7）。

.. code-block:: python

   from e2m2e.integrators import cowell_step, initialize_cowell_history

   def accel(t, x):   # 谐振子: x'' = -x
       return -np.asarray(x, dtype=float)

   t, x, v, history = initialize_cowell_history(0.0, x0, v0, h, accel)
   for _ in range(100):
       result = cowell_step(t, h, 1e-12, accel, history)
       x = np.asarray(result.x_new, dtype=float)
       t += h
       history = result.history

选择决策树
-----------

.. code-block:: text

   右端是否含速度项（阻力、推力方向等）？
   ├── 是 → 只能选一阶化方法（RK 或 ABM）
   │       需要自适应/未知步长？
   │       ├── 是 → RK（PD45/PD78/RK89）
   │       └── 否 → ABM（固定步长、低函数开销）
   └── 否 → 纯引力问题
           追求最少状态维度/最高位置精度？
           ├── 是 → Cowell（8 阶双积分）
           └── 否 → RK 或 ABM（与上同）

精度与效率对比
--------------

归一化 LEO + J2（1 天弧段，容差 1e-13）：PD45 约 5000 步 × 7 次调用；
PD78 约 800 × 13；RK89 约 800 × 16；ABM(h=0.002) 约 500 步 × 2 次。
高阶方法总调用数降低约 3 倍。
