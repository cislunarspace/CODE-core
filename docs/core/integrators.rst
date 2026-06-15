积分器族选择与配置
========================

e2m2e 提供三类数值积分器，覆盖从自适应单步到固定步长多步、从一阶化到双积分的完整谱系。所有积分器核心由 Rust 实现（PyO3 绑定），Python 层仅做类型转换与初始化辅助。


积分器概览
----------

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
----------------------------

三个嵌套式（embedded）显式 RK 方法共享同一 ``rk_step`` 入口，仅 Butcher 表不同。每一步同时计算高阶解与低阶解，二者差值作为局部误差估计，驱动自适应步长。

方法选择
^^^^^^^^

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

PD45 是低阶、低开销的默认选择；PD78 和 RK89 在相同容差下步数更少（高阶方法通常节省 5 倍以上步数），适合作为长弧段传播或交叉验证的基准。

基本用法
^^^^^^^^

.. code-block:: python

   import numpy as np
   from e2m2e.integrators import rk_step, RkMethod

   def rhs(t, y):
       """谐振子: y' = [v, -x]"""
       return np.array([y[1], -y[0]])

   y0 = np.array([1.0, 0.0])
   t, y, h = 0.0, y0.copy(), 0.1
   tol = 1e-12

   result = rk_step(RkMethod.PD78, t, y, h, tol, rhs)
   print(result)   # StepResult(y_new=..., error=..., h_next=...)

   y = np.asarray(result.y_new, dtype=float)
   t += h
   h = result.h_next   # 自适应建议的下一步长

自适应传播循环
^^^^^^^^^^^^^^

以下是一个完整的接受-拒绝式传播循环，容差按状态范数缩放，兼顾归一化单位与物理单位：

.. code-block:: python

   def propagate_rk(method, rhs, y0, t_span, tol=1e-12, h0=1.0):
       t0, tf = t_span
       t = float(t0)
       y = np.asarray(y0, dtype=float).copy()
       h = float(h0)
       n_steps = 0
       while t < tf:
           abs_tol = tol * max(1.0, float(np.linalg.norm(y)))
           h_step = min(h, tf - t)
           result = rk_step(method, t, y, h_step, abs_tol, rhs)
           if result.error <= abs_tol:
               y = np.asarray(result.y_new, dtype=float)
               t += h_step
               h = min(result.h_next, h_step * 2.0)
           else:
               h = result.h_next
           n_steps += 1
       return t, y, n_steps

关键参数说明：

- ``tol`` — 相对容差。实际接受阈值是 ``tol * max(1, ||y||)``，保证在归一化单位（~O(1)）和物理单位（km/s，~O(7000)）下行为一致。
- ``h0`` — 初始试探步长。自适应控制器会在几步内收敛到合适步长，``h0`` 只需大致量级正确即可。
- ``result.error`` — 高阶解与嵌入解的 L2 范数差，即局部截断误差估计。
- ``result.h_next`` — 控制器建议的下一步长：``h_next = h * clamp(0.9 * (tol/error)^(1/(p+1)), 0.1, 5)``，其中 ``p`` 为嵌入阶数。


Adams-Bashforth-Moulton（固定步长多步）
---------------------------------------

ABM 是 4 步 4 阶 PECE（Predict-Evaluate-Correct-Evaluate）预测-校正器。与 RK 不同，它每一步只调用一次右端函数（预测后一次、校正后一次），但依赖前 4 步的导数样本（history）。

特点
^^^^

- **固定步长**：history 中的样本按等间距 ``h`` 存储，改变 ``h`` 必须重新初始化 history。
- **低函数求值开销**：长弧段传播时总右端调用次数远低于同阶 RK。
- **启动依赖**：前 4 个样本由 RK89 启动（``initialize_abm_history`` 自动完成）。

基本用法
^^^^^^^^

.. code-block:: python

   from e2m2e.integrators import MultistepMethod, multistep_step, initialize_abm_history

   def rhs(t, y):
       return np.array([y[1], -y[0]])

   y0 = np.array([1.0, 0.0])
   h = 0.01

   # 启动：用 RK89 走 3 步，生成 4 个导数样本
   t, y, history = initialize_abm_history(0.0, y0, h, rhs, n_stages=3)

   # 固定步长推进
   for _ in range(100):
       result = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, rhs, history)
       y = np.asarray(result.y_new, dtype=float)
       t += h
       history = result.history   # 滚动 history（丢弃最老样本，加入最新）

固定步长传播辅助
^^^^^^^^^^^^^^^^

.. code-block:: python

   def propagate_abm(rhs, y0, h, target_t, t0=0.0):
       t, y, history = initialize_abm_history(t0, y0, h, rhs, n_stages=3)
       n_steps = int(round((target_t - t) / h))
       for _ in range(n_steps):
           result = multistep_step(MultistepMethod.ABM, t, y, h, 1e-12, rhs, history)
           y = np.asarray(result.y_new, dtype=float)
           t += h
           history = result.history
       return t, y

步长变更与重新启动
^^^^^^^^^^^^^^^^^^

ABM 的 history 与步长绑定。中途改变 ``h`` 必须重新调用 ``initialize_abm_history``：

.. code-block:: python

   # 先以 h=0.01 传播一段
   t, y, history = initialize_abm_history(0.0, y0, 0.01, rhs)
   ...

   # 需要更密采样：重新启动，不能用旧 history
   t, y, history = initialize_abm_history(t, y, 0.005, rhs)


Cowell（Störmer-Cowell 双积分）
--------------------------------

Cowell 直接积分二阶 ODE :math:`\ddot{x} = a(t, x)`，避免将位置和速度拼接成一阶状态向量。对于纯引力问题，这意味着状态维度减半、每步运算量降低，且 8 阶精度直接作用于位置。

特点
^^^^

- **位置-only 输出**：``cowell_step`` 返回 ``x_new``（位置），不直接给出速度。速度可通过有限差分恢复。
- **加速度-only 右端**：回调 ``accel(t, x)`` 只接收位置，返回加速度。适用于重力、J2 等仅依赖位置的力；含速度项的力（如大气阻力）需要改用 RK 或 ABM。
- **固定步长**：同 ABM，改变 ``h`` 需重新初始化。
- **8 阶精度**：需要 8 个历史加速度样本 + 2 个位置样本，共 10 向量 history。

基本用法
^^^^^^^^

.. code-block:: python

   from e2m2e.integrators import cowell_step, initialize_cowell_history

   def accel(t, x):   # 谐振子: x'' = -x
       return -np.asarray(x, dtype=float)

   x0 = np.array([1.0])
   v0 = np.array([0.0])
   h = 0.01

   # 启动：7 步 RK89 生成 10 向量 history
   t, x, v, history = initialize_cowell_history(0.0, x0, v0, h, accel)

   # 固定步长推进（只更新位置）
   for _ in range(100):
       result = cowell_step(t, h, 1e-12, accel, history)
       x = np.asarray(result.x_new, dtype=float)
       t += h
       history = result.history

固定步长传播辅助
^^^^^^^^^^^^^^^^

.. code-block:: python

   def propagate_cowell(accel, x0, v0, h, target_t, t0=0.0, tol=1e-12):
       t, x, v, history = initialize_cowell_history(t0, x0, v0, h, accel, tol=tol)
       n_steps = int(round((target_t - t) / h))
       for _ in range(n_steps):
           result = cowell_step(t, h, tol, accel, history)
           x = np.asarray(result.x_new, dtype=float)
           t += h
           history = result.history
       return t, x

启动参数
^^^^^^^^

``initialize_cowell_history`` 的 ``n_startup`` 默认 7，必须 ≥ 7 才能填满 8 个加速度样本。增大 ``n_startup`` 会让启动阶段走得更远，但通常 7 步已足够让 RK89 的截断误差低于 Cowell 本身的局部误差。


选择决策树
----------

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

以归一化 LEO + J2 问题（1 天弧段，容差 1e-13）为例：

.. list-table::
   :header-rows: 1

   * - 方法
     - 阶数
     - 步数（约）
     - 每步右端调用
     - 总调用数（约）
   * - PD45
     - 5
     - ~5000
     - 7（7 级）
     - ~35000
   * - PD78
     - 8
     - ~800
     - 13（13 级）
     - ~10400
   * - RK89
     - 9
     - ~800
     - 16（16 级）
     - ~12800
   * - ABM（h=0.002）
     - 4
     - ~500
     - 2（PECE）
     - ~1000
   * - Cowell（h=0.04）
     - 8
     - ~25
     - 2（PECE）
     - ~50

注意：ABM 和 Cowell 的步数与步长直接相关，上表步长是经验取值；RK 的步数由自适应控制器自动决定。Cowell 在此问题上的总调用数最少，但仅限于纯引力场景。


结果类型
--------

.. list-table::
   :header-rows: 1

   * - 类型
     - 字段
     - 说明
   * - ``StepResult``
     - ``y_new``, ``error``, ``h_next``
     - RK 单步结果（新状态、误差估计、建议步长）
   * - ``MultistepResult``
     - ``y_new``, ``error``, ``h_next``, ``history``
     - ABM 单步结果（含滚动 history）
   * - ``CowellResult``
     - ``x_new``, ``error``, ``h_next``, ``history``
     - Cowell 单步结果（位置-only，含滚动 history）


完整轨道传播示例
----------------

以下示例用三种积分器传播同一 LEO + J2 轨道 1 天，比较最终位置：

.. code-block:: python

   import numpy as np
