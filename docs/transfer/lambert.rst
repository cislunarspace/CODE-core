Lambert 求解与 porkchop 扫描
============================

Lambert 问题给定两端位置与飞行时间，求连接两点的二体轨道。本页介绍三部分：
二体 Lambert 求解器（Rust Izzo 内核）、porkchop 扫描，以及以二体解为初猜的
CR3BP 三体打靶。

二体 Lambert 求解器
-------------------

:func:`~e2m2e.transfer.lambert.solve_lambert` 解单次二体 Lambert 问题，
算法为 Izzo (2015)，内核用 Rust 实现（``e2m2e-propagation`` crate），
经 ``e2m2e._integrators`` 暴露，Python 侧只做类型转换与结果封装。

.. code-block:: python

   from e2m2e.transfer import solve_lambert

   # Vallado 经典算例
   r0 = [5000.0, 10000.0, 2100.0]      # km
   rf = [-14600.0, 2500.0, 7000.0]     # km
   tof = 3600.0                        # s
   mu = 398600.4418                    # 地球 GM，km³/s²

   sol = solve_lambert(r0, rf, tof, mu)
   print(f"出发速度: {sol.v0}")        # km/s
   print(f"到达速度: {sol.vf}")
   print(f"迭代次数: {sol.n_iter}")

``direction`` 选择转移角方向：``"short"`` 表示转移角 < π（默认），``"long"`` 表示 > π。
``revs`` 指定完整圈数，多圈时返回右分支（低能）解；飞行时间低于该圈数
最小转移时间时抛 ``ValueError``。

返回的 :class:`~e2m2e.transfer.lambert.LambertSolution` 含出发速度 ``v0`` 与
到达速度 ``vf``，均为 ``(3,)`` 向量，单位 km/s；另有迭代次数 ``n_iter``
与圈数 ``revs``。

批量求解
--------

:func:`~e2m2e.transfer.lambert.solve_lambert_batch` 对 N 组几何 × M 个飞行时间
的网格批量求解，一次调用进入 Rust 内核：

.. code-block:: python

   import numpy as np
   from e2m2e.transfer import solve_lambert_batch

   r0_list = np.tile(r0, (4, 1))       # (N, 3)
   rf_list = np.tile(rf, (4, 1))       # (N, 3)
   tofs = [3600.0, 7200.0, 10800.0]    # (M,)

   out = solve_lambert_batch(r0_list, rf_list, tofs, mu)
   # out 形状 (N, M, 2, 3)：[..., 0, :] 为 v0，[..., 1, :] 为 vf

无解的组合（如弦长为零）对应位置填 NaN，不影响其余组合。

porkchop 扫描
-------------

:func:`~e2m2e.transfer.porkchop.porkchop` 在出发时间 × 飞行时间网格上逐点解
Lambert 问题，得到双脉冲 ΔV 网格，即 porkchop 图的数据层。出发与到达终端
经 :class:`~e2m2e.transfer.terminal.TerminalCondition` 接口提取状态
（如 :class:`~e2m2e.transfer.terminal.OrbitTerminal`，或自定义实现），
本函数不关心状态如何产生。

对网格点 ``(t_dep, tof)``：出发终端状态取 ``t_dep`` 时刻，到达终端状态取
``t_dep + tof`` 时刻，脉冲为转移速度与终端轨道速度之差。

.. code-block:: python

   import numpy as np
   from e2m2e.transfer import porkchop

   t_dep = np.linspace(0.0, 3600.0, 20)        # 出发时间网格，s
   tof = np.linspace(900.0, 5 * 3600.0, 30)    # 飞行时间网格，s

   data = porkchop(dep, arr, t_dep, tof, mu=398600.4418, dynamics=None)

   print(data.dv1.shape)     # (20, 30)，出发脉冲，km/s
   print(data.dv2.shape)     # 到达脉冲
   print(data.total.shape)   # 总脉冲 dv1 + dv2

   # 画总 ΔV 等值线图
   ax = data.plot()

其中 ``dep`` / ``arr`` 是实现 ``TerminalCondition`` 的终端对象；
解析终端（如圆轨道）不依赖动力学传播时 ``dynamics`` 可传 ``None``。
返回的 :class:`~e2m2e.transfer.porkchop.PorkchopData` 携带三个网格与时间轴，
``plot()`` 直接用 matplotlib 画等值线。

三体打靶 ThreeBodyLambert
-------------------------

二体 Lambert 解忽略第三体引力，在地月空间只是初猜。
:class:`~e2m2e.transfer.three_body_lambert.ThreeBodyLambert` 以二体解为初猜，
在 CR3BP 动力学下用阻尼 Newton 打靶修正出发速度，使给定飞行时间后的末端
位置命中目标：

1. 两端物理状态 (km, km/s) 经 ``CR3BP_System.physical_to_dimensionless`` 无量纲化
2. 无量纲几何上以 μ = 1 调 ``solve_lambert`` 得初猜出发速度
3. Newton 迭代：传播 ``with_stm=True``，取末端 STM 的 Φ_rv 块（``Φ[0:3, 3:6]``）
   解修正量；整步增大误差时步长减半；收敛判据为末端位置误差 < 1e-8（无量纲）

.. code-block:: python

   from e2m2e.transfer import StateTerminal, ThreeBodyLambert

   shooter = ThreeBodyLambert(dynamics)   # system 须已初始化特征尺度

   # 终端状态为物理单位 (km, km/s)；到达端仅位置为约束，速度用于算到达脉冲
   sol = shooter.solve(
       StateTerminal(s0, 0.0),
       StateTerminal(s1, tof),
       tof,                    # 飞行时间，s
       guess="lambert",        # 初猜来源；"orbit" 直接用出发速度
   )

   if sol.converged:
       print(f"出发脉冲: {sol.arcs[0].delta_v:.6f} km/s")
       print(f"到达脉冲: {sol.arrival_delta_v:.6f} km/s")
       print(f"总脉冲: {sol.total_delta_v:.6f} km/s")

返回 :class:`~e2m2e.transfer.config.TransferSolution`，单弧，物理单位；
未收敛时 ``converged=False`` 且 ``message`` 说明残余误差。典型场景（同一周期
轨道上两相位点间转移、Lyapunov → Halo 交会）的收敛行为见
``tests/transfer/test_three_body_lambert.py``。

多脉冲转移与主矢量检验
----------------------

双脉冲解只在特定几何下最优。
:class:`~e2m2e.transfer.multi_impulse.MultiImpulseTransfer` 在固定端点
（:class:`~e2m2e.transfer.terminal.StateTerminal`，位置、速度、时刻均固定）
之间规划 n 脉冲转移：决策变量为各中途脉冲节点的时刻与位置 ``[t_i, r_i]``，
相邻节点间的弧段由 Lambert 封闭（默认二体 ``solve_lambert``，可切
``ThreeBodyLambert`` 打靶精修），脉冲是封闭结果的进出弧速度差，由 scipy
SLSQP 最小化总 ΔV。

.. code-block:: python

   import numpy as np
   from e2m2e.transfer import MultiImpulseTransfer, StateTerminal

   MU_EARTH = 398600.4418  # km³/s²
   R1, R2 = 7000.0, 42164.0  # LEO → GEO，km
   TOF_HOHMANN = np.pi * np.sqrt(((R1 + R2) / 2) ** 3 / MU_EARTH)

   def circular(r, angle=0.0):
       """半径 r、相位角 angle 的圆轨道状态（逆时针）。"""
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
   print(f"总脉冲: {sol.total_delta_v:.4f} km/s")   # 霍曼基准 3.7708

``optimize(n_impulses, x0=...)`` 的决策变量只含中途节点（m = n − 2 个），
n=2 时无自由变量、直接封闭单弧；``x0`` 给出中途节点的初猜
``[t_1, r_1, ...]``。优化后 ``transfer.legs`` 刷新为
:class:`~e2m2e.transfer.multi_impulse.Impulse` 与
:class:`~e2m2e.transfer.multi_impulse.CoastArc` 交替的序列。

:meth:`~e2m2e.transfer.multi_impulse.MultiImpulseTransfer.check_primer_vector`
对给定解做 Lawden 主矢量检验：由端点横截条件
p(t0) = Δv̂₀、p(tf) = Δv̂_f 确定主矢量初值，协态经 STM 携载得到 p(t) 曲线
（公式出处为 Prussing《Optimal Spacecraft Trajectories》第 3–4 章）。
最优性的必要条件是全程 ``|p(t)| ≤ 1`` 且脉冲点 ``|p| = 1``、方向与 p 共线；
弧内 ``|p| > 1`` 时在峰值处插入中途脉冲可降低总 ΔV（Lion & Handelsman 1968）。
霍曼转移满足 Lawden 条件；同一端点但飞行时间取 0.5 倍霍曼时间的双脉冲解
弧内 ``|p| > 1``，检验会给出插入建议，以建议点为零脉冲初猜做三脉冲优化，
总 ΔV 随之下降：

.. code-block:: python

   report = transfer.check_primer_vector(sol, n_samples=300)
   print(f"Lawden 条件满足: {report.lawden_satisfied}")
   if not report.lawden_satisfied:
       x0 = np.concatenate(
           [[report.suggested_insertion_time],
            report.suggested_insertion_position]
       )
       sol3 = transfer.optimize(3, x0=x0)   # 三脉冲总 ΔV 低于双脉冲

完整算例见 ``tests/transfer/test_multi_impulse.py``。

.. automodule:: e2m2e.transfer.lambert
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.transfer.porkchop
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.transfer.three_body_lambert
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.transfer.multi_impulse
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
