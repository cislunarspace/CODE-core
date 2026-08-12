不变流形与庞加莱截面
====================

不变流形（稳定/不稳定）是周期轨道附近的渐近轨道族，是低能量转移设计的
基本构件。本页介绍流形计算、庞加莱截面工具，以及基于流形拼接的低能
转移流水线。

不变流形计算
------------

种子生成原理：周期轨道单值矩阵 M（沿轨道传播一周的 STM）的实特征值给出
流形方向，稳定流形取 abs(λ) < 1 的实特征向量，不稳定流形取 abs(λ) > 1 的
实特征向量；
单位圆上的特征值（含 λ=1 的周期方向）不构成双曲方向，予以剔除。
沿轨道均匀取 n_points 个相位点，用从首点到各相位的 STM 把特征向量转运到
该相位，位置部分归一化后施加 ±ε 的无量纲扰动得到种子；稳定流形反向积分、
不稳定流形正向积分得到流形管。

:class:`~e2m2e.algorithm.manifold.manifolds.InvariantManifold` 的用法：

.. code-block:: python

   from e2m2e.algorithm.manifold import InvariantManifold, ManifoldKind

   # orbit 为周期轨道：须关联 system 且 period 已知（可只存首点）
   epsilon = 50.0 / 384405.0   # 无量纲扰动幅度，典型取 50 km / DU
   manifold = InvariantManifold(orbit, ManifoldKind.STABLE, "-", epsilon)

   # 相位扫掠种子，形状 (n_points, 6)
   seeds = manifold.seeds(12)

   # 批量传播流形弧（稳定流形反向积分，t_span 取绝对值）
   tube = manifold.propagate(4.0)
   print(f"流形弧数: {len(tube.trajectories)}")

``branch`` 取 ``"+"`` 或 ``"-"``，对应扰动的两个方向，分别走向轨道两侧。
返回的 :class:`~e2m2e.algorithm.manifold.manifolds.ManifoldTube` 携带轨道引用、
流形类型、分支与 ε，``trajectories`` 为流形弧列表（无量纲 CR3BP 态）。

给 ``propagate`` 传入 ``section`` 参数时，每条弧在首次穿越截面处截断，
并把求精后的穿越态追加为弧的末点：

.. code-block:: python

   from e2m2e.algorithm.manifold import PoincareSection

   section = PoincareSection.periapsis("earth", orbit.system)
   tube = manifold.propagate(4.0, section=section)

庞加莱截面
----------

:class:`~e2m2e.algorithm.manifold.sections.PoincareSection` 由标量函数 s(state) 的
零等值面定义，提供两类构造：

- ``PoincareSection.plane(axis, value)``：平面截面 s = state[axis] − value，
  ``axis`` 为状态分量索引（0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz）
- ``PoincareSection.periapsis(center, system)``：近拱点截面 s = r·v，
  r 为相对 center 天体（主/次天体名称，不区分大小写）的位置

穿越检测采用事后方案：传播时密采样（流形传播默认步长 0.005 无量纲时间），
逐采样点求截面函数值，符号变化区间内对分段线性插值态用 Brent 法求精。
平面截面穿越残差可达 1e-10 以下。

``crossings()`` 检测流形管中所有弧的穿越，返回
:class:`~e2m2e.algorithm.manifold.sections.SectionCrossings`：

.. code-block:: python

   crossings = section.crossings(tube)
   print(crossings.states.shape)          # (k, 6)，插值求精后的穿越态
   print(crossings.times.shape)           # (k,)，穿越时刻
   print(crossings.trajectory_index)      # (k,)，每个穿越点所属弧的索引

``crossings()`` 是事后检测（先传播、再在采样点上找穿越）。若要在积分
过程中检测穿越（例如首次到达截面即停），用
:meth:`~e2m2e.algorithm.manifold.sections.PoincareSection.event` 生成 scipy 语义的
事件函数传给 ``Dynamics.propagate(events=...)``：

.. code-block:: python

   event = section.event(direction=-1, terminal=True)
   result = dynamics.propagate(y0, (0.0, 10.0), events=[event])

详见 :doc:`../core/dynamics` 的「事件检测」一节。

流形拼接与低能转移
------------------

两条流形管在同一截面上的穿越点两两配对，可拼接成转移初猜。
:func:`~e2m2e.algorithm.transfer.low_energy.patch_manifolds` 按加权拼接代价
``w_r·|Δr| + w_v·|Δv|`` 升序输出候选（:class:`~e2m2e.algorithm.transfer.low_energy.PatchCandidate`）：

.. code-block:: python

   from e2m2e.algorithm.transfer import patch_manifolds

   # 出发轨道不稳定流形 + 目标轨道稳定流形，同一截面
   candidates = patch_manifolds(tube_a, tube_b, section, weights=(1.0, 1.0))
   best = candidates[0]
   print(f"|Δr|={best.delta_r:.4e}, |Δv|={best.delta_v:.4e}")

:func:`~e2m2e.algorithm.transfer.low_energy.design_low_energy_transfer` 把上述步骤
串成流水线：出发轨道不稳定流形与目标轨道稳定流形（± 分支四种组合全局
取最优）传播到次天体近拱点截面，取最优拼接候选，出发弧直接用流形弧，
拼接点之后由 :class:`~e2m2e.algorithm.transfer.three_body_lambert.ThreeBodyLambert`
打靶闭合到目标轨道。脉冲由三段构成：出发脉冲（上出发流形）、拼接脉冲
（截面处）、到达脉冲（入目标流形）。

.. code-block:: python

   from e2m2e.algorithm.transfer import OrbitTerminal, design_low_energy_transfer
   from e2m2e.data.templates import ConvergenceState

   sol = design_low_energy_transfer(OrbitTerminal(departure_orbit), target_orbit)

   if sol.status == ConvergenceState.CONVERGED:
       print(f"弧段数: {len(sol.arcs)}")           # 2
       print(f"出发脉冲: {sol.arcs[0].delta_v:.6f} km/s")
       print(f"拼接脉冲: {sol.arcs[1].delta_v:.6f} km/s")
       print(f"到达脉冲: {sol.arrival_delta_v:.6f} km/s")
       print(f"总脉冲: {sol.total_delta_v:.6f} km/s")
       print(f"转移时间: {sol.transfer_time:.1f} s")

返回两段弧的 :class:`~e2m2e.algorithm.transfer.config.TransferSolution`，物理单位。
当前仅支持 CR3BP 模型；星历转换（CR3BP 闭合解 → 星历模型）尚未接入，
``epoch`` 参数为其预留入口。端到端基准见 ``tests/algorithm/transfer/test_low_energy.py``：
L1 Lyapunov 族内中间轨道到大幅值轨道，拼接脉冲在几十 m/s 量级。

.. automodule:: e2m2e.algorithm.manifold.manifolds
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.manifold.sections
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.transfer.low_energy
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
