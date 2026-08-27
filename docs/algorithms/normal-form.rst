Normal Form Reduction Pipeline / 标准形化简流水线
=================================================

[English](#normal-form-reduction-pipeline) | [简体中文](#中文)

English
-------

The normal form reduces the complex dynamics near CR3BP libration points into
action-angle characterizing parameters, letting designers describe an orbit that
would otherwise need six time-evolving state variables with a few constants.

``NormalFormPipeline`` chains the reduction into one call:

    ephemeris orbit initial values → dynamical substitute orbit → quasi-Floquet
    transform → center-manifold reduction → characterizing parameters

Features
~~~~~~~~

- Ephemeris-orbit → characterizing parameters in one line
- The returned ``NormalFormResult`` exposes both reduction diagnostics and
  coordinate-transform entries
- Libration point (L1–L5), epoch, expansion order configured via ``NormalFormContext``

Usage
~~~~~

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormContext, NormalFormPipeline
   from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint

   system = CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")
   context = NormalFormContext(
       system=system,
       libration_point=LibrationPoint.L1,
       epoch=2451545.0,
       order=4,
   )

   # rho-frame initial values [ρ, ρ̇] (nondimensional)
   x0 = [1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4]

   result = NormalFormPipeline(context).reduce(x0)

   # rho coords → characterizing parameters [q1, p1, I2, θ2, I3, θ3]
   param = result.catalog_transformer.rho_to_param(x0, t=0.0)

``result`` is an immutable container providing whole-pipeline convergence
diagnostics (``success``, ``substitute_residual``, ``message``, ``metadata``)
and exposing the four sub-results as first-class fields for consumers digging
into one tier. ``residual`` is a backward-compat alias of ``substitute_residual``
— prefer the latter in new code.

Persistence
~~~~~~~~~~~

``NormalFormResult`` serializes to disk for archiving and reuse:

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormResult

   # Save as .npz (all sub-results + context params serialized)
   result.save("result.npz")

   # Rebuild; catalog_transformer reconstructed from three sub-results automatically
   result = NormalFormResult.load("result.npz")

``catalog_transformer`` isn't stored separately — rebuilt on load from the
dynamical-substitute / quasi-Floquet / center-manifold sub-results, numerically
equivalent on ``rho_to_param`` / ``param_to_rho`` to the original.

Tunable knobs
~~~~~~~~~~~~~

Constructing ``NormalFormPipeline`` can override defaults:

- ``quasi_floquet_method``: matrix (default; 36-dim direct integration) or
  lie_algebra (21-dim sp(6) parameterization, symplectic by construction)
- ``center_max_order``: center-manifold Lie-transform truncation order, default 10
- ``center_steps``: reduction-step tuple, default ("invariant", "center")
- ``dynamical_kwargs``: overrides passed to the substitute corrector (e.g., {"t_total": 8.0})

Submodules
~~~~~~~~~~

The pipeline calls these reducers in turn; reach one tier's results via
``result.ds_result`` / ``result.qf_result`` / ``result.cm_result``:

- Dynamical substitute: ``DynamicalSubstituteCorrector``
- Quasi-Floquet transform: ``QuasiFloquetReducer``
- Center manifold: ``CenterManifoldReducer``
- Characterizing-parameter catalog: ``LibrationCatalogTransformer``

中文
----

标准形（Normal Form）把圆型限制性三体问题（CR3BP）平动点附近的复杂动力学
化简为一组作用量-角变量形式的表征参数，让轨道设计师用几个常数描述一条
本需六个状态量随时间演化的轨道。

``NormalFormPipeline`` 把这条化简路径串成一行调用：

    星历轨道初值 → 动力学替代轨道 → quasi-Floquet 变换 → 中心流形化简 → 表征参数

特性
~~~~

- 一行代码完成星历轨道 → 表征参数
- 返回的 ``NormalFormResult`` 同时暴露化简诊断与坐标变换入口
- 平动点（L1~L5）、历元、展开阶数由 ``NormalFormContext`` 统一配置

使用方法
~~~~~~~~

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormContext, NormalFormPipeline
   from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint

   system = CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")
   context = NormalFormContext(
       system=system,
       libration_point=LibrationPoint.L1,
       epoch=2451545.0,
       order=4,
   )

   # rho 坐标初值 [ρ, ρ̇]（无量纲）
   x0 = [1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4]

   result = NormalFormPipeline(context).reduce(x0)

   # rho 坐标 → 表征参数 [q1, p1, I2, θ2, I3, θ3]
   param = result.catalog_transformer.rho_to_param(x0, t=0.0)

``result`` 是一个不可变容器，既给出整条流水线的收敛诊断
（``success`` 、``substitute_residual`` 、``message`` 、``metadata`` ），也把四个子结果作为一等公民
字段暴露，供需要深入到某一层的调用方使用。``residual`` 是
``substitute_residual`` 的向后兼容别名，新代码应使用后者。

结果保存与加载
~~~~~~~~~~~~~~~

``NormalFormResult`` 可序列化到磁盘，便于把化简结果存档复用：

.. code-block:: python

   from e2m2e.algorithm.normal_form import NormalFormResult

   # 保存为 .npz（全部子结果与 context 参数一并序列化）
   result.save("result.npz")

   # 重建；catalog_transformer 由三个子结果自动复原
   result = NormalFormResult.load("result.npz")

``catalog_transformer`` 不单独存储，加载时由动力学替代、quasi-Floquet、
中心流形三个子结果重建，与原始对象在 ``rho_to_param`` / ``param_to_rho``
上数值等价。

可配置旋钮
~~~~~~~~~~~

构造 ``NormalFormPipeline`` 时可覆盖默认行为：

- ``quasi_floquet_method`` ：matrix（默认，36 维直接积分）或 lie_algebra（21 维 sp(6) 参数化，自动保辛）
- ``center_max_order`` ：中心流形 Lie 变换截断阶数，默认 10
- ``center_steps`` ：化简步骤元组，默认 ("invariant", "center")
- ``dynamical_kwargs`` ：透传给动力学替代 corrector 的覆盖项（如 {"t_total": 8.0}）

子模块
~~~~~~

流水线内部依次调用以下 reducer，需要单独访问某一层结果时可经
``result.ds_result`` / ``result.qf_result`` / ``result.cm_result`` 取得：

- 动力学替代：``DynamicalSubstituteCorrector``
- quasi-Floquet 变换：``QuasiFloquetReducer``
- 中心流形化简：``CenterManifoldReducer``
- 表征参数目录：``LibrationCatalogTransformer``
