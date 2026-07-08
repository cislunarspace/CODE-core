标准形化简流水线
================

标准形（Normal Form）把圆型限制性三体问题（CR3BP）平动点附近的复杂动力学
化简为一组作用量-角变量形式的"表征参数"，让轨道设计师用几个常数描述一条
本需六个状态量随时间演化的轨道。

``NormalFormPipeline`` 把这条化简路径串成一行调用：

    星历轨道初值 → 动力学替代轨道 → quasi-Floquet 变换 → 中心流形化简 → 表征参数

特性
----

- 一行代码完成"星历轨道 → 表征参数"
- 返回的 ``NormalFormResult`` 同时暴露化简诊断与坐标变换入口
- 平动点（L1–L5）、历元、展开阶数由 ``NormalFormContext`` 统一配置

使用方法
--------

.. code-block:: python

   from e2m2e.algorithms.normal_form import NormalFormContext, NormalFormPipeline
   from e2m2e.core import CR3BP_System, LibrationPoint

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
（``success``、``residual``、``metadata``），也把四个子结果作为一等公民
字段暴露，供需要深入到某一层的调用方使用。

可配置旋钮
----------

构造 ``NormalFormPipeline`` 时可覆盖默认行为：

- ``quasi_floquet_method``：``"matrix"``（默认，36 维直接积分）或
  ``"lie_algebra"``（21 维 sp(6) 参数化，自动保辛）
- ``center_max_order``：中心流形 Lie 变换截断阶数，默认 ``10``
- ``center_steps``：化简步骤元组，默认 ``("invariant", "center")``
- ``dynamical_kwargs``：透传给动力学替代 corrector 的覆盖项
  （如 ``{"t_total": 8.0}``）

子模块
------

流水线内部依次调用以下 reducer，需要单独访问某一层结果时可经
``result.ds_result`` / ``result.qf_result`` / ``result.cm_result`` 取得：

- 动力学替代：``DynamicalSubstituteCorrector``
- quasi-Floquet 变换：``QuasiFloquetReducer``
- 中心流形化简：``CenterManifoldReducer``
- 表征参数目录：``LibrationCatalogTransformer``

可运行示例见 ``examples/normal_form_example.py``。
