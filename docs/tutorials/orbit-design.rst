任务轨道设计
============

目标
----

用 ``Facade.design_orbit`` 设计一条地月 L2 Halo 轨道：CR3BP 初猜 → 星历修正 →
输出标称轨道。这是 e2m2e 最核心的一档任务。

前置
----

- 装好 e2m2e（:doc:`/install`）；
- SPICE 内核在仓库根 ``kernels/`` 或 ``$SPICE_KERNEL_DIR``（星历修正需要）。

代码
----

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()

   result = facade.design_orbit(
       orbit_type="HALO",
       collinear_point=2,
       amplitude=30000.0,
       epoch=(2024, 1, 1, 0, 0, 0.0),
       duration=7.0 * 86400.0,
       output_step=3600.0,
   )

   print(result.status, result.cause, result.message)
   print(result.orbit_type, result.cr3bp_jacobi, result.correction_iterations)
   print(result.initial_state)

输出解读
--------

``result`` 是 ``DesignOrbitResponse``，按三类信息读：

- 状态三元组：``status == ConvergenceState.CONVERGED`` 表示修正收敛；
  未收敛时 ``cause`` 给稳定原因码（如 ``MAX_ITERATIONS_REACHED``），
  ``message`` 是人读说明。
- 轨道几何：``initial_state`` 是修正后的初始状态（无量纲会合系），
  ``cr3bp_jacobi`` 是 CR3BP Jacobi 常数，``correction_iterations`` 记录
  微分修正迭代次数，``correction_method`` 回显实际执行的修正方法
  （Halo 未显式指定时分派为 ``segmented`` 全程分段打靶）。
- 标称星历：``states`` / ``times`` 是 CR3BP 参考周期轨道（无量纲），
  ``ephemeris`` 是标称星历（GCRS km／m·s⁻¹ + 会合系），直接可画图或喂给
  :doc:`station-keeping`。``taxonomy_labels`` 给分类学标签（如
  ``halo_l2_northern``）。

参数约束（振幅上下限、各 ``orbit_type`` 的必填组合）可用
``facade.valid_ranges().design_orbit`` 查询；完整字段表见 API 参考的
``DesignOrbitRequest`` / ``DesignOrbitResponse``。

延伸
----

- CLI：``e2m2e design-orbit --help``；MCP：工具名 ``design_orbit``；
- 逐族参数示例：:doc:`/tutorials/orbit-catalog` 的族生成参数表；
- 端到端示例脚本：:doc:`/examples/design`。
