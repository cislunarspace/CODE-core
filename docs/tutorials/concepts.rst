核心概念
========

本篇给出理解其余教程所需的最低限度背景与约定，并验证环境可用。

CR3BP 与会合坐标系
------------------

e2m2e 的多数设计能力建立在圆型限制性三体问题（CR3BP）上：地球与月球绕公共质心
做圆周运动，航天器质量不影响两者。在该旋转坐标系（会合系）里，五个平动点
L1–L5 静止不动，Halo、NRHO、DRO 等周期轨道族围绕它们存在。会合系坐标原点在
地月质心，长度单位常用地月距离（384400 km）无量纲化。

五层架构
--------

仓库自底向上分五层，依赖严格单向：

#. ``e2m2e/data/``——常数、星历内核、坐标系数据、轨道库存储；
#. ``crates/``——Rust 数值层（积分器、力模型、SPICE FFI），经
   ``e2m2e._integrators`` PyO3 扩展暴露；
#. ``e2m2e/algorithm/``——动力学编排、轨道族、微分修正、转移设计；
#. ``e2m2e/api/``——接口层：``Facade`` / ``Catalog`` / ``Spatiography`` 三个
   暴露类与 MCP / CLI / sidecar 传输层；
#. ``e2m2e/tools/``——日志等辅助。

``e2m2e/mbse/`` 是独立顶层子系统，不在调用链上。教程只涉及接口层；algorithm
层细粒度 API 见 :doc:`/examples/index` 与 API 参考。

统一信封与状态三元组
--------------------

MCP / CLI / sidecar 的每个工具响应都是统一信封
``{status, data, error, meta}``；进程内调用则直接拿到 Pydantic
Response 对象。所有任务级 Response 顶层携带 **状态三元组**
``(status, cause, message)``：

- ``status``（``ConvergenceState``）判断成败；
- ``cause``（``FailureCause``）稳定原因码，用于程序化归因；
- ``message`` 人读说明。

两者由库内不变量约束：每个 ``FailureCause`` 恰好映射到固定的
``ConvergenceState``（如 ``NONE`` 必对应 ``CONVERGED``），不一致会直接抛错。

.. list-table::
   :header-rows: 1

   * - ``ConvergenceState``
     - 含义
   * - ``CONVERGED``
     - 收敛成功
   * - ``ITERATING``
     - 迭代中（仅中间态，同步结果不会以此结束）
   * - ``DIVERGED``
     - 发散
   * - ``STAGNATED``
     - 迭代停滞
   * - ``MAX_ITERATIONS``
     - 达到迭代上限
   * - ``INFEASIBLE``
     - 问题不可行（如无交点、约束违反）
   * - ``COLLISION``
     - 撞击天体
   * - ``FAILED``
     - 其他失败（积分失败、奇异雅可比等）

.. list-table::
   :header-rows: 1

   * - ``FailureCause``
     - 含义
   * - ``NONE``
     - 无失败（成功路径）
   * - ``INTEGRATION_FAILED``
     - 积分失败
   * - ``SINGULAR_JACOBIAN``
     - 雅可比奇异
   * - ``INVALID_PERIOD``
     - 周期不合法
   * - ``MAX_ITERATIONS_REACHED``
     - 迭代上限
   * - ``STAGNATION_DETECTED``
     - 检测到停滞
   * - ``DIVERGENCE_DETECTED``
     - 检测到发散
   * - ``NO_INTERSECTION``
     - 无交点
   * - ``CONSTRAINT_VIOLATION``
     - 约束违反
   * - ``BODY_COLLISION``
     - 撞击天体
   * - ``LEVEL1_CORRECTION_FAILED``
     - 一级修正失败
   * - ``BACKEND_FAILURE``
     - 数值后端失败
   * - ``INVALID_INPUT``
     - 输入不合法
   * - ``UNKNOWN``
     - 未归类失败

确定性失败（参数越界、环境缺失）抛异常（``E2M2EError`` 层次）；搜索类问题
不可行时 **不抛异常** ，返回 ``INFEASIBLE`` 等状态由调用方判断。Rust 扩展缺失抛
``RustExtensionUnavailableError``，绝不静默回退 Python 数值实现。

验证环境
--------

.. code-block:: python

   from e2m2e.api import Facade, tool_inventory

   facade = Facade()
   tools = tool_inventory(facade)
   print(f"e2m2e 工具数：{len(tools)}")
   for tool in tools[:5]:
       print(tool.name)

``tool_inventory`` 返回全部暴露工具（一档任务 + 轨道库 + 分区分析）；能打印出
清单即说明 Python 侧接口层可用。数值层（Rust 扩展）在首次真正计算时按需校验，
缺失时报 ``RustExtensionUnavailableError`` 并给出 ``make dev`` 修复指引。

延伸
----

- 接口面权威清单：``tool_inventory(Facade())``（MCP / CLI / sidecar 均由此派生）；
- 下一步：:doc:`orbit-design` 设计第一条轨道。
