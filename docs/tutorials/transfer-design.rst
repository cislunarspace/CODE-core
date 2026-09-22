转移轨道设计
============

目标
----

用 ``Facade.transfer_design`` 设计一条地月脉冲转移。支持四种
``transfer_type``：``HMN``（霍曼直接转移）、``LGA``（月球引力辅助）、
``WSB``（弱稳定边界弹道捕获）、``low_thrust``（低推力）。本篇以 ``HMN``
最小调用入门。

前置
----

- 装好 e2m2e；``HMN`` 为地心惯性两体几何，``LGA``／``WSB`` 需要会合系目标
  星历（惯性星历必须先经 :doc:`spacetime-transform` 的
  ``j2000_to_synodic`` 转换，否则目标态几何全错）。

代码
----

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()

   result = facade.transfer_design(
       transfer_type="HMN",
       tli_epoch="2027-08-02T00:00:00",
       target_orbit_radius_km=384400.0,
       tof_range=[4.0, 6.0],
   )

   print(result.status, result.cause, result.message)
   print(result.transfer_type, "Δv =", result.delta_v, "km/s")
   for event in result.maneuver_events:
       print(event.kind, round(event.t_sec / 86400.0, 2), "天", event.dv_km_s, "km/s")

输出解读
--------

- ``delta_v`` 是总特征速度（km/s）；``maneuver_events`` 是结构化机动事件
  （按 ``t_sec`` 升序）：HMN 为 ``departure``／``arrival`` 两条，到达点即
  近月点。
- ``trajectory`` / ``trajectory_times`` 是转移轨迹（会合旋转系、质心原点、
  物理 km／km·s⁻¹）；``trajectory_gcrs_km`` 是并行的惯性几何段，两者共享
  时刻序列。``state_frame`` 字段标注轨迹的数据系。
- 搜索类类型（``LGA``／``WSB``）不可行时 **不抛异常** ：返回
  ``status=INFEASIBLE`` 等状态，用三元组判断。
- ``top_n`` 参数可开启 top-N 可行解契约，``result.candidates`` 按上报 Δv
  升序返回候选。

长任务与取消
------------

``transfer_design``（与 ``orbit_family_generation``）是 **长任务** ：经 MCP /
CLI / sidecar 调用时跑在独立 worker 子进程
（``python -m e2m2e.api.mcp.worker``）里，取消请求传播为进程 kill，不占调用方
进程；进程内直接调用（如本教程）则在当前进程执行。进度回调形状为
``cb(fraction, message)``。

延伸
----

- CLI：``e2m2e transfer-design --help``；MCP：工具名 ``transfer_design``；
- 二体 Lambert 底层示例（无 SPICE、秒级）：:doc:`/examples/transfer`。
