轨道预报
========

目标
----

用 ``Facade.orbit_propagation`` 把一个初始状态在高精度力模型下外推一段短弧，
拿到逐点轨迹。适合星座外推、轨道衰减分析等确定性预报。

前置
----

- 装好 e2m2e；SPICE 内核可用（默认力模型含太阳／月球第三体引力，读星历）。

代码
----

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()

   result = facade.orbit_propagation(
       initial_state=[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
       epoch="2027-08-02T00:00:00",
       duration=6.0 * 3600.0,
       output_step=600.0,
   )

   print(result.status, result.cause, result.n_points)
   print(result.position_km[-1])
   print(result.final_state)

输出解读
--------

``result`` 是 ``PropagationResponse``：

- ``initial_state`` 是 GCRS 地心惯性系 [x, y, z, vx, vy, vz]，单位 km、km/s；
  ``epoch`` 接受 ISO 字符串或 ``[年, 月, 日, 时, 分, 秒]`` 列表。
- ``time_sec`` / ``times_jd_tdb`` 是输出时刻（秒、JD_TDB 双份），
  ``position_km`` / ``velocity_km_s`` 是 (n, 3) 轨迹，``final_state`` 是
  (6,) 末态——画图、插值、交接下游都用它们。
- ``force_config`` 缺省用默认三体力模型；要自定义（球谐、光压、大气），
  传 ``force_config`` 字典，键值语义同 ``examples/main_propagate.py`` 里的
  ``perturbation`` 开关。

确定性传播失败（步长塌缩等）抛 ``PropagationFailure`` 异常而非返回失败状态。

延伸
----

- CLI：``e2m2e orbit-propagation --help``；MCP：工具名 ``orbit_propagation``；
- 60 天高精度外推示例：:doc:`/examples/propagate`。
