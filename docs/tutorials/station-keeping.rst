轨道保持
========

目标
----

用 ``Facade.control_orbit`` 对一条 Halo 标称轨道做蒙特卡洛保持仿真：测定轨
误差、推力误差下，按目标点宽松控制律（``control_mode=1``）反复修正，统计
Δv 消耗。

前置
----

- 装好 e2m2e；SPICE 内核可用；
- 一条标称轨道：先用 ``design_orbit`` 设计。design→control 的星历交接走
  轨道库记录（``input_record_id``，ADR 0031 的谱系接缝），因此给 ``Config``
  配 ``catalog_dir`` 并开 ``catalog_enabled``；也可以直接给 ``input_ephemeris``
  （星历文件路径或 ``EphemerisTable`` 对象），本篇取记录接缝。

代码
----

.. code-block:: python

   import tempfile
   from pathlib import Path

   from e2m2e.api import Facade
   from e2m2e.api.config import Config

   facade = Facade(Config(catalog_dir=Path(tempfile.mkdtemp()), catalog_enabled=True))

   nominal = facade.design_orbit(
       orbit_type="HALO",
       collinear_point=2,
       amplitude=30000.0,
       epoch=(2024, 1, 1, 0, 0, 0.0),
       duration=36.0 * 86400.0,
       output_step=3600.0,
   )
   print("标称轨道：", nominal.status, "入库记录：", nominal.record_id)

   ctl = facade.control_orbit(
       input_record_id=nominal.record_id,
       control_mode=1,
       control_interval=10.0,
       num_controls=2,
       num_monte_carlo=1,
       output_step=3600.0,
       mu=nominal.mu,
   )

   print(ctl.status, ctl.cause, ctl.message)
   print("失败样本数：", ctl.num_failed)
   print("总 Δv / 最大单次 Δv（m/s）：", ctl.sk_statistic["rows"][0][:2])

输出解读
--------

- ``design_orbit`` 在库开启时把产物自动入库，``record_id`` 即记录 id；
  ``control_orbit`` 经 ``input_record_id`` 取其星历段作标称轨道，站保产物
  记录自动以 ``source_record_id`` 指回设计记录（谱系跨进程不断）。
- ``num_failed`` 是蒙特卡洛失败样本数；``sk_statistic["rows"]`` 逐样本统计，
  首行前两列是总 Δv 与最大单次 Δv（m/s）；``maneuvers`` 含逐次机动的
  时刻与脉冲（``mjd_tdb`` / ``delta_v_mps``）。
- ``controlled_ephemeris`` 是受控真实轨道星历（全部样本失败时为 ``None``）。
- 惯例参数：工程评估把 ``num_monte_carlo`` 提到 100；测定轨与推力误差、
  推力上下限等都有缺省值（见 API 参考 ``ControlOrbitRequest``）。
  控制模式 1–6 对应目标点宽松／严格、特征点及其角动量管理组合。

样本量与弧长是本教程与工程仿真的唯一差异：``examples/main_control.py`` 用
同一链路出对比图。

延伸
----

- CLI：``e2m2e control-orbit --help``；MCP：工具名 ``control_orbit``；
- 出图示例：:doc:`/examples/control`。
