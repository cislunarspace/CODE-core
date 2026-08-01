坐标系
======

e2m2e 的坐标系层负责位置、速度等矢量在不同参考框架之间的变换。

核心概念
--------

- **Axes（坐标轴）** — 坐标系的"朝向"部分。给定历元 ``et``，返回旋转矩阵 ``R``，
  使得 ``r_icrf = R @ r_axes``。
- **Origin（原点）** — 坐标系的"位置"部分。``state(et)`` 返回原点在 ICRF/J2000 中的绝对状态。
- **CoordinateSystem（坐标系）** — 一个 Axes 加一个 Origin 拼成的完整数学参考框架。

.. code-block:: python

   from e2m2e.algorithm.dynamics import CoordinateSystem, ICRSAxes, ITRFSpiceAxes, CelestialBodyOrigin

   # ICRF 惯性系（地球中心）
   icrf = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

坐标系族位于 ``e2m2e.algorithm.coordinate`` 子包，上述类在 ``e2m2e.core`` 顶层
重导出；旧的 ``e2m2e.core.<mod>`` 模块路径保留兼容别名。

坐标轴类型
----------

e2m2e 提供多种坐标轴实现：

.. list-table::
   :header-rows: 1

   * - 类型
     - 说明
     - 用途
   * - ``ICRSAxes``
     - 惯性 ICRF 系（单位阵）
     - 星历传播默认
   * - ``ITRFSpiceAxes``
     - SPICE-backed 高精度 ITRF93（需加载 LSK/PCK 内核）
     - 球谐引力场展开
   * - ``ITRFApproxAxes``
     - 低精度 ITRF 近似
     - 大气阻力计算
   * - ``VNBAxes``
     - 动态坐标轴（速度-法向-副法向）
     - 推力方向
   * - ``LVLHAxes``
     - 动态坐标轴（当地垂直-当地水平）
     - 轨道保持

动态坐标轴
----------

VNBAxes 和 LVLHAxes 是动态坐标轴——旋转矩阵不仅依赖历元，还依赖航天器瞬时状态。
使用前必须先调用 ``update(et, state)`` 刷新内部方向缓存。

状态转换
--------

.. code-block:: python

   # 在不同坐标系间转换状态
   state_itrf = icrf.transform_state(
       state_j2000, from_cs=j2000_cs, to_cs=itrf_cs, et=et
   )

   # 转换向量（不含原点平移）
   accel_j2000 = icrf.transform_vector(
       accel_itrf, from_cs=itrf_cs, to_cs=j2000_cs, et=et
   )

坐标系与单位
------------

同一坐标系中的状态可用不同单位系统表示。``UnitSystem`` 枚举标识数值的量纲：

- ``DIMENSIONLESS`` — 无量纲单位（DU, TU, VU）
- ``SI`` — 国际单位制（km, s, km/s）

``Orbit`` 状态由绑定的 ``System`` 解释：``frame`` 与 ``unit_system`` 由
``System`` 基类定义，分别标识参考系与量纲；坐标变换由 ``CoordinateSystem``
对象完成（星历系统可持有一个默认 ``coordinate_system``）。
