星历系统与动力学
================

e2m2e 提供基于 SPICE 的星历系统和 N 体动力学模型，用于高精度轨道设计。

SPICE 管理器
------------

:class:`~e2m2e.core.spice.SPICEManager` 封装 NASA SPICE 工具包，提供天体星历查询、时间转换和内核管理。

**内核类型：**

- **闰秒内核** (``.tls``)：UTC ↔ ET 时间转换，自动加载
- **星历内核** (``.bsp``)：天体位置/速度数据，需手动加载

**推荐星历内核：**

============  ====================================
文件名        说明
============  ====================================
de440.bsp     JPL DE440（推荐，覆盖 1550–2650 年）
de440s.bsp    JPL DE440 精简版（1849–2150 年）
de435.bsp     JPL DE435（1550–2650 年）
============  ====================================

内核文件从 `NASA NAIF <https://naif.jpl.nasa.gov/naif/data.html>`_ 下载。

.. code-block:: python

   from e2m2e.core.spice import SPICEManager

   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   # 时间转换
   et = mgr.utc_to_et("2024-01-01T00:00:00")

   # 获取月球状态
   state = mgr.get_body_state("MOON", et, "J2000", "EARTH")

   mgr.unload_kernel("path/to/de440.bsp")

星历系统
--------

:class:`~e2m2e.core.ephemeris_system.EphemerisSystem` 管理一组天体的星历查询，提供统一的数据访问层。

.. code-block:: python

   from e2m2e.core.spice import SPICEManager
   from e2m2e.core.ephemeris_system import EphemerisSystem

   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=mgr,
       origin="EARTH",
       frame="J2000"
   )

   # 查询月球位置
   moon_pos = system.get_body_position("MOON", et)  # km

   # 查询太阳状态（位置+速度）
   sun_state = system.get_body_state("SUN", et)  # [km, km/s]

   # 获取引力参数
   gm_moon = system.get_gm("MOON")

星历动力学
----------

:class:`~e2m2e.core.ephemeris_dynamics.EphemerisDynamics` 实现受限 N 体问题，利用 SPICE 星历数据计算多体引力加速度。

**物理模型：**

以 ``system.origin`` 为坐标原点（通常是地球），原点天体施加中心引力，其余天体施加第三体摄动：

.. math::

   \mathbf{a} = -\frac{\mu_0 \mathbf{r}}{|\mathbf{r}|^3}
   - \sum_i \mu_i \left[
     \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3}
     + \frac{\mathbf{r}_i}{|\mathbf{r}_i|^3}
   \right]

其中 :math:`\mathbf{r}_i` 为第 i 个天体的位置，:math:`\mu_i` 为其引力参数。

.. code-block:: python

   from e2m2e.core.ephemeris_dynamics import EphemerisDynamics

   dynamics = EphemerisDynamics(system)

   # 传播轨道
   import numpy as np
   state0 = np.array([r1, r2, r3, v1, v2, v3])  # km, km/s
   t_span = (0, 86400)  # 秒
   result = dynamics.propagate(state0, t_span)

   # 结果
   print(result.states.shape)  # (n_points, 6)
   print(result.stm.shape)     # (n_points, 6, 6)

**积分器配置：**

- 默认积分器: ``DOP853`` (8 阶 Runge-Kutta)
- 默认最大步长：60 秒
- 自适应步长：根据传播时长自动调整

与 CR3BP 动力学的区别
---------------------

.. list-table::
   :header-rows: 1

   * - 特性
     - CR3BP_Dynamics
     - EphemerisDynamics
   * - 坐标系
     - 旋转坐标系（无量纲）
     - 惯性坐标系（J2000，有量纲）
   * - 单位
     - 无量纲（DU, TU）
     - 物理单位（km, s）
   * - 天体数量
     - 2 个主天体
     - N 个天体（可配置）
   * - 精度
     - 理想圆轨道
     - 高精度星历数据
   * - 适用场景
     - 概念设计、周期轨道
     - 高精度任务设计
