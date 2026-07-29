系统
====

e2m2e 的系统层定义天体几何、引力与运动学模型，是后续所有计算的上下文。

System 基类
-----------

:class:`~e2m2e.core.system.System` 是抽象基类，定义系统要回答的三个问题：

1. 用什么坐标框架？由 ``frame`` 标识。
2. 数值用什么单位？由 ``unit_system`` 决定。
3. 各天体的引力参数？通过 ``gravitational_parameter(body)`` 查询。

``coordinate_system`` 不在基类接口中，仅 ``EphemerisSystem`` 可选持有
（供坐标变换与 ForceModel 传播使用）。

CR3BP 系统
----------

:class:`~e2m2e.core.cr3bp_system.CR3BP_System` 描述圆型限制性三体问题：
两个主天体绕公共质心作圆周运动，第三体质量可忽略。

**创建系统：**

.. code-block:: python

   from e2m2e.core import CR3BP_System

   system = CR3BP_System(
       mu=0.0121506683,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

``_with_default_scales()`` 自动设置地月特征尺度（384405 km、27.32 天）。
内置 Earth-Moon、Sun-Earth、Sun-Jupiter 三组默认尺度；其他组合抛
``ValueError``，需用 ``set_characteristic_scales()`` 显式设置。

**质量参数 μ：**

.. math::

   \mu = \frac{m_2}{m_1 + m_2}

- 地月系统: μ ≈ 0.01215
- 日地系统: μ ≈ 3.0039×10⁻⁶
- 日木系统: μ ≈ 9.535×10⁻⁴

**平动点：**

.. code-block:: python

   system.compute_libration_points()

   print(system.L1)  # L1 坐标 [x, 0, 0]
   print(system.L2)
   print(system.L3)
   print(system.L4)  # 三角平动点
   print(system.L5)

五个平动点中，L1、L2、L3 为共线平衡点（在 x 轴上），L4、L5 为三角平衡点。

**Jacobi 常数：**

.. math::

   C_J = 2\Omega - v^2

其中 Ω 为伪势能，v 为速度大小。Jacobi 常数是 CR3BP 中唯一的运动积分。

.. code-block:: python

   state = [0.8, 0, 0, 0, 0.6, 0]
   CJ = system.get_jacobi_constant(state)
   print(f"C_J = {CJ}")

**单位转换：**

.. code-block:: python

   # 无量纲 → 物理
   phys = system.dimensionless_to_physical(dimensionless_state)

   # 物理 → 无量纲
   dim = system.physical_to_dimensionless(physical_state)

星历系统
--------

:class:`~e2m2e.core.ephemeris_system.EphemerisSystem` 基于 SPICE 内核查询天体星历，
采用 J2000 惯性坐标系，物理单位（km, s, km/s）。详见 :doc:`ephemeris`。

.. code-block:: python

   from e2m2e.core import EphemerisSystem, ReferenceFrame, SPICEManager

   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice,
       origin="EARTH",
       frame=ReferenceFrame.J2000,
   )

CR3BP 与星历系统的区别
-----------------------

.. list-table::
   :header-rows: 1

   * - 特性
     - CR3BP_System
     - EphemerisSystem
   * - 坐标系
     - 旋转坐标系（无量纲）
     - 惯性坐标系（J2000，有量纲）
   * - 单位
     - 无量纲（DU, TU）
     - 物理单位（km, s）
   * - 天体数量
     - 2 个主天体
     - N 个天体（可配置）
   * - 平动点
     - 有（L1–L5）
     - 无
   * - 自治性
     - 自治（不依赖时间）
     - 非自治（依赖历元）
   * - 适用场景
     - 概念设计、周期轨道族
     - 高精度任务设计
