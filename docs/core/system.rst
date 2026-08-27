Systems / 系统
==============

[English](#systems) | [简体中文](#中文)

English
-------

The system layer defines celestial geometry, gravity, and kinematics models —
the context for all subsequent computation.

The System base class
~~~~~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.system.System` is the abstract base defining
three questions a system must answer:

1. Which coordinate frame? Identified by ``frame``.
2. Which units? Decided by ``unit_system``.
3. Gravitational parameters of bodies? Queried via
   ``gravitational_parameter(body)``.

``coordinate_system`` is not in the base interface; only ``EphemerisSystem``
optionally holds one (for frame conversion and ForceModel propagation).

CR3BP system
~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.CR3BP_System` describes the Circular
Restricted Three-Body Problem: two primaries revolve around their common
barycenter in circles; the third body's mass is negligible.

**Create a system:**

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.data.constants import Datum

   system = CR3BP_System(
       mu=Datum.DE421.mu,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

For the Earth-Moon branch ``_with_default_scales()`` applies the DE421 self-
consistent datum: distance = ``Datum.DE421.char_length_km`` (384400 km), period
= ``2π × Datum.DE421.char_time_s`` (TU ≈ 375190 s ≈ 27.28 days). Built-in
defaults cover Earth-Moon, Sun-Earth, Sun-Jupiter; other combos raise
``ValueError`` and need explicit ``set_characteristic_scales()``.

**Mass parameter μ:**

.. math::

   \mu = \frac{m_2}{m_1 + m_2}

- Earth-Moon: μ ≈ 0.01215
- Sun-Earth: μ ≈ 3.0039×10⁻⁶
- Sun-Jupiter: μ ≈ 9.535×10⁻⁴

**Libration points:**

.. code-block:: python

   system.compute_libration_points()

   print(system.L1)  # L1 coordinates [x, 0, 0]
   print(system.L2)
   print(system.L3)
   print(system.L4)  # triangular points
   print(system.L5)

Of the five libration points, L1/L2/L3 are collinear (on the x axis); L4/L5 are
triangular.

**Jacobi constant:**

.. math::

   C_J = 2\Omega - v^2

where Ω is the pseudo-potential and speed = ‖v‖. The Jacobi constant is CR3BP's
only integral of motion.

.. code-block:: python

   state = [0.8, 0, 0, 0, 0.6, 0]
   CJ = system.get_jacobi_constant(state)
   print(f"C_J = {CJ}")

**Unit conversion:**

.. code-block:: python

   # Nondimensional → physical
   phys = system.dimensionless_to_physical(dimensionless_state)

   # Physical → nondimensional
   dim = system.physical_to_dimensionless(physical_state)

BCR4BP system
~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.BCR4BPSystem` describes the Bicircular
Restricted Four-Body Problem: solar point-mass perturbation added atop the
Earth-Moon synodic rotating frame of CR3BP. Under the bicircular approximation,
Earth-Moon revolves circularly about its barycenter (the CR3BP assumption) while
the Sun also moves on a coplanar circle about the barycenter in the synodic
frame — an analytic function of time requiring no ephemeris:

.. math::

   \mathbf{r}_s(t) = a_s (\cos\theta,\ \sin\theta,\ 0),\quad
   \theta = \theta_0 + \omega_s t

ω_s = n_s − 1 < 0: n_s is the Sun's nondimensional inertial revolution rate;
subtracting the synodic frame's own rate 1 yields the Sun's retrograde rate in
the synodic frame. The system is time-periodic with period ``T = 2π/|ω_s|``,
about one synodic month.

**Create:**

.. code-block:: python

   from e2m2e.algorithm.dynamics import BCR4BPSystem

   system = BCR4BPSystem.earth_moon(sun_phase0=0.0)

   print(system.sun_position(0.0))        # Sun position at t=0 (nondimensional)
   print(system.gravitational_parameter("sun"))   # Solar nondimensional mass m_s

The two systems' Earth-Moon scale datums differ: ``earth_moon()`` uses constant
``EARTH_MOON_DISTANCE_KM`` (DU = 384405 km) with period 27.32 days;
``CR3BP_System._with_default_scales``'s Earth-Moon branch uses DE421 self-
consistent values (DU = 384400 km, period ≈ 27.28 days). ``earth_moon()`` takes
standard sun parameters:

- m_s = GM_sun / GM_EMB ≈ 328900.56 (both GMs from DE440)
- a_s = mean Earth-Sun distance / Earth-Moon distance ≈ 389.17 (distance per GMAT nominalSun)
- ω_s = 27.32/365.25 − 1 ≈ −0.9252 (Julian-year derivation; negative = retrograde)

**Differences from CR3BP:** BCR4BP has no Jacobi integral (explicitly time-
dependent solar terms); ``compute_libration_points`` returns the corresponding
CR3BP's points as reference positions only. Companion dynamics:
:class:`~e2m2e.algorithm.dynamics.BCR4BP_Dynamics`, see :doc:`dynamics`. On
accuracy, the bicircular approximation vs ephemeris (Earth+Moon+Sun point
masses) diverges ~1e3 km over one day of extrapolation, dominated by the Moon's
circular-orbit approximation.

Ephemeris system
~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_system.EphemerisSystem` queries body
ephemerides via SPICE kernels, using J2000 inertial frames and physical units
(km, s, km/s). See :doc:`ephemeris`.

.. code-block:: python

   from e2m2e.algorithm.dynamics import EphemerisSystem
   from e2m2e.data.templates.enums import ReferenceFrame
   from e2m2e.data.kernels.manager import SPICEManager

   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice,
       origin="EARTH",
       frame=ReferenceFrame.J2000,
   )

CR3BP vs Ephemeris systems
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Property
     - CR3BP_System
     - EphemerisSystem
   * - Frame
     - Rotating (nondimensional)
     - Inertial (J2000, dimensional)
   * - Units
     - Nondimensional (DU, TU)
     - Physical (km, s)
   * - Bodies
     - 2 primaries
     - N bodies (configurable)
   * - Libration points
     - Yes (L1–L5)
     - No
   * - Autonomy
     - Autonomous (time-free)
     - Non-autonomous (epoch-dependent)
   * - Use case
     - Concept design, periodic families
     - High-fidelity mission design

中文
----

系统层定义天体几何、引力与运动学模型，是后续所有计算的上下文。

System 基类
~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.system.System` 是抽象基类，定义系统要回答的三个问题：

1. 用什么坐标框架？由 ``frame`` 标识。
2. 数值用什么单位？由 ``unit_system`` 决定。
3. 各天体的引力参数？通过 ``gravitational_parameter(body)`` 查询。

``coordinate_system`` 不在基类接口中，仅 ``EphemerisSystem`` 可选持有
（供坐标变换与 ForceModel 传播使用）。

CR3BP 系统
~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.CR3BP_System` 描述圆型限制性三体问题：
两个主天体绕公共质心作圆周运动，第三体质量可忽略。

**创建系统：**

.. code-block:: python

   from e2m2e.algorithm.dynamics import CR3BP_System
   from e2m2e.data.constants import Datum

   system = CR3BP_System(
       mu=Datum.DE421.mu,
       primary="Earth",
       secondary="Moon",
   )._with_default_scales()

``_with_default_scales()`` 对地月分支采用 DE421 自洽基准：距离取
``Datum.DE421.char_length_km`` （384400 km），周期取
``2π × Datum.DE421.char_time_s`` （TU ≈ 375190 s，约 27.28 天）。
内置 Earth-Moon、Sun-Earth、Sun-Jupiter 三组默认尺度；其他组合抛
``ValueError`` ，需用 ``set_characteristic_scales()`` 显式设置。

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

BCR4BP 系统
~~~~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.BCR4BPSystem` 描述双圆限制性四体问题
（Bicircular Restricted Four-Body Problem）：在 CR3BP 的地月会合旋转系
之上叠加太阳质点摄动。双圆近似下，地月绕公共质心作圆周运动（CR3BP
假设），太阳也在会合系中绕质心作共面圆周运动，其位置是时间的解析函数，
无需星历：

.. math::

   \mathbf{r}_s(t) = a_s (\cos\theta,\ \sin\theta,\ 0),\quad
   \theta = \theta_0 + \omega_s t

ω_s = n_s − 1 < 0：n_s 为太阳公转的无量纲角速度（惯性系），减去会合系
自身角速度 1 后是太阳在会合系中的逆行角速度。系统是时间周期的，
周期 ``T = 2π/|ω_s|`` ，约一个会合月。

**创建系统：**

.. code-block:: python

   from e2m2e.algorithm.dynamics import BCR4BPSystem

   system = BCR4BPSystem.earth_moon(sun_phase0=0.0)

   print(system.sun_position(0.0))        # t=0 时刻太阳位置（无量纲）
   print(system.gravitational_parameter("sun"))   # 太阳无量纲质量 m_s

两套系统的地月特征尺度基准不同：``earth_moon()`` 使用常量
``EARTH_MOON_DISTANCE_KM`` （DU = 384405 km），周期取 27.32 天；
``CR3BP_System._with_default_scales`` 的地月分支使用 DE421 自洽值
（DU = 384400 km，周期约 27.28 天）。``earth_moon()`` 的太阳参数取标准值：

- m_s = GM_sun / GM_EMB ≈ 328900.56（太阳与地月质心 GM 均取 DE440）
- a_s = 日地平均距离 / 地月距离 ≈ 389.17（日地距离取 GMAT nominalSun）
- ω_s = 27.32/365.25 − 1 ≈ −0.9252（按儒略年推导，负号表示逆行）

**与 CR3BP 的区别：** BCR4BP 无 Jacobi 积分（太阳项显式含时）；
``compute_libration_points`` 给出的是对应 CR3BP 的平动点，仅作参考
位置。配套动力学为 :class:`~e2m2e.algorithm.dynamics.BCR4BP_Dynamics`，
见 :doc:`dynamics` 的 BCR4BP 动力学一节。精度上，双圆近似与星历
（地+月+日点质量）对比 1 天外推位置误差在 1e3 km 量级，主误差来自
月球圆轨道近似。

星历系统
~~~~~~~~

:class:`~e2m2e.algorithm.dynamics.ephemeris_system.EphemerisSystem` 基于 SPICE 内核查询天体星历，
采用 J2000 惯性坐标系，物理单位（km, s, km/s）。详见 :doc:`ephemeris`。

.. code-block:: python

   from e2m2e.algorithm.dynamics import EphemerisSystem
   from e2m2e.data.templates.enums import ReferenceFrame
   from e2m2e.data.kernels.manager import SPICEManager

   spice = SPICEManager()
   spice.load_kernel("kernels/de440s.bsp")

   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice,
       origin="EARTH",
       frame=ReferenceFrame.J2000,
   )

CR3BP 与星历系统的区别
~~~~~~~~~~~~~~~~~~~~~~~

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
     - 有（L1~L5）
     - 无
   * - 自治性
     - 自治（不依赖时间）
     - 非自治（依赖历元）
   * - 适用场景
     - 概念设计、周期轨道族
     - 高精度任务设计
