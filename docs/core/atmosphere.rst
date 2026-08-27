Atmosphere Models / 大气密度模型
================================

[English](#english) | [简体中文](#中文)

English
-------

e2m2e provides pluggable atmosphere density models, dependency-injected into
``DragModel``. The current implementation is a US Standard Atmosphere 1976
piecewise-exponential model covering 0–1000 km altitude.

**ExponentialAtmosphere**: USSA76 piecewise-exponential densities via
``density(altitude)``, with first-order F10.7 / Ap corrections.

Within each layer ``ρ(h) = ρ₀ · exp(-(h - h₀) / H)``; layer scale heights derive
from adjacent breakpoint density ratios — continuous and monotonically
decreasing. F10.7 (solar radio flux) and Ap (geomagnetic index) scale the base
density linearly.

.. code-block:: python

   from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

   # Defaults: F10.7=150 sfu (moderate solar), Ap=15 (moderate geomagnetic)
   atm = ExponentialAtmosphere()

   rho_surface = atm.density(0.0)      # 1.225 kg/m³
   rho_100km   = atm.density(100.0)    # 5.604e-7 kg/m³
   rho_400km   = atm.density(400.0)    # 2.803e-12 kg/m³
   rho_1000km  = atm.density(1000.0)   # 0.0 (beyond model ceiling)

   atm_high = ExponentialAtmosphere(f107=200, ap=50)
   rho_high = atm_high.density(400.0)  # higher than defaults

中文
----

e2m2e 提供可插拔的大气密度模型，供阻力力模型 ``DragModel`` 依赖注入使用。
当前实现基于 US Standard Atmosphere 1976 分段指数模型，覆盖 0-1000 km 高度范围。

核心概念
--------

- **ExponentialAtmosphere**：USSA76 分段指数大气密度模型，提供 ``density(altitude)`` 接口，支持 F10.7 / Ap 一阶修正。

ExponentialAtmosphere
---------------------

US Standard Atmosphere 1976 分段指数大气密度模型。

在每个高度层内使用 ``ρ(h) = ρ₀ · exp(-(h - h₀) / H)`` 计算密度，
层间标高由相邻断点密度比推导，确保密度连续且单调递减。
F10.7 太阳射电通量和 Ap 地磁指数通过线性乘法因子对基准密度做一阶修正。

.. code-block:: python

   from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

   # 默认参数：F10.7=150 sfu（中等太阳活动），Ap=15（中等地磁活动）
   atm = ExponentialAtmosphere()

   # 查询不同高度的密度
   rho_surface = atm.density(0.0)      # 1.225 kg/m³
   rho_100km   = atm.density(100.0)    # 5.604e-7 kg/m³
   rho_400km   = atm.density(400.0)    # 2.803e-12 kg/m³
   rho_1000km  = atm.density(1000.0)   # 0.0（超出模型上限）

   # 调整太阳活动参数
   atm_high = ExponentialAtmosphere(f107=200, ap=50)
   rho_high = atm_high.density(400.0)  # 密度高于默认值

模型范围与边界行为
-------------------

- **高度范围**：0 km 至 1000 km。
- **高于 1000 km**：返回 0.0（阻力可忽略）。
- **低于 0 km**：钳到 0 km，返回地表密度（避免负高度导致指数爆炸）。

参数说明
--------

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
     - 默认值
   * - ``f107``
     - F10.7 太阳射电通量（sfu），反映太阳极紫外辐射强度
     - 150.0
   * - ``ap``
     - Ap 地磁指数，反映地磁活动强度
     - 15.0

F10.7 和 Ap 越高，大气热膨胀越显著，同高度密度越大。
模型对两者的修正为线性乘法因子：

.. math::

   \text{factor} = \left(1 + 0.5 \cdot \frac{f107 - 150}{150}\right)
                 \cdot \left(1 + 0.1 \cdot \frac{ap - 15}{15}\right)

与 DragModel 配合：LEO 轨道衰减分析
------------------------------------

完整工作流示例：选择指数大气模型、配置 ``DragModel`` 、设置初始状态、运行轨道衰减传播。要点（完整代码见原文历史或英文节扩展）：

.. code-block:: python

   atmosphere = ExponentialAtmosphere(f107=150.0, ap=4.0)
   drag = DragModel(atmosphere=atmosphere, body="EARTH", cd=2.2, area=10.0, mass=1000.0)
   fm = ForceModel(system)
   fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
   fm.add_force(drag, name="drag")
   # LEO 400 km 圆轨道初值后传播一天，分析半长轴随时间衰减
