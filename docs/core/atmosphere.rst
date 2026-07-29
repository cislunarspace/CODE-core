大气密度模型
============

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

   from e2m2e.core.atmosphere import ExponentialAtmosphere

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
------------------

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
-----------------------------------

以下示例展示 LEO 轨道衰减分析工作流：
选择指数大气模型、配置 ``DragModel``、设置初始状态、运行轨道衰减传播。

.. code-block:: python

   import numpy as np
   from e2m2e.core.spice import SPICEManager
   from e2m2e.core.ephemeris_system import EphemerisSystem
   from e2m2e.core.coordinate_system import CoordinateSystem
   from e2m2e.core.standard_axes import ICRSAxes
   from e2m2e.core.standard_origins import CelestialBodyOrigin
   from e2m2e.core.atmosphere import ExponentialAtmosphere
   from e2m2e.core.forces import ForceModel, GravityField, DragModel

   # 1. 准备星历系统（ICRF + 地球中心）
   spice = SPICEManager()
   spice.load_kernel("path/to/de440.bsp")

   system = EphemerisSystem(bodies=["EARTH"], spice=spice, origin="EARTH")
   system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

   # 2. 选择大气模型并配置阻力
   atmosphere = ExponentialAtmosphere(f107=150.0, ap=4.0)
   drag = DragModel(
       atmosphere=atmosphere,
       body="EARTH",
       cd=2.2,        # 阻力系数
       area=10.0,     # 迎风面积 m²
       mass=1000.0,   # 航天器质量 kg
   )

   # 3. 构建力模型：J2 重力 + 大气阻力
   fm = ForceModel(system)
   fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
   fm.add_force(drag, name="drag")

   # 4. 设置 LEO 初始状态（400 km 圆轨道，km / km/s）
   r = 6378.137 + 400.0
   v = np.sqrt(398600.4415 / r)
   state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

   # 5. 传播 1 天，观察轨道衰减
   et0 = spice.utc_to_et("2025-06-21T11:00:06")
   t_span = (et0, et0 + 86400.0)
   t_eval = np.linspace(et0, et0 + 86400.0, 200)

   result = fm.propagate(state0, t_span, t_eval=t_eval, max_steps=200_000)

   # 6. 分析结果：半长轴随时间衰减
   states = result["states"]
   times = result["time"]

   def semi_major_axis(state, mu=398600.4415):
       r = np.linalg.norm(state[:3])
       v = np.linalg.norm(state[3:6])
       energy = v**2 / 2.0 - mu / r
       return -mu / (2.0 * energy)

   a_history = np.array([semi_major_axis(s) for s in states])
   delta_a = a_history[-1] - a_history[0]

   print(f"初始半长轴: {a_history[0]:.2f} km")
   print(f"1 天后半长轴: {a_history[-1]:.2f} km")
   print(f"半长轴衰减: {delta_a:.4f} km")

配置驱动方式（推荐）
--------------------

上述工作流也可通过配置字典一键构建，便于序列化与复用：

.. code-block:: python

   config = {
       "version": 1,
       "forces": [
           {
               "name": "j2",
               "type": "GravityField",
               "enabled": True,
               "params": {"body": "EARTH", "degree": 2, "order": 0},
           },
           {
               "name": "drag",
               "type": "DragModel",
               "enabled": True,
               "params": {
                   "body": "EARTH",
                   "cd": 2.2,
                   "area": 10.0,
                   "mass": 1000.0,
                   "atmosphere": {
                       "type": "ExponentialAtmosphere",
                       "params": {"f107": 150.0, "ap": 4.0},
                   },
               },
           },
       ],
   }

   fm = ForceModel.from_config(config, system)

   # round-trip 验证
   assert fm.to_config() == config

