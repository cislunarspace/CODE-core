力模型
======

e2m2e 的力模型子包提供可配置、可组合的航天器摄动力模型，支持从配置字典构建、序列化到 JSON，并与星历动力学系统配合完成轨道传播。

核心概念
--------

- **PhysicalModel**：所有力模型的抽象基类，定义 ``compute_acceleration(t, state, system)`` 接口。
- **ForceModel**：容器类，聚合多个 ``PhysicalModel``，按名注册、启用/禁用，并通过 Rust ``rk_step`` 步进器完成传播。
- **ForceEntry**：容器内单个力模型的注册记录，包含 ``name``、``force`` 和 ``enabled`` 三个字段。

支持力模型类型
--------------
   * - 类型
     - 说明
     - 配置 type 名
   * - GravityField
     - 球谐重力场（支持 EGM96 及自定义 .gfc）
     - ``GravityField``
   * - DragModel
     - 大气阻力（依赖注入大气密度模型）
     - ``DragModel``
   * - SolarRadiationPressure
     - 太阳光压（cannonball + 可选阴影模型）
     - ``SolarRadiationPressure``
   * - FiniteBurn
     - 连续推力（封闭 DSL：constant/pulse 推力 + fixed 方向）
     - ``FiniteBurn``

配置 Schema
-----------

顶层配置字典结构：

.. code-block:: python

   {
       "version": 1,
       "forces": [
           {
               "name": "j2",
               "type": "GravityField",
               "enabled": True,
               "params": {
                   "body": "EARTH",
                   "degree": 2,
                   "order": 0,
               },
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
           {
               "name": "srp",
               "type": "SolarRadiationPressure",
               "enabled": True,
               "params": {
                   "area": 5.0,
                   "mass": 1000.0,
                   "cr": 1.5,
                   "shadow": None,  # 全光照
               },
           },
           {
               "name": "burn",
               "type": "FiniteBurn",
               "enabled": False,
               "params": {
                   "mass": 1000.0,
                   "thrust_profile": {
                       "kind": "constant",
                       "thrust": 0.5,
                   },
                   "direction": {
                       "kind": "fixed",
                       "vector": [1.0, 0.0, 0.0],
                   },
               },
           },
       ],
   }

关键字段说明：

- ``version``：当前固定为 ``1``，用于日后 schema 迁移。
- ``forces``：力模型条目数组，顺序即传播时的叠加顺序。
- ``name``：容器内唯一标识，用于 ``get_force`` / ``enable`` / ``disable`` / ``remove_force``。
- ``type``：Python 类名，也是 ``force_config`` 注册表的 key。
- ``enabled``：布尔开关；``False`` 时传播跳过该力，但保留在容器内。
- ``params``：构造参数。嵌套依赖（如 ``atmosphere``、``shadow``）用 ``{"type": ..., "params": ...}`` 递归表达；``null`` 表示未注入。

name-based 注册机制
-------------------

.. code-block:: python

   from e2m2e.core.forces import ForceModel, GravityField, DragModel
   from e2m2e.core.atmosphere import ExponentialAtmosphere

   fm = ForceModel(system)

   # 省略 name 时自动按类名生成，遇同类自动消歧
   fm.add_force(GravityField("EARTH", degree=2, order=0))
   fm.add_force(GravityField("EARTH", degree=4, order=4))   # 自动命名为 GravityField_2

   # 显式命名
   fm.add_force(DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0), name="drag")

   # 按名操作
   fm.disable("drag")          # 暂时关闭阻力
   fm.enable("drag")         # 重新打开
   fm.get_force("drag")      # 返回 DragModel 实例
   fm.remove_force("drag")   # 从容器中移除

   # 列出所有注册记录
   for entry in fm.list_forces():
       print(f"{entry.name}: {type(entry.force).__name__}, enabled={entry.enabled}")

配置驱动构建完整流程
--------------------

以下示例展示从配置字典构建 ``ForceModel``，接入 ``EphemerisDynamics`` 完成一次 LEO 轨道传播。
   from e2m2e.core.standard_axes import ICRSAxes
   from e2m2e.core.standard_origins import CelestialBodyOrigin
   from e2m2e.core.forces import ForceModel

   # 1. 准备星历系统（ICRF + 地球中心）
   spice = SPICEManager()
   spice.load_kernel("path/to/de440.bsp")

   system = EphemerisSystem(bodies=["EARTH"], spice=spice, origin="EARTH")
   system.coordinate_system = CoordinateSystem(
       axes=ICRSAxes(),
       origin=CelestialBodyOrigin(body="EARTH", spice=spice),
   )

   # 2. 定义配置字典
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

   # 3. 从配置构建 ForceModel
   fm = ForceModel.from_config(config, system)

   # 4. 设置初始状态（400 km 圆轨道，km / km/s）
   r = 6378.137 + 400.0
   v = np.sqrt(398600.4415 / r)
   state0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

   # 5. 传播 10 分钟
   et0 = spice.utc_to_et("2025-06-21T11:00:06")
   t_span = (et0, et0 + 600.0)
   t_eval = np.linspace(et0, et0 + 600.0, 50)

   result = fm.propagate(state0, t_span, t_eval=t_eval, max_steps=200_000)

   print(f"states shape: {result['states'].shape}")   # (50, 6)
   print(f"time span: {result['time'][0]} -> {result['time'][-1]}")

   # 6. 序列化回配置（验证 round-trip）
   config_roundtrip = fm.to_config()
   assert config_roundtrip == config   # 字典完全相等

   # 7. 存盘 / 读回
   from e2m2e.core.forces import dump_force_config, load_force_config

   dump_force_config(fm, "leo_forces.json")
   fm_loaded = load_force_config("leo_forces.json", system)

JSON 文件 IO
------------

.. code-block:: python

   from e2m2e.core.forces import dump_force_config, load_force_config

   # 写入 JSON
   dump_force_config(fm, "forces.json")

   # 从 JSON 读回
   fm2 = load_force_config("forces.json", system)

常见错误与排错
--------------

**未注册名称**

.. code-block:: python

   fm.get_force("nonexistent")   # KeyError: 'nonexistent'
   fm.disable("nonexistent")      # KeyError: 'nonexistent'

**显式命名冲突**

.. code-block:: python

   fm.add_force(GravityField("EARTH"), name="primary")
   fm.add_force(GravityField("EARTH"), name="primary")  # ValueError: force name 'primary' already exists

**配置版本不匹配**

.. code-block:: python

   ForceModel.from_config({"version": 2, "forces": []}, system)
   # ValueError: unsupported config version 2; expected 1

**未知力类型**

.. code-block:: python

   build_force("UnknownType", {})
   # ValueError: unknown force type 'UnknownType'; known types: ['DragModel', 'FiniteBurn', 'GravityField', 'SolarRadiationPressure']

**不可序列化的推力剖面**

``FiniteBurn`` 若使用用户手写的 ``lambda`` 作为 ``thrust_profile``，仍可正常传播，但 ``to_config()`` 会抛出 ``NotSerializableError``。解决方式：改用 ``force_config`` 提供的 DSL（``{"kind": "constant"}`` 或 ``{"kind": "pulse"}``）构造推力剖面。
.. automodule:: e2m2e.core.forces.force_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: e2m2e.core.forces.force_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: e2m2e.core.forces.physical_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: e2m2e.core.forces.gravity_field
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: e2m2e.core.forces.drag
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: e2m2e.core.forces.thrust
   :members:
   :undoc-members:
   :show-inheritance:

太阳辐射压与阴影模型

e2m2e 提供基于 cannonball 模型的太阳辐射压（SRP）力模型，以及圆锥阴影模型用于计算地影/月影对光照的遮挡效应。两者均通过 ``PhysicalModel`` 接口与 ``ForceModel`` 容器集成，支持配置驱动的序列化与反序列化。

太阳辐射压模型
--------------

:class:`~e2m2e.core.forces.srp.SolarRadiationPressure` 实现 Montenbruck & Gill 的 cannonball SRP 模型：

.. math::

   \mathbf{a} = \text{flux} \cdot P_{1\text{AU}} \left(\frac{1\ \text{AU}}{r}\right)^2 \frac{C_R \, A}{m} \, \hat{\mathbf{u}}

其中 ``P_1AU = 4.56e-6 N/m²`` 为 1 AU 处太阳光压常数，``r`` 为航天器到太阳的距离，
``C_R`` 为辐射反射系数（1 = 全吸收，2 = 全反射），``A`` 为迎风截面积（m²），
``m`` 为质量（kg）。``flux ∈ [0, 1]`` 由阴影模型给出，全光照为 1，本影为 0。

**参数说明：**
   * - 参数
     - 含义
     - 默认值
   * - ``area``
     - 航天器迎风截面积（m²）
     - 必填
   * - ``mass``
     - 航天器质量（kg）
     - 必填
   * - ``cr``
     - 辐射反射系数 C_R
     - 1.5
   * - ``shadow``
     - 阴影模型实例（注入）
     - ``None``（全光照）

阴影模型
--------

:class:`~e2m2e.core.forces.shadow.ShadowModel` 是抽象基类，定义 ``flux_factor(t, state, system) -> float`` 接口。当前实现为圆锥阴影模型 :class:`~e2m2e.core.forces.shadow.ConicalShadowModel`。

圆锥阴影模型
^^^^^^^^^^^^

实现 GMAT ``ShadowState`` 的圆锥阴影算法（Montenbruck & Gill §3.4.2），
从航天器看太阳与遮挡体的视角径 (a, b) 与角距 c，分四分支判定：

- **全光照**：遮挡体与太阳圆盘不相交
- **本影**：遮挡体完全遮住太阳圆盘（flux = 0）
- **半影**：部分重叠，用 M&G eq. 3.92-3.94 精确圆面重叠面积计算
- **环形食**：遮挡体小于太阳，中心对齐但边缘透光

多遮挡体（如地球 + 月球）的光照份额合成遵循 GMAT GMT-6543 规范：
任一遮挡体本影 → 0；两体部分阴影且不重叠 → 包容排斥；重叠 → 保守取最小值。

**参数说明：**

.. list-table::
   :header-rows: 1

   * - 参数
     - 含义
     - 默认值
   * - ``bodies``
     - 遮挡体名称列表（大写）
     - ``("EARTH",)``
   * - ``radii``
     - 天体半径覆盖字典（km）
     - ``None``（使用内置默认值）

内置默认半径：

.. list-table::
   :header-rows: 1

   * - 天体
     - 半径（km）
   * - EARTH
     - 6378.1363
   * - MOON
     - 1737.4
   * - SUN
     - 695700.0

配置与序列化
------------

SRP 与阴影模型支持通过 :mod:`~e2m2e.core.forces.force_config` 进行配置驱动的序列化。

**配置字典格式：**

.. code-block:: python

   {
       "type": "SolarRadiationPressure",
       "params": {
           "area": 2.0,
           "mass": 1000.0,
           "cr": 1.5,
           "shadow": {
               "type": "ConicalShadowModel",
               "params": {
                   "bodies": ["EARTH", "MOON"],
                   "radii": None
               }
           }
       }
   }

完整工作流示例
--------------

以下示例展示 SRP + 地影/月影 + ``EphemerisDynamics`` 的完整传播流程：
   from e2m2e.core.standard_axes import J2000Axes
   from e2m2e.core.standard_origins import CelestialBodyOrigin
   from e2m2e.core.forces import (
       SolarRadiationPressure,
       ConicalShadowModel,
       ForceModel,
   )

   # 1. 加载 SPICE 内核
   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   # 2. 构建星历系统（含地球、月球、太阳）
   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=mgr,
       origin="EARTH",
       frame="J2000",
   )

   # 3. 设置坐标系（J2000 惯性系，力模型要求惯性系）
   axes = J2000Axes()
   origin = CelestialBodyOrigin(body="EARTH", spice=mgr)
   system.coordinate_system = CoordinateSystem(axes=axes, origin=origin)

   # 4. 创建阴影模型（地影 + 月影）
   shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])

   # 5. 创建 SRP 力模型
   srp = SolarRadiationPressure(
       area=2.0,      # m²
       mass=1000.0,   # kg
       cr=1.5,
       shadow=shadow,
   )

   # 6. 组装 ForceModel
   fm = ForceModel(system)
   fm.add_force(srp, name="SRP")

   # 7. 初始状态：LEO 近似圆轨道
   r0 = 6678.0  # km（约 300 km 高度）
   v0 = np.sqrt(398600.435507 / r0)  # km/s
   state0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])

   # 8. 传播（1 个轨道周期，约 90 分钟）
   from e2m2e.core.spice import SPICEManager
   et0 = mgr.utc_to_et("2024-06-21T00:00:00")
   period = 2 * np.pi * r0 / v0  # ~5400 s
   result = fm.propagate(state0, (et0, et0 + period))

   print(f"传播点数: {len(result['time'])}")
   print(f"末状态: {result['states'][-1]}")

   # 9. 序列化配置到 JSON
   from e2m2e.core.forces import dump_force_config
   dump_force_config(fm, "srp_config.json")

   # 10. 从 JSON 恢复
   from e2m2e.core.forces import load_force_config
   fm2 = load_force_config("srp_config.json", system)

   # 验证 round-trip
   assert fm2.to_config() == fm.to_config()

   # 11. 启用/禁用 SRP 对比
   fm.disable("SRP")
   result_no_srp = fm.propagate(state0, (et0, et0 + period))

   fm.enable("SRP")
   result_with_srp = fm.propagate(state0, (et0, et0 + period))

   # 对比末位置差异
   diff = np.linalg.norm(result_with_srp["states"][-1, :3]
                        - result_no_srp["states"][-1, :3])
   print(f"SRP 引起的 1 周期位置差异: {diff:.3f} km")

纯函数测试路径
--------------

SRP 和阴影模型均提供纯函数接口，可在无 SPICE 环境下直接测试：

.. code-block:: python

   import numpy as np
   from e2m2e.core.forces.srp import SolarRadiationPressure
   from e2m2e.core.forces.shadow import ConicalShadowModel

   # SRP 纯函数测试
   srp = SolarRadiationPressure(area=2.0, mass=1000.0, cr=1.5)
   sun_to_sc = np.array([1.0, 0.0, 0.0]) * 149597870.691  # 1 AU
   accel = srp._compute_srp_acceleration(sun_to_sc, flux_factor=1.0)
   print(f"1 AU 处全光照 SRP 加速度: {accel} km/s²")

   # 阴影模型纯函数测试
   shadow = ConicalShadowModel()
   sc_pos = np.array([7000.0, 0.0, 0.0])
   body_pos = np.array([0.0, 0.0, 0.0])
   sun_pos = np.array([1.5e8, 0.0, 0.0])
   flux = shadow._body_flux_factor(
       sc_pos, body_pos, sun_pos,
       body_radius=6378.1363, sun_radius=695700.0
   )
   print(f"地影光照份额: {flux}")
