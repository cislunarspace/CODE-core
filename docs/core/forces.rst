力模型
======

e2m2e 的力模型子包提供可配置、可组合的航天器摄动力模型，支持从配置字典构建、序列化到 JSON，并与星历动力学系统配合完成轨道传播。

核心概念
--------

这三个概念构成力模型子包的核心接口：

- **PhysicalModel**：所有力模型的抽象基类，定义 ``compute_acceleration(t, state, system)`` 接口。另有两个可选钩子：``compute_jacobian(t, state, system)`` 返回解析雅可比 ∂a/∂r（默认返回 ``None``，由 ``ForceModel`` 有限差分兜底）；``to_rust_spec(system)`` 把力序列化为 Rust 编译路径接受的元组（默认返回 ``None``，表示该力不支持 Rust 编译）。
- **ForceModel**：力模型组合，把多个 ``PhysicalModel`` 组合成一次传播所需的运动方程；按名登记、启用/禁用，并通过 Rust 积分器完成传播。
- **ForceEntry**：力模型组合内单个力模型的登记记录，包含 ``name``、``force`` 和 ``enabled`` 三个字段。

支持力模型类型
--------------

.. list-table::
   :header-rows: 1

   * - 类型
     - 说明
     - 配置 type 名
   * - PointMassGravity
     - 中心天体点质量引力
     - ``PointMassGravity``
   * - ThirdBodyGravity
     - 第三体引力摄动（含间接项）
     - ``ThirdBodyGravity``
   * - IndirectTerm
     - 单独的间接项修正
     - ``IndirectTerm``
   * - GravityField
     - 球谐重力场（天体无关，支持 EGM96/GRGM900C 及自定义 .gfc/.cof）
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
   * - RelativisticCorrection
     - 后牛顿相对论修正（Schwarzschild / Lense-Thirring / de Sitter）
     - ``RelativisticCorrection``

另有机动事件 ``ImpulsiveBurn``（瞬时 Δv，在指定 epoch 直接修改状态速度），
它不是 ``PhysicalModel``，不参与加速度叠加，也不进配置注册表；施加方式见
下文「传播接口」小节的 ``propagate_maneuvers``。

.. warning::

   用 ``GravityField`` 模拟月球（含 degree=0 中心项）时，必须单独补月球间接项
   ``IndirectTerm("MOON")``，且不能再加 ``ThirdBodyGravity("MOON")``——后者会与
   ``GravityField`` 的中心项重复计算月球点质量。

内置力模型公式
--------------

**PointMassGravity** — 中心天体点质量引力。适用于参考系原点天体自身的二体引力：

.. math::

   \mathbf{a} = -\frac{\mu}{|\mathbf{r}|^3} \, \mathbf{r}

其中 ``μ`` 为引力参数（km³/s²），``r`` 为航天器相对中心天体的位置。
它不查任何第三体的位置，只能表达参考系原点天体自身的引力；其他天体的
引力贡献用 ThirdBodyGravity。

**ThirdBodyGravity** — 参考系原点之外的天体引力摄动。一个实例对应一个摄动
天体（如 ``ThirdBodyGravity("MOON")``）。加速度由直接项与间接项合成：

.. math::

   \mathbf{a} = -\mu_i \left[ \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3} + \frac{\mathbf{r}_i}{|\mathbf{r}_i|^3} \right]

其中 ``r`` 为航天器相对原点的位置，``r_i`` 为摄动天体相对原点的位置（由
SPICE 查询）。间接项扣除摄动天体对原点的引力，保持坐标原点固定。

**GravityField** — 完全正规化球谐重力场（Cnm/Snm），用 Pines 递推计算非球形
引力加速度。天体无关：地球（EGM96）、月球（GRGM900C）等共用同一个类，按
``body`` 参数自动切换 body-fixed 轴与系数文件。位势展开为：

.. math::

   U = \frac{\mu}{r} \sum_{n=0}^{N} \left(\frac{R}{r}\right)^n \sum_{m=0}^{n} \left(C_{nm}\cos m\lambda + S_{nm}\sin m\lambda\right) \bar{P}_{nm}(\sin\phi)

加速度由 Pines 方法直接递推位势梯度得到，不经过球谐系数的解析微分。内置
固体潮修正：地球支持 Step1（天体无关）+ Step2（频率相关）+ 极潮 + 永久潮；
月球支持 k₂ = 0.024116 Love 数的固体潮。

**DragModel** — 大气阻力。在 ITRF（地固系）中计算密度与相对速度，自动完成
参考系↔ITRF 坐标变换。大气在 ITRF 中静止，相对速度等于航天器 ITRF 速度：

.. math::

   \mathbf{a}_{\text{drag}} = -\frac{1}{2} \, \rho \, \frac{C_d A}{m} \, |\mathbf{v}_{\text{rel}}| \, \mathbf{v}_{\text{rel}}

其中 ``ρ`` 为大气密度（由 ``ExponentialAtmosphere`` 提供，US Standard
Atmosphere 1976 分段指数模型），``C_d`` 为阻力系数（默认 2.2），``A/m`` 为
面积质量比。

**FiniteBurn** — 连续推力加速度力模型。推力大小（``thrust_profile(t)`` →
标量 N）与方向（``direction``）解耦，方向支持传播惯性系、VNB、LVLH 三种
坐标系：

.. math::

   \mathbf{a}_{\text{thrust}} = \frac{T(t)}{m} \, \hat{\mathbf{d}}

其中 ``T(t)`` 为标量推力函数，``d̂`` 为归一化方向向量。配置往返支持固定
推力/脉冲剖面与固定方向的封闭 DSL；任意 Python callable 可传播但无法序列化。

**RelativisticCorrection** — 后牛顿相对论修正，含三项，公式与 GMAT 对齐：

- Schwarzschild 项（质量引起的时空弯曲）：

  .. math::

     \mathbf{a}_S = \frac{\gamma \mu}{c^2 r^3} \left[ \left(\frac{4\mu}{r} - v^2\right) \mathbf{r} + 4(\mathbf{r} \cdot \mathbf{v})\mathbf{v} \right]

- Lense-Thirring 项（参考系拖曳）：

  .. math::

     \mathbf{a}_{LT} = \frac{2\mu}{c^2 r^3} \left[ \frac{3}{r^2}(\mathbf{r} \cdot \mathbf{J})(\mathbf{r} \times \mathbf{v}) + \mathbf{v} \times \mathbf{J} \right]

- de Sitter 项（测地进动）：

  .. math::

     \mathbf{a}_{dS} = 2 \, \boldsymbol{\omega} \times \mathbf{v}

其中 ``γ = 1.0`` 为后牛顿参数，``c`` 为光速，``J`` 为天体角动量参数，
``ω`` 为 de Sitter 进动角速度。

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

传播接口
--------

``ForceModel.propagate`` 用 Rust ``rk_step`` 单步步进器做自适应传播：

.. code-block:: python

   from e2m2e.integrators import RkMethod

   result = fm.propagate(
       state0,
       (t0, tf),
       t_eval=t_eval,             # 输出采样点，默认 linspace(t0, tf, 100)
       with_stm=True,             # 同时积分状态转移矩阵
       initial_step=1.0,          # 初始步长，默认从初始状态估算
       events=[...],              # 终止事件列表，见下
       max_steps=200_000,
       method=RkMethod.PD45,      # RK 方法，默认 PD45
   )

``with_stm=True`` 时把 6 维状态与 6×6 STM 展平拼成 42 维增广状态一起积分：
各力的解析雅可比 ``compute_jacobian`` 直接叠加，不提供解析雅可比的力由
``ForceModel`` 用三点中心差分兜底；STM 分量不参与步长误差控制，接受/拒绝
只看前 6 维物理状态（对齐 GMAT）。返回字典在 ``time``、``states``、
``terminal_event_index`` 之外多一个 ``stm`` 键，形状 ``(n_points, 6, 6)``。

``events`` 是终止事件列表，每个事件是 ``f(t, state) -> float`` 的可调用，
函数值符号变化即在该步停止，``terminal_event_index`` 记录触发的事件下标；
``with_stm=True`` 时事件函数接收 6 维物理状态（非增广状态）。

带脉冲机动时用 ``propagate_maneuvers(initial_state, t_span, burns)``：按
``epoch`` 排序 ``ImpulsiveBurn`` 列表，逐段 coast 传播，在每个 burn epoch
处施加 ``state[3:6] += delta_v`` 后续传；返回字典额外含 ``burns`` 键，
记录每次机动的施加位置与前后速度。

.. code-block:: python

   from e2m2e.core.forces import ImpulsiveBurn

   burns = [ImpulsiveBurn(epoch=et0 + 1800.0, delta_v=np.array([0.0, 0.01, 0.0]))]
   result = fm.propagate_maneuvers(state0, (et0, et0 + 7200.0), burns)

Rust 编译快速路径
^^^^^^^^^^^^^^^^^

spice feature 启用、无 ``events``、不带 STM，且所有启用力模型的
``to_rust_spec()`` 都非 ``None`` 时，``propagate`` 自动分流到 Rust
``propagate_compiled``：力模型一次序列化后整个积分循环在 Rust 内完成，
消除逐步 Python↔Rust 跨界；30 天 NRHO 传播约 9.6 s，Python 路径约 95 s。
任一条件不满足时自动回退 Python 路径，返回格式一致，无需用户干预。

带 STM 时有对应的 Rust compiled STM 快速路径（``propagate_compiled_stm_py``），
但 ``SolarRadiationPressure`` 与 ``RelativisticCorrection`` 无解析雅可比，
含这两个力的 STM 传播不走该路径，回退 Python 增广积分。

配置驱动构建流程
------------------

以下示例展示从配置字典构建 ``ForceModel``，接入 ``EphemerisDynamics`` 完成一次 LEO 轨道传播。

.. code-block:: python

   import numpy as np

   from e2m2e.core import (
       CoordinateSystem,
       EphemerisSystem,
       ICRSAxes,
       SPICEManager,
   )
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

   # 6. 序列化回配置，再构建一次（验证 round-trip 契约）
   # 注意：to_config 输出是规范形式（如 GravityField 恒带 input_frame/gravity_file
   # 两键），与手写 config 不一定逐键相等；契约是“再序列化一次字典相等”。
   config_roundtrip = fm.to_config()
   fm2 = ForceModel.from_config(config_roundtrip, system)
   assert fm2.to_config() == config_roundtrip

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

   from e2m2e.core.forces.force_config import build_force

   build_force("UnknownType", {})
   # ValueError: unknown force type 'UnknownType'; known types: ['DragModel', 'FiniteBurn', 'GravityField', 'IndirectTerm', 'PointMassGravity', 'RelativisticCorrection', 'SolarRadiationPressure', 'ThirdBodyGravity']

**不可序列化的推力剖面**

``FiniteBurn`` 若使用用户手写的 ``lambda`` 作为 ``thrust_profile``，仍可正常传播，但 ``to_config()`` 会抛出 ``NotSerializableError``。解决方式：改用 ``force_config`` 提供的 DSL（``{"kind": "constant"}`` 或 ``{"kind": "pulse"}``）构造推力剖面。

.. automodule:: e2m2e.core.forces.force_model
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.force_config
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.physical_model
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.gravity_field
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.drag
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.thrust
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.indirect_term
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.core.forces.relativistic_correction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

太阳辐射压与阴影模型
====================

e2m2e 提供基于 cannonball 模型的太阳辐射压（SRP）力模型，以及圆锥阴影模型用于计算地影/月影对光照的遮挡效应。两者均通过 ``PhysicalModel`` 接口与力模型组合集成，支持配置驱动的序列化与反序列化。

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
     - ``None`` （全光照）

阴影模型
--------

:class:`~e2m2e.core.forces.shadow.ConicalShadowModel` 定义 ``flux_factor(t, state, system) -> float`` 接口，是当前唯一的阴影模型实现。

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
     - ``None`` （使用内置默认值）

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

SRP 工作流示例
--------------

以下示例展示 SRP + 地影/月影 + ``EphemerisDynamics`` 的传播流程：

.. code-block:: python

   import numpy as np

   from e2m2e.core import (
       CelestialBodyOrigin,
       CoordinateSystem,
       EphemerisSystem,
       ICRSAxes,
       SPICEManager,
   )
   from e2m2e.core.forces import (
       SolarRadiationPressure,
       ConicalShadowModel,
       ForceModel,
   )

   # 1. 加载 SPICE 内核
   mgr = SPICEManager()
   mgr.load_kernel("path/to/de440.bsp")

   # 2. 构建星历系统（含地球、月球、太阳；frame 默认 J2000）
   system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=mgr,
       origin="EARTH",
   )

   # 3. 设置坐标系（ICRF 惯性系，力模型要求惯性系）
   axes = ICRSAxes()
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
