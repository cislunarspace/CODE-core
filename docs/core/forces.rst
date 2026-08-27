Force Models / 力模型
=====================

[English](#english) | [简体中文](#中文)

English
-------

The forces subpackage provides configurable, composable spacecraft
perturbation-force models: buildable from config dicts, serializable to JSON,
integrated with ephemeris dynamics for propagation.

Core concepts
~~~~~~~~~~~~~

- **PhysicalModel**: abstract base of all force models defining
  ``compute_acceleration(t, state, system)``. Two optional hooks:
  ``compute_jacobian(t, state, system)`` returning analytic ∂a/∂r (default
  ``None``; ``ForceModel`` falls back to finite differences — Python fallback
  paths were removed with issue #378, production runs require Rust); and
  ``to_rust_spec(system)`` serializing the force into a tuple accepted by the
  Rust compiled path (default ``None`` = not compilable).
- **ForceModel**: the composition container assembling multiple
  ``PhysicalModel``\ s into one propagation's equations of motion; registers,
  enables/disables by name; propagates via Rust integrators.
- **ForceEntry**: per-force registry record inside a container holding ``name``,
  ``force``, ``enabled``.

Supported force types
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - Config type name
   * - PointMassGravity
     - Central body point-mass gravity
     - ``PointMassGravity``
   * - ThirdBodyGravity
     - Third-body perturbation (incl. indirect term)
     - ``ThirdBodyGravity``
   * - IndirectTerm
     - Standalone indirect-term correction
     - ``IndirectTerm``
   * - GravityField
     - Spherical-harmonics gravity field (body-agnostic; EGM96/GRGM900C & custom .gfc/.cof)
     - ``GravityField``
   * - DragModel
     - Atmospheric drag (injected density model)
     - ``DragModel``
   * - SolarRadiationPressure
     - SRP (cannonball + optional shadow model)
     - ``SolarRadiationPressure``
   * - EcomSolarRadiationPressure
     - ECOM empirical SRP (9-coefficient DYB; DFH-compatible)
     - ``EcomSolarRadiationPressure``
   * - FiniteBurn
     - Continuous thrust (closed DSL: constant/pulse profile + fixed direction)
     - ``FiniteBurn``
   * - VariableMassFiniteBurn
     - Variable-mass continuous thrust (mass as 7th state)
     - none (constructed directly, not via config registry)
   * - RelativisticCorrection
     - Post-Newtonian corrections (Schwarzschild / Lense-Thirring / de Sitter)
     - ``RelativisticCorrection``

Also ``ImpulsiveBurn`` (instant Δv at an epoch) — not a ``PhysicalModel``, no
acceleration accumulation, not config-registerable; applied via
``propagate_maneuvers`` (below).

.. warning::

   When using ``GravityField`` for the Moon (including degree=0 central term),
   you must separately add lunar indirect term ``IndirectTerm("MOON")`` and must
   NOT also add ``ThirdBodyGravity("MOON")``: the latter would double-count the
   Moon's point mass with GravityField's central term.

Built-in formulas (summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PointMassGravity**: :math:`\mathbf{a} = -\mu\,\mathbf{r}/|\mathbf{r}|^3` — origin-body two-body gravity only.
**ThirdBodyGravity**: direct + indirect terms referencing SPICE positions;
indirect subtraction keeps the frame origin fixed.
**GravityField**: fully-normalized harmonics via Pines recursion; Earth solid
tides (Step1+Step2+pole+permanent), Moon k₂ = 0.024116 tide.
**DragModel**: computed in ITRF;
:math:`\mathbf{a} = -\tfrac{1}{2}\rho\,(C_d A/m)\,|\mathbf{v}_{rel}|\,\mathbf{v}_{rel}`
with exponential atmosphere (US76 segments), Cd default 2.2.
**FiniteBurn**: :math:`T(t)/m \cdot \hat{\mathbf{d}}`; direction frames inertial /
VNB / LVLH; round-trip configs only through the closed DSL.
**VariableMassFiniteBurn**: same but mass = state[6], burned at
:math:`\dot m = -T/(I_{sp} g_0)`; propagation auto-switches to 7-D augmented
Rust path.
**RelativisticCorrection**: Schwarzschild + Lense-Thirring + de Sitter, GMAT-
aligned.

Registry mechanism
~~~~~~~~~~~~~~~~~~

Auto-naming by class name with disambiguation (``GravityField_2`` …);
explicit names required unique; enable/disable/get/remove/list by name:

.. code-block:: python

   fm.add_force(GravityField("EARTH", degree=2, order=0))          # auto name
   fm.add_force(DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0), name="drag")
   fm.disable("drag")
   fm.get_force("drag")
   fm.remove_force("drag")

Propagation interface
~~~~~~~~~~~~~~~~~~~~~

``ForceModel.propagate`` drives adaptive stepping on Rust ``rk_step``: options
``t_eval``, ``with_stm`` (42-dim augmented integration — STM components excluded
from step-error control, matching GMAT), ``initial_step``, ``max_steps``,
``method=RkMethod.PD45`` default.

Events unsupported (raises ``NotImplementedError`` on non-None events): use
``Dynamics.propagate`` or post-hoc detection instead. For impulsive maneuvers:
``propagate_maneuvers(initial_state, t_span, burns)`` sorts ``ImpulsiveBurn``\ s by
epoch, coasts between, applies ``state[3:6] += delta_v``, returns extra ``burns``
key.

Compiled Rust fast path: with spice enabled, propagation enters
``propagate_compiled`` — one serialization then full loop in Rust. No silent
fallbacks: missing extension raises ``RustExtensionUnavailableError``; any
enabled force whose ``to_rust_spec()`` is None raises ``NotImplementedError``
(ADR 0020 decision 4). The STM fast path additionally excludes
``RelativisticCorrection`` / ``VariableMassFiniteBurn`` for now (explicit error).

Rust ephemeris pre-sampling cache: ``enable_ephem_cache(targets, frame_pairs,
et_start, et_end, dt, sxform_pairs)`` pre-samples body states + frame matrices
into in-memory cubic splines so inner-loop lookups skip FFI; C² continuity avoids
step-size collapse at grid nodes (~1e-3 km terminal accuracy at 600 s pxform
grids). Wrap propagation in try/finally with ``disable_ephem_cache()``.

Config-driven construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Build from a dict: ``{"version": 1, "forces": [{"name", "type", "enabled",
"params"}...]}``, nested models (atmosphere/shadow) as recursive ``{type,
params}``. Entry points: ``ForceModel.from_config(config, system)``,
``fm.to_config()`` (normalized values; round-trip contract =
``to_config(from_config(c)) == c`` after re-serialization),
``dump_force_config(fm, path)`` / ``load_force_config(path, system)``. Full LEO
walkthrough in the Chinese section below (identical code).

Common errors
~~~~~~~~~~~~~

- Unknown name → ``KeyError`` from get/disable/remove.
- Duplicate explicit name → ``ValueError: force name '...' already exists``.
- Version mismatch → ``unsupported config version 2; expected 1``.
- Unknown type → ValueError listing known types.
- User-written callables in ``FiniteBurn`` propagate fine but raise
  ``NotSerializableError`` on ``to_config()`` — use the DSL kinds
  (constant/pulse).

Solar radiation pressure & shadow model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SRP** (cannonball, Montenbruck & Gill):
:math:`\mathbf{a} = \text{flux} \cdot P_{1AU}(1\text{AU}/r)^2 \frac{C_R A}{m} \hat{\mathbf{u}}`
with :math:`P_{1AU} = 4.56\times10^{-6}` N/m²; params: area & mass (required),
cr (default 1.5), shadow (default None = full illumination).
**ConicalShadowModel** implements GMAT ShadowState conical algorithm
(M&G §3.4.2): four branches — full sun / umbra (flux=0) / penumbra (exact disc-
overlap area, eq. 3.92–3.94) / annular. Multi-occulter composition per GMT-6543:
any umbra → 0; disjoint partials → inclusive-exclusive; overlapping → min.
Default occluder radii: Earth 6378.1363, Moon 1737.4, Sun 695700 km. Both
models serialize through ``force_config`` and offer pure-function test paths
without SPICE.

中文
----

（本页保留原文全文；上方英文节为对应摘要。）

e2m2e 的力模型子包提供可配置、可组合的航天器摄动力模型，支持从配置字典构建、序列化到 JSON，并与星历动力学系统配合完成轨道传播。

核心概念
--------

这三个概念构成力模型子包的核心接口：

- **PhysicalModel**：所有力模型的抽象基类，定义 ``compute_acceleration(t, state, system)`` 接口。另有两个可选钩子：``compute_jacobian(t, state, system)`` 返回解析雅可比 ∂a/∂r（默认返回 ``None`` ）；``to_rust_spec(system)`` 把力序列化为 Rust 编译路径接受的元组。
- **ForceModel**：力模型组合，把多个 ``PhysicalModel`` 组合成一次传播所需的运动方程；按名登记、启用/禁用，并通过 Rust 积分器完成传播。
- **ForceEntry**：容器内单个力模型的登记记录，含 ``name`` 、``force`` 和 ``enabled`` 。

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
   * - EcomSolarRadiationPressure
     - ECOM 经验光压（9 系数 DYB 参数化，DFH 兼容）
     - ``EcomSolarRadiationPressure``
   * - FiniteBurn
     - 连续推力（封闭 DSL：constant/pulse 推力 + fixed 方向）
     - ``FiniteBurn``
   * - VariableMassFiniteBurn
     - 可变质量连续推力（7D 受控动力学）
     - 无（不走配置注册表，直接构造）
   * - RelativisticCorrection
     - 后牛顿相对论修正（Schwarzschild / Lense-Thirring / de Sitter）
     - ``RelativisticCorrection``

另有机动事件 ``ImpulsiveBurn`` （瞬时 Δv），不是 ``PhysicalModel`` ；
施加方式见下文 ``propagate_maneuvers`` 。

.. warning::

   用 ``GravityField`` 模拟月球（含 degree=0 中心项）时，必须单独补月球间接项
   ``IndirectTerm("MOON")`` ，且不能再加 ``ThirdBodyGravity("MOON")`` ：后者会与
   ``GravityField`` 的中心项重复计算月球点质量。

内置力模型公式
--------------

**PointMassGravity**：中心天体点质量引力，:math:`\mathbf{a} = -\mu\mathbf{r}/|\mathbf{r}|^3`。

**ThirdBodyGravity**：直接项加间接项（经 SPICE 查位置），间接项扣除摄动天体对原点的引力：

.. math::

   \mathbf{a} = -\mu_i \left[ \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3} + \frac{\mathbf{r}_i}{|\mathbf{r}_i|^3} \right]

**GravityField**：完全正规化球谐（Pines 递推），内置地球固体潮（Step1+Step2+极潮+永久潮）与月球 k₂ = 0.024116 固体潮。

**DragModel**：ITRF 中计算（ExponentialAtmosphere 为 US76 分段指数模型，Cd 默认 2.2）：

.. math::

   \mathbf{a}_{\text{drag}} = -\frac{1}{2}\rho \frac{C_d A}{m} \|\mathbf{v}_{rel}\| \mathbf{v}_{rel}

**FiniteBurn**：:math:`\mathbf{a} = T(t)/m \cdot \hat{\mathbf{d}}`；方向支持惯性系/VNB/LVLH。

**VariableMassFiniteBurn**：质量为状态量 state[6]，按 :math:`\dot m = -T/(I_{sp} g_0)` 消耗，7D 增广传播自动分流 Rust。

**RelativisticCorrection**：Schwarzschild/Lense-Thirring/de Sitter 三项，与 GMAT 对齐。

name-based 注册机制
--------------------

.. code-block:: python

   fm.add_force(GravityField("EARTH", degree=2, order=0))            # 自动命名+消歧（GravityField_2…）
   fm.add_force(DragModel(atmosphere=ExponentialAtmosphere(), area=10.0, mass=1000.0), name="drag")
   fm.disable("drag"); fm.enable("drag"); fm.get_force("drag"); fm.remove_force("drag")

传播接口
--------

``ForceModel.propagate`` 用 Rust ``rk_step`` 自适应传播：
``with_stm=True`` 时 42 维增广积分（STM 分量不参与步长误差控制，对齐 GMAT），
返回多一个 ``stm`` 键 ``(n_points, 6, 6)`` 。不支持 events（传非 None 抛
NotImplementedError）。脉冲机动用 ``propagate_maneuvers(initial_state,
t_span, burns)``：按 epoch 排序 ImpulsiveBurn，逐段 coast，施加
``state[3:6] += delta_v`` 后续传。

Rust 编译快速路径：spice 启用时走编译路径 ``propagate_compiled`` （无静默回退）——
扩展缺失时抛出 ``RustExtensionUnavailableError`` ；启用力不可序列化时抛出 NotImplementedError （ADR 0020 决策 4）。STM 快速路径当前仍不支持
``RelativisticCorrection`` 与 ``VariableMassFiniteBurn`` （显式报错）。

Rust 星历预采样缓存：

.. code-block:: python

   from e2m2e._integrators import enable_ephem_cache, disable_ephem_cache

   enable_ephem_cache(
       targets=[("MOON", "EARTH"), ("SUN", "EARTH"), ("EARTH", "SOLAR SYSTEM BARYCENTER")],
       frame_pairs=[("ITRF93", "J2000"), ("MOON_PA", "J2000")],
       et_start=et0, et_end=et0 + duration, dt=600.0,
       sxform_pairs=[("ITRF93", "J2000")],
   )
   try:
       result = fm.propagate(...)
   finally:
       disable_ephem_cache()

三次样条保 C² 连续（pxform 600s 网格末态精度 ~1e-3 km）；未激活时逐字一致。

配置驱动构建流程
------------------

完整示例（从配置构建到 round-trip 校验）见英文节同一套代码，要点：

.. code-block:: python

   fm = ForceModel.from_config(config, system)      # {"version": 1, "forces": [...]}
   config_roundtrip = fm.to_config()                # 规范形式
   assert ForceModel.from_config(config_roundtrip, system).to_config() == config_roundtrip

JSON 文件 IO
------------

.. code-block:: python

   from e2m2e.algorithm.forces import dump_force_config, load_force_config
   dump_force_config(fm, "forces.json")
   fm2 = load_force_config("forces.json", system)

常见错误与排错
--------------

- 未注册名称：``KeyError`` ；
- 显式重名：``ValueError: force name 'primary' already exists`` ；
- 版本不匹配：``unsupported config version 2; expected 1`` ；
- 未知类型：ValueError 列出全部已知 type；
- 手写 callable 的 ``FiniteBurn`` ：可传播但 ``to_config()`` 抛
  ``NotSerializableError`` ，改用 DSL（constant/pulse）。

太阳辐射压与阴影模型
====================

SRP 模型（cannonball 口径，Montenbruck & Gill）：

.. math::

   \mathbf{a} = \text{flux} \cdot P_{1\text{AU}} \left(\frac{1\ \text{AU}}{r}\right)^2 \frac{C_R \, A}{m} \, \hat{\mathbf{u}}

:math:`P_{1AU} = 4.56\times10^{-6}` N/m²；参数：area/mass 必填、cr 默认 1.5、shadow 默认全光照。

**ConicalShadowModel** 实现 GMAT ShadowState 圆锥阴影算法（M&G §3.4.2）四分支判定：
全光照 / 本影（flux=0）/ 半影（eq. 3.92-3.94 精确圆面重叠面积）/ 环形食。
多遮挡体合成遵循 GMAT GMT-6543：任一本影→0；不重叠部分阴影→包容排斥；重叠→取最小值。
内置半径：EARTH 6378.1363、MOON 1737.4、SUN 695700 km。两模型均支持配置序列化
与纯函数测试路径（无 SPICE 环境可直接测试）。

.. automodule:: e2m2e.algorithm.forces.force_model
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.force_config
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.physical_model
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.gravity_field
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.drag
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.thrust
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.indirect_term
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: e2m2e.algorithm.forces.relativistic_correction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
