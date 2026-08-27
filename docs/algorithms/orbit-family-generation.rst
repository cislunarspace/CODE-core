Unified Orbit Family Generation / 统一轨道族生成接口
=====================================================

[English](#unified-orbit-family-generation) | [简体中文](#中文)

English
-------

``Facade.orbit_family_generation()`` generates eight family types — Halo, NRHO,
Axial, Lissajous, SPO, LPO, Horseshoe, DRO — through one interface. The Facade
returns its dedicated Pydantic response ``FamilyGenerationResponse``; it
inherits :class:`e2m2e.data.types.orbit.OrbitFamily` to keep existing reading
interfaces while carrying the ``status/cause/message`` triple directly.
``n_orbits`` caps member count without guaranteeing fullness; on numeric
soft failures the same response retains already-converged members. The
algorithm-layer entry keeps using
:class:`e2m2e.algorithm.results.FamilyGenerationResult` for soft failure.

Parameter contract
~~~~~~~~~~~~~~~~~~

Common parameters: ``orbit_type``, ``libration_point``, ``n_orbits``; DRO is
Moon-centered — requests must not carry ``libration_point`` (explicit presence
rejects, consistent with other families rejecting cross-family fields). All
other fields are per-family:

.. list-table::
   :header-rows: 1

   * - Family
     - Libration point
     - Family params
     - Method
   * - Halo
     - L1/L2
     - ``max_amplitude_km``, ``sampling_mode=natural-z0`` (north +, south −)
     - Natural-parameter continuation at fixed z0
   * - NRHO
     - L1/L2
     - ``north_south``, perilune height, ``continuation_direction=toward-moon``
     - L1: single Rust PAL; L2: fold then fixed-x0 continuation
   * - Axial
     - L1/L2
     - ``max_amplitude_km``, ``continuation_direction=increase-amplitude``
     - Walking with vz0 as Type-B family parameter
   * - Lissajous
     - L1/L2/L3
     - In/out-of-plane amplitudes, two phases, ``sampling_mode=linear-amplitudes``
     - Amplitude sampling (not continuation)
   * - SPO
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Planar full-period PAL
   * - LPO
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Planar full-period PAL
   * - Horseshoe
     - L4/L5
     - amplitude range, ``continuation_direction`` (increase/decrease-x0), match tol.
     - Large-amplitude classification of the LPO chain
   * - DRO
     - none (Moon-centered)
     - amplitude range, ``sampling_mode=natural-x0``
     - Single x0 natural continuation from the standard seed (bidirectional across-seed walking)

Request models fill defaults and validate conditional ranges per family.
Ranges are queryable before constructing requests:

.. code-block:: python

   from e2m2e.api.models import FamilyGenerationRequest

   ranges = FamilyGenerationRequest.valid_ranges("HORSESHOE", libration_point=4)
   print(ranges["min_amplitude_km"].format_interval())  # [50000.0, 110000.0]

   options = FamilyGenerationRequest.valid_options("LPO")
   print(options["continuation_direction"])
   # ("decrease-x0", "increase-x0")

DRO amplitude reuses single-orbit ``design_dro``'s definition (mean of min/max
lunar distance over one period, km) with matching request envelope
(1737–110000 km). Members come from one x0 natural continuation: below the
standard-seed amplitude (~90786 km) walk moonward, above walk earthward,
straddling seeds walk both ways; members return ascending by amplitude;
member parameters carry geometry only (``amplitude_km`` etc.), no libration point.

Periodic vs quasi-periodic semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Halo/NRHO/Axial/SPO/LPO/Horseshoe/DRO members are strictly periodic
(``family.periodicity == "periodic"``). Lissajous's in/out-plane frequencies are
irrational-ratio: members are quasi-periodic bounded multi-point trajectories on
Rust's nonlinear center-reduced flow without periodic closure; the unified
container distinguishes explicitly:

.. code-block:: python

   family = facade.orbit_family_generation(
       orbit_type="LISSAJOUS",
       libration_point=2,
       amplitude_in_km=2400.0,
       amplitude_out_km=7200.0,
       phase_in=0.01,
       phase_out=0.55,
       n_orbits=3,
   )

   assert family.is_quasi_periodic
   assert family.metadata["periodicity"] == "quasi-periodic"

A Lissajous member's ``period`` is the nominal in-plane one, not full-trajectory
closure. Downstream must neither re-propagate ``states[0]`` as a strict periodic
initial state nor apply periodic-closure assertions.

Numeric boundary
~~~~~~~~~~~~~~~~

Python only validates requests, dispatches families, rewraps domain results.
Rust's single family-generation entry hides seed construction, CR3BP propagation
+ STM, differential correction, PAL Newton, step control, member filtering,
collinear-point center modes, Lissajous trajectory sampling, and perilune-distance
/ L4-L5 radial / out-of-plane amplitude measurement — no Python numeric fallback.
Lissajous advances with the full CR3BP nonlinear potential gradient inside the
4-D center subspace, freezing hyperbolic directions; equivalence with the single-
orbit entry's high-order normal form is not claimed.

Example
~~~~~~~

This request generates up to five L4 SPO members between 5,000–20,000 km:

.. code-block:: python

   from e2m2e.api.facade import Facade

   family = Facade().orbit_family_generation(
       orbit_type="SPO",
       libration_point=4,
       min_amplitude_km=5000.0,
       max_amplitude_km=20000.0,
       n_orbits=5,
   )

   for orbit in family:
       print(orbit.parameters["amplitude_km"], orbit.period)

Runnable version: ``examples/main_family_generation.py``.

中文
----

``Facade.orbit_family_generation()`` 统一生成 Halo、NRHO、Axial、
Lissajous、SPO、LPO、Horseshoe 和 DRO 八类轨道族。Facade 返回专属
Pydantic 响应 ``FamilyGenerationResponse`` ；它继承
:class:`e2m2e.data.types.orbit.OrbitFamily` 以保持既有读取接口，并直接携带
``status/cause/message`` 状态三元组。``n_orbits`` 表示成员数量上限，不保证
一定生成满额；数值延拓软失败时同一响应保留已收敛成员。算法层入口仍使用
:class:`e2m2e.algorithm.results.FamilyGenerationResult` 表达软失败。

参数契约
~~~~~~~~

公共参数是 ``orbit_type`` 、``libration_point`` 和 ``n_orbits`` ；DRO 是
月心族，请求不携带 ``libration_point`` （显式携带即拒绝，与其余七族拒绝
跨族字段的口径一致）。其余字段按族解释：

.. list-table::
   :header-rows: 1

   * - 族
     - 平动点
     - 族参数
     - 生成方法
   * - Halo
     - L1/L2
     - ``max_amplitude_km`` 、``sampling_mode=natural-z0`` （正北、负南）
     - 固定 z0 自然参数延拓
   * - NRHO
     - L1/L2
     - ``north_south`` 、近月点高度、``continuation_direction=toward-moon``
     - L1 单次 Rust 伪弧长延拓；L2 折叠后固定 x0 延拓
   * - Axial
     - L1/L2
     - ``max_amplitude_km`` 、``continuation_direction=increase-amplitude``
     - 以 vz0 为 Type B 族参数行走
   * - Lissajous
     - L1/L2/L3
     - 面内/面外振幅、两相位、``sampling_mode=linear-amplitudes``
     - 振幅参数采样（非延拓）
   * - SPO
     - L4/L5
     - 振幅范围、``continuation_direction`` （increase/decrease-x0）、匹配容差
     - 平面全周期伪弧长延拓
   * - LPO
     - L4/L5
     - 振幅范围、``continuation_direction`` （increase/decrease-x0）、匹配容差
     - 平面全周期伪弧长延拓
   * - Horseshoe
     - L4/L5
     - 振幅范围、``continuation_direction`` （increase/decrease-x0）、匹配容差
     - LPO 链的大振幅成员分类
   * - DRO
     - 无（月心族）
     - 振幅范围、``sampling_mode=natural-x0``
     - 从标准种子单次 x0 自然参数延拓（跨种子窗口双向行走）

请求模型按轨道族填默认值并校验条件范围。调用方可在构造请求前查询范围：

.. code-block:: python

   from e2m2e.api.models import FamilyGenerationRequest

   ranges = FamilyGenerationRequest.valid_ranges("HORSESHOE", libration_point=4)
   print(ranges["min_amplitude_km"].format_interval())  # [50000.0, 110000.0]

   options = FamilyGenerationRequest.valid_options("LPO")
   print(options["continuation_direction"])
   # ("decrease-x0", "increase-x0")

DRO 的振幅沿用单轨 ``design_dro`` 的定义（一个周期内距月心距离
min/max 均值，km），请求包络与单轨一致（1737~110000 km）。族成员
由一次 x0 自然参数延拓产出：窗口位于标准种子振幅（约 90786 km）下方
时向月侧行走、上方时向地侧行走、跨种子时双向行走，成员按振幅升序
返回；成员参数只含 ``amplitude_km`` 等几何量，不含平动点。

周期与拟周期语义
~~~~~~~~~~~~~~~~

Halo、NRHO、Axial、SPO、LPO、Horseshoe 和 DRO 成员是严格周期轨道，
``family.periodicity == "periodic"`` 。Lissajous 面内/面外频率不可约，
成员是 Rust 非线性中心约化流上的拟周期有界多点轨迹，不做周期闭合；统一容器
通过以下标注明确区分：

.. code-block:: python

   family = facade.orbit_family_generation(
       orbit_type="LISSAJOUS",
       libration_point=2,
       amplitude_in_km=2400.0,
       amplitude_out_km=7200.0,
       phase_in=0.01,
       phase_out=0.55,
       n_orbits=3,
   )

   assert family.is_quasi_periodic
   assert family.metadata["periodicity"] == "quasi-periodic"

Lissajous 成员的 ``period`` 是面内名义周期，不表示完整轨迹闭合。下游
不得把 ``states[0]`` 当作严格周期初态重传播，也不得套用周期闭合断言。

数值边界
~~~~~~~~

Python 只负责请求校验、族分派和领域结果重包。Rust 的单次族生成入口隐藏
种子构造、CR3BP 传播与 STM、微分修正、PAL Newton、步长控制、成员筛选、
共线点中心模态、Lissajous 轨迹采样，以及近月距、L4/L5 径向振幅和面外
振幅测量；该入口没有 Python 数值回退。Lissajous 在四维中心子空间内使用
完整 CR3BP 非线性势梯度推进，冻结双曲方向；它不宣称等同于单轨入口的高阶
normal-form 展开。

调用示例
~~~~~~~~

以下请求生成 L4 SPO 族中振幅为 5,000 至 20,000 km 的至多五个成员：

.. code-block:: python

   from e2m2e.api.facade import Facade

   family = Facade().orbit_family_generation(
       orbit_type="SPO",
       libration_point=4,
       min_amplitude_km=5000.0,
       max_amplitude_km=20000.0,
       n_orbits=5,
   )

   for orbit in family:
       print(orbit.parameters["amplitude_km"], orbit.period)

可运行版本见 ``examples/main_family_generation.py`` 。
