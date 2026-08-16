统一轨道族生成接口
==================

``Facade.orbit_family_generation()`` 统一生成 Halo、NRHO、Axial、
Lissajous、SPO、LPO 和 Horseshoe 七类轨道族。Facade 返回专属 Pydantic
响应 ``FamilyGenerationResponse``；它继承
:class:`e2m2e.data.types.orbit.OrbitFamily` 以保持既有读取接口，并直接携带
``status/cause/message`` 状态三元组。``n_orbits`` 表示成员数量上限，不保证
一定生成满额；数值延拓软失败时同一响应保留已收敛成员。算法层入口仍使用
:class:`e2m2e.algorithm.results.FamilyGenerationResult` 表达软失败。

参数契约
--------

公共参数是 ``orbit_type``、``libration_point`` 和 ``n_orbits``。其余字段
按族解释：

.. list-table::
   :header-rows: 1

   * - 族
     - 平动点
     - 族参数
     - 生成方法
   * - Halo
     - L1/L2
     - ``max_amplitude_km``、``sampling_mode=natural-z0`` （正北、负南）
     - 固定 z0 自然参数延拓
   * - NRHO
     - L1/L2
     - ``north_south``、近月点高度、``continuation_direction=toward-moon``
     - L1 单次 Rust 伪弧长延拓；L2 折叠后固定 x0 延拓
   * - Axial
     - L1/L2
     - ``max_amplitude_km``、``continuation_direction=increase-amplitude``
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

请求模型按轨道族填默认值并校验条件范围。调用方可在构造请求前查询范围：

.. code-block:: python

   from e2m2e.api.models import FamilyGenerationRequest

   ranges = FamilyGenerationRequest.valid_ranges("HORSESHOE", libration_point=4)
   print(ranges["min_amplitude_km"].format_interval())  # [50000.0, 110000.0]

   options = FamilyGenerationRequest.valid_options("LPO")
   print(options["continuation_direction"])
   # ("decrease-x0", "increase-x0")

周期与拟周期语义
----------------

Halo、NRHO、Axial、SPO、LPO 和 Horseshoe 成员是严格周期轨道，
``family.periodicity == "periodic"``。Lissajous 面内/面外频率不可约，
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
--------

Python 只负责请求校验、族分派和领域结果重包。Rust 的单次族生成入口隐藏
种子构造、CR3BP 传播与 STM、微分修正、PAL Newton、步长控制、成员筛选、
共线点中心模态、Lissajous 轨迹采样，以及近月距、L4/L5 径向振幅和面外
振幅测量；该入口没有 Python 数值回退。Lissajous 在四维中心子空间内使用
完整 CR3BP 非线性势梯度推进，冻结双曲方向；它不宣称等同于单轨入口的高阶
normal-form 展开。

调用示例
--------

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

可运行版本见 ``examples/main_family_generation.py``。
