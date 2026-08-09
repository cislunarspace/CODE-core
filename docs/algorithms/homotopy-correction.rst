固定步长同伦星历转换
======================

本节描述 e2m2e 的 **固定步长同伦过渡** （fixed-step homotopy
transition）星历转换 MVP，源自 issue #235 的实现分解（#239 标准多重打靶
MVP、#241 二层多重打靶接入、#242 真实 SPICE 集成与文档）。

术语
----

- **同伦过渡 / homotopy transition** ：在动力学参数空间内构造一族
  连续插值模型 ``a(lambda) = a_base + lambda * (a_target - a_base)`` ，
  从 ``lambda=0`` （base 模型）平滑过渡到 ``lambda=1`` （目标模型），
  用以扩大多重打靶法的收敛盆地。
- **base 模型 / base bodies** ：星历动力学中显式参与引力计算的天体子集。
  MVP 默认 ``["EARTH", "MOON"]`` （不含太阳）。
- **目标模型 / target bodies** ：星历动力学中全部参与引力计算的天体集合。
  MVP 默认 ``["EARTH", "MOON", "SUN"]`` 。
- **固定步长 / fixed step** ：lambda 序列 ``[0.25, 0.50, 0.75, 1.00]``
  不做自适应或反馈调节，按预设顺序执行。
- **标准多重打靶 / standard multiple shooting** ：
  ``MultipleShooting`` ，最小二乘修正所有节点状态与时间。
- **二层多重打靶 / two-level multiple shooting** ：
  ``TwoLevelMultipleShooting`` ，交替求解位置连续（Level 1）和
  速度连续（Level 2）子问题。

同伦过渡在 e2m2e 中的语义
-------------------------

同伦过渡 = **天体分组策略** （body grouping strategy），不涉及
**坐标转换** ，也 **不引入 CR3BP 动力学** ：

- 输入 ``t_patch`` 是 **SPICE ET 秒** （J2000 历元起算）；
- 输入 ``state_patch`` 形状 ``(N, 6)`` ，单位 **km / km/s** ，参考系
  ``J2000`` ；
- CR3BP 轨道数据若要作为初值，必须由调用方完成 synodic → J2000、
  无量纲 → 物理量的转换后传入；同伦流程内部不进行任何转换。
- 加速度 / 雅可比按天体分组线性插值：

  .. math::

     a(\lambda) = a_{\text{base}} + \lambda \cdot (a_{\text{full}} - a_{\text{base}})

  .. math::

     J(\lambda) = J_{\text{base}} + \lambda \cdot (J_{\text{full}} - J_{\text{base}})

适用范围与不承诺项
------------------

**承诺** ：

- base_bodies ⊆ full_bodies，且 ``origin ∈ base_bodies`` ；
- lambda 步长严格递增、值域 ``[0, 1]`` 、末步必须为 ``1.0`` ；
- 中间步使用 ``tolerance * 10`` 容差，末步使用严格容差；
- 每步用上一步返回的 ``t_patch / state_patch`` 作为下一步初值
  （即使上一步未收敛）；
- ``inner_method ∈ {"standard", "two_level"}`` ；
- 拒绝 ``inner_method="homotopy"`` ，防止无限递归。

**不承诺** ：

- 自适应步长（lambda 序列固定为 ``[0.25, 0.50, 0.75, 1.00]`` ）；
- CR3BP + 星历混合动力学（同伦流程内只有纯星历动力学）；
- 同伦流程内的坐标转换（synodic ↔ J2000 等）；
- 非 MVP 天体策略自动选择（如 Earth+Moon+Sun+Jupiter）；
- 同伦参数搜索 / 延拓。

API
---

高层入口（推荐）：``correct_ephemeris_patch_points`` 的 ``method="homotopy"`` 。

.. code-block:: python

   from e2m2e.algorithm.ephemeris_correction import (
       correct_ephemeris_patch_points,
   )
   from e2m2e.algorithm.dynamics.ephemeris_dynamics import EphemerisDynamics
   from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
   from e2m2e.data.templates.enums import ReferenceFrame
   from e2m2e.data.kernels.manager import SPICEManager

   spice = SPICEManager()
   spice.load_kernel("/path/to/de440.bsp")

   eph_system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice,
       origin="EARTH",
       frame=ReferenceFrame.J2000,
   )
   dynamics = EphemerisDynamics(system=eph_system)
   dynamics.rtol = 1e-10
   dynamics.atol = 1e-10
   dynamics.max_step = 600.0

   # t_patch: ET seconds, state_patch: km / km/s in J2000
   result = correct_ephemeris_patch_points(
       method="homotopy",
       dynamics=dynamics,
       t_patch=t_patch,           # np.ndarray (N,)
       state_patch=state_patch,   # np.ndarray (N, 6)
       tolerance=1e-3,
       max_iter=10,
       verbose=False,
       n_workers=1,
       kernel_dir="/path/to/kernels",
       base_bodies=["EARTH", "MOON"],
       # lambda_steps=[0.25, 0.50, 0.75, 1.00]  # 默认值
       # inner_method="standard",                # 默认值；可改为 "two_level"
   )

低层入口：``correct_with_homotopy`` （模块
``e2m2e.algorithm.ephemeris_correction.homotopy`` ），
用于需要更细粒度控制的研究场景。

参数
~~~~

- ``dynamics`` ：``EphemerisDynamics`` 实例，其 ``system.bodies``
  即 target bodies 集合。
- ``t_patch`` / ``state_patch`` ：ET 秒 / km-km·s⁻¹ 的 patch points 初值。
- ``tolerance`` ：末步收敛的位置容差。中间步用 ``tolerance * 10`` 。
- ``max_iter`` ：每个 lambda 步内的最大迭代次数。
- ``n_workers`` / ``kernel_dir`` ：透传给 ``MultipleShooting`` 或
  ``TwoLevelMultipleShooting`` 。
- ``base_bodies`` ：base 天体子集，必须包含 ``origin`` 。
- ``lambda_steps`` ：默认 ``[0.25, 0.50, 0.75, 1.00]`` 。
- ``inner_method`` ：``"standard"`` （默认）或 ``"two_level"`` 。

聚合结果字段
~~~~~~~~~~~~

``EphemerisCorrectionResult`` ：

- ``converged`` ：最后一步的 ``converged`` 标志。
- ``iterations`` ：所有 lambda 步的迭代次数之和（standard 是
  ``MultipleShootingResult.outer_iterations`` ，two_level 是
  ``TwoLevelMultipleShootingResult.outer_iterations`` ）。
- ``max_residual`` ：最后一步的最终位置残差。
- ``residual_history`` ：所有 lambda 步位置残差历史的扁平拼接。
- ``t_patch`` / ``state_patch`` ：最后一步返回的 patch points
  （ET 秒 / km-km·s⁻¹ / J2000）。
- ``velocity_residual`` ：仅 two_level 路径填充；最后一步最终速度残差。
- ``velocity_residual_history`` ：仅 two_level 路径填充；所有 lambda 步
  速度残差历史的扁平拼接。

最小代码示例
------------

**Earth+Moon → Earth+Moon+Sun，标准多重打靶同伦过渡** ：

.. code-block:: python

   import numpy as np
   from e2m2e.algorithm.ephemeris_correction import (
       correct_ephemeris_patch_points,
   )
   from e2m2e.algorithm.ephemeris_correction.homotopy import correct_with_homotopy
   from e2m2e.algorithm.dynamics.ephemeris_dynamics import EphemerisDynamics
   from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
   from e2m2e.data.kernels.manager import SPICEManager
   from e2m2e.data.templates.enums import ReferenceFrame

   KERNEL_DIR = "/path/to/kernels"
   spice = SPICEManager()
   spice.load_kernel(f"{KERNEL_DIR}/de440.bsp")

   eph_system = EphemerisSystem(
       bodies=["EARTH", "MOON", "SUN"],
       spice=spice, origin="EARTH", frame=ReferenceFrame.J2000,
   )
   dynamics = EphemerisDynamics(system=eph_system)

   # 5 patch points, J2000 / ET seconds / km / km/s
   t_patch = np.linspace(0.0, 1200.0, 5)
   state_patch = np.tile(
       np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]), (5, 1)
   )

   result = correct_ephemeris_patch_points(
       method="homotopy",
       dynamics=dynamics,
       t_patch=t_patch,
       state_patch=state_patch,
       tolerance=1e-3,
       max_iter=10,
       verbose=False,
       n_workers=1,
       kernel_dir=KERNEL_DIR,
       base_bodies=["EARTH", "MOON"],   # Earth+Moon → Earth+Moon+Sun
   )
   print(result.converged, result.iterations, result.max_residual)

**二层多重打靶同伦过渡** ：

.. code-block:: python

   result_two_level = correct_with_homotopy(
       dynamics=dynamics,
       t_patch=t_patch,
       state_patch=state_patch,
       tolerance=1e-3,
       max_iter=5,
       n_workers=1,
       kernel_dir=KERNEL_DIR,
       base_bodies=["EARTH", "MOON"],
       inner_method="two_level",         # 切到二层多重打靶
       # velocity_tolerance=1e-6,         # 可选，默认 1e-6
   )
   print(
       result_two_level.converged,
       result_two_level.velocity_residual,
       result_two_level.velocity_residual_history,
   )

异常与校验
----------

- ``base_bodies`` 不是 ``system.bodies`` 子集：
  ``ValueError: base_bodies ... must be a subset of system.bodies ...`` 。
- ``base_bodies`` 不包含 ``origin`` ：
  ``ValueError: base_bodies ... must include origin ...`` 。
- ``lambda_steps`` 为空 / 非严格递增 / 越界 / 末值 ≠ 1.0：
  ``ValueError: lambda_steps ...`` 。
- ``inner_method="homotopy"`` ：
  ``ValueError: inner_method='homotopy' is not allowed (would recurse)`` 。
- 内层求解器异常：包装为 ``RuntimeError`` ，异常信息包含
  ``lambda step <index> (lambda=<value>, inner_method=<method>)`` ，
  便于日志定位失败步。

测试覆盖
--------

- :mod:`tests.algorithm.correction.test_ephemeris_correction_dispatch`：
  dispatch delegation。
- :mod:`tests.algorithm.correction.test_homotopy_correction_orchestration`：
  参数校验、容差分层、初值传递、失败路径、残差历史可观测性。
- :mod:`tests.algorithm.correction.test_homotopy_correction_dynamics`：
  ``HomotopyEphemerisDynamics`` 在 lambda=0/1/中间值的加速度/雅可比
  线性插值语义，使用 ``FakeSpice`` 不依赖真实内核。
- :mod:`tests.algorithm.correction.test_homotopy_two_level`：
  two_level 路径的 delegation、聚合语义、参数校验。
- :mod:`tests.algorithm.correction.test_homotopy_ephemeris_integration`：
  真实 SPICE 内核的端到端集成测试，无内核时自动 ``skip`` 。

参考
----

- 同伦连续方法在多体动力学中的标准实践：``a(lambda) = (1-lambda) a0 + lambda a1`` ，
  见 ``HomotopyEphemerisDynamics._compute_acc_and_jacobian`` 。
- 二层多重打靶的接口契约：``TwoLevelMultipleShooting.correct`` 。
- 标准多重打靶的接口契约：``MultipleShooting.correct`` 。
