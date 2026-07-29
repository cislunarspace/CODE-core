稳定性分析
==========

稳定性分析计算周期轨道单值矩阵的特征值谱，以分类轨道稳定性。

Floquet 理论
------------

周期轨道的稳定性由单值矩阵（monodromy matrix）的特征值决定。
单值矩阵是一个周期的状态转移矩阵：

.. math::

   \mathbf{M} = \boldsymbol{\Phi}(T, 0)

其中 T 为轨道周期。特征值 λ 满足：

- abs(λ) < 1：稳定方向
- abs(λ) > 1：不稳定方向
- abs(λ) = 1：中心方向（中性稳定）

由于 Hamiltonian 系统的辛结构，特征值成对出现：(λ, 1/λ)。

使用方法
--------

:class:`~e2m2e.algorithms.stability.StabilityAnalysis` 分析单条周期轨道的稳定性：

.. code-block:: python

   from e2m2e.algorithms.stability import StabilityAnalysis

   # 构造时传入轨道；orbit 关联了 system 时 dynamics 可省略
   analyzer = StabilityAnalysis(orbit, dynamics)

   # analyze() 不接受参数，返回不可变的 OrbitStability 结果对象
   result = analyzer.analyze()

   print(f"特征值: {result.eigenvalues}")
   print(f"稳定性指数: {result.stability_indices}")
   print(f"分类: {result.classification['stability_type']}")

``analyze()`` 返回 frozen 数据类 ``OrbitStability``，字段包括：

- ``monodromy_matrix`` — 单值矩阵，形状 (6, 6)
- ``eigenvalues`` — 单值矩阵特征值（Floquet 乘子）
- ``stability_indices`` — 稳定性指数字典，键为 ``nu1``/``nu2``/``nu3``/``broucke``
- ``classification`` — 稳定性分类结果（含 ``stability_type``、``is_stable`` 等）
- ``bifurcation`` — 分岔分析结果
- ``numerical_errors`` — 数值误差估计

稳定性指标
----------

稳定性指数采用 Broucke 定义：对每对倒数特征值 (λ, 1/λ) 求和取实部，

.. math::

   \nu = \lambda + \frac{1}{\lambda}

- \|ν\| < 2：该模态稳定（特征值位于单位圆上）
- \|ν\| > 2：该模态不稳定（ν 越大越不稳定）

``stability_indices`` 字典给出各模态的 ``nu1``/``nu2``/``nu3``，以及
``broucke`` = \|ν1\| + \|ν2\|。特征值的最大模可从
``classification["max_eigenvalue_magnitude"]`` 读取。

批量分析
--------

对轨道族中每条轨道逐一分析：

.. code-block:: python

   from e2m2e.algorithms.stability import StabilityAnalysis

   results = []
   for orbit in family:
       result = StabilityAnalysis(orbit, dynamics).analyze()
       results.append(result)
       nu_max = max(v for v in result.stability_indices.values() if v is not None)
       print(f"周期={orbit.period:.4f}, 稳定性指数={nu_max:.6f}")

分岔检测
--------

``analyze()`` 返回的 ``bifurcation`` 字段由 ``analyze_bifurcation()`` 生成，
根据特征值与 +1、-1 及单位圆的关系识别分岔类型（``BifurcationType`` 枚举）：

- 特征值接近 +1 → 鞍结分岔（``SADDLE_NODE``）
- 特征值接近 -1 → 倍周期分岔（``PERIOD_DOUBLING``）
- 复特征值接近单位圆 → 环面分岔（``TORUS``）

对整个轨道族检测分岔点，使用静态方法：

.. code-block:: python

   # 遍历轨道族，返回特征值接近 +1 的分岔点列表
   bifurcations = StabilityAnalysis.detect_bifurcation_in_family(
       orbits=family,
       dynamics=dynamics,
       tolerance=1e-8,
   )

   # 或直接定位最接近目标 x0 的分岔点，未找到时返回 None
   bp = StabilityAnalysis.find_nearest_bifurcation(
       orbits=family,
       dynamics=dynamics,
       target_x0=0.85,
   )

数值误差校验
------------

单值矩阵由数值积分得到，``numerical_errors`` 字段给出两项残差，
用于校验积分精度是否足够：

- ``determinant_error`` — \|det(M) − 1\|，辛矩阵行列式应恒为 1
- ``symplectic_error`` — ‖MᵀJM − J‖ 的范数，辛性质残差

残差显著偏大时，应收紧积分容差后重新分析。

参考
----

- Hairer E, Nørsett S P, Wanner G. *Solving Ordinary Differential Equations I*, Chapter IV.8.
- Howell K C. *Three-dimensional, periodic, 'halo' orbits*, 1983.

可运行示例见 ``examples/stability_analysis.py``。
