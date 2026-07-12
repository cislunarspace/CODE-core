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

   analyzer = StabilityAnalysis(dynamics)

   # 分析单条轨道
   result = analyzer.analyze(orbit)

   print(f"特征值: {result.eigenvalues}")
   print(f"稳定性指标: {result.stability_index}")
   print(f"不稳定方向数: {result.n_unstable}")

稳定性指标
----------

稳定性指标定义为：

.. math::

   \nu = \max_i |\lambda_i|

- ν = 1：中性稳定
- ν > 1：不稳定（ν 越大越不稳定）
- ν < 1：渐近稳定（Hamiltonian 系统中不会出现）

批量分析
--------

对轨道族中每条轨道逐一分析：

.. code-block:: python

   from e2m2e.algorithms.stability import StabilityAnalysis

   analyzer = StabilityAnalysis(dynamics)

   results = []
   for orbit in family:
       result = analyzer.analyze(orbit)
       results.append(result)
       print(f"周期={orbit.period:.4f}, 稳定性指标={result.stability_index:.6f}")

参考
----

- Hairer E, Nørsett S P, Wanner G. *Solving Ordinary Differential Equations I*, Chapter IV.8.
- Howell K C. *Three-dimensional, periodic, 'halo' orbits*, 1983.
