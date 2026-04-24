微分修正
========

微分修正算法用于求解周期轨道。

基本原理
--------

微分修正是一种迭代方法，通过修正初始条件使轨道闭合。

.. math::

   \delta x_0 = -J^{-1} \cdot f(x_0)

其中 J 为 Jacobian 矩阵，f 为残差函数。

实现
----

.. code-block:: python

   from e2m2e.algorithms.differential_correction import correct_orbit

   # 修正轨道
   corrected = correct_orbit(orbit, system, tolerance=1e-10)
