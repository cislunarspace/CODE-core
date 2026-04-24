稳定性分析
==========

周期轨道的稳定性分析方法。

Floquet 理论
------------

通过分析状态转移矩阵的特征值判断稳定性：

- 特征值在单位圆内: 稳定
- 特征值在单位圆外: 不稳定

.. code-block:: python

   from e2m2e.algorithms.stability import analyze_stability

   # 分析轨道稳定性
   result = analyze_stability(orbit, system)
   print(f"稳定: {result.is_stable}")
