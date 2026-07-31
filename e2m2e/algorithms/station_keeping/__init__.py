"""轨道保持（station-keeping）算法包（DFH 功能码 2 的数值内核）。

沿标称轨道的闭环修正问题，与两点间弹道设计（``transfer/``）是不同问题。
三种控制律 + 三轨道误差仿真 + 蒙特卡洛驱动的规格见
``docs/plans/dfh-parity-prd.md`` FR2，算法以《控制方案.md》（hybrid_auto
版）为准：

- ``special_point.py``：特征点控制（x-z 平面穿越约束，STM 子矩阵牛顿迭代）
- ``target_point.py``：目标点严格控制（位置重合微分修正）与宽松控制
  （线性二次解析最优）
- ``error_models.py``：Box-Muller 采样、测定轨扰动、分段控制误差、光压
  弧段随机误差
- ``monte_carlo.py``：三轨道结构（目标/真实/测量）批量仿真驱动
"""
