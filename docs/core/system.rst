CR3BP 系统
==========

圆型限制性三体问题 (CR3BP) 系统定义。

概述
----

CR3BP 描述一个质量可忽略的第三体在两个主天体引力场中的运动。
两个主天体绕其公共质心做圆周运动，采用旋转坐标系使主天体固定。

质量参数
--------

质量参数 μ = m₂/(m₁+m₂)，其中 m₂ 为较小天体质量。

- 地月系统: μ ≈ 0.01215
- 日地系统: μ ≈ 3.0039e-6
- 日木系统: μ ≈ 0.0009535

拉格朗日点
----------

系统有 5 个拉格朗日点 (L1-L5)，是旋转坐标系中的平衡点。

.. math::

   L1, L2, L3: 共线平衡点（x 轴上）
   L4, L5: 三角平衡点（等边三角形顶点）

Jacobi 常数
-----------

Jacobi 常数是 CR3BP 中唯一的运动积分：

.. math::

   C_J = 2\Omega - v^2

其中 Ω 为伪势能，v 为速度大小。

使用示例
--------

.. code-block:: python

   from e2m2e.core.system import CR3BP_System

   # 创建地月系统
   system = CR3BP_System.from_known_system("earth_moon")

   # 获取质量参数
   print(f"μ = {system.mu}")

   # 计算拉格朗日点
   L1 = system.libration_points[0]
   print(f"L1 = {L1}")

   # 计算 Jacobi 常数
   state = [0.8, 0, 0, 0, 0.6, 0]
   CJ = system.get_jacobi_constant(state)
   print(f"C_J = {CJ}")
