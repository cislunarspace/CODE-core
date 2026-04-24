CR3BP 动力学
============

圆型限制性三体问题的运动方程与太阳辐射压力 (SRP) 摄动。

运动方程
--------

CR3BP 在旋转坐标系下的运动方程：

.. math::

   \ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}

   \ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}

   \ddot{z} = \frac{\partial \Omega}{\partial z}

其中 Ω 为伪势能函数。

伪势能 Ω
---------

伪势能由引力势与离心势组成：

.. math::

   \Omega = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}

   r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}

   r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}

其中 r₁、r₂ 分别为第三体到两个主天体的距离。

科里奥利力与离心力
------------------

旋转坐标系中的惯性力：

- **科里奥利力**: 与速度方向垂直，不改变能量
- **离心力**: 沿径向向外，与伪势能中的离心势对应

状态转移矩阵 (STM)
-------------------

状态转移矩阵描述初始状态微小偏差的线性演化：

.. math::

   \delta \mathbf{x}(t) = \boldsymbol{\Phi}(t, t_0) \, \delta \mathbf{x}(t_0)

其中 :math:`\boldsymbol{\Phi}` 为 6×6 状态转移矩阵，满足：

.. math::

   \dot{\boldsymbol{\Phi}} = \mathbf{A}(t) \, \boldsymbol{\Phi}, \quad \boldsymbol{\Phi}(t_0, t_0) = \mathbf{I}

:math:`\mathbf{A}(t)` 为运动方程的雅可比矩阵。

太阳辐射压力 (SRP)
-------------------

SRP 对第三体的摄动加速度：

.. math::

   \mathbf{a}_{SRP} = -\frac{P_{SR} \, C_R \, A}{m} \hat{\mathbf{r}}_s

其中：

- :math:`P_{SR}` — 太阳辐射压强
- :math:`C_R` — 反射系数
- :math:`A/m` — 面质比
- :math:`\hat{\mathbf{r}}_s` — 朝向太阳方向的单位向量

使用示例
--------

.. code-block:: python

   from e2m2e.core.system import CR3BP_System
   from e2m2e.core.dynamics import CR3BP_Dynamics, SRP_Perturbation

   # 创建系统与动力学对象
   system = CR3BP_System.from_known_system("earth_moon")
   dynamics = CR3BP_Dynamics(system)

   # 计算加速度
   state = [0.8, 0, 0, 0, 0.6, 0]
   accel = dynamics.compute_acceleration(state)
   print(f"accel = {accel}")

   # 计算状态转移矩阵
   stm = dynamics.compute_stm(state)
   print(f"STM shape = {stm.shape}")

   # 添加 SRP 摄动
   srp = SRP_Perturbation(CR=1.2, Am=0.01)
   dynamics_with_srp = dynamics.with_perturbation(srp)
