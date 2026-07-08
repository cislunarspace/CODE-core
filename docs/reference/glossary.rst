术语表
======

CR3BP 相关术语
---------------

.. glossary::

   CR3BP
      圆型限制性三体问题 (Circular Restricted Three-Body Problem)

   质量参数
      μ = m₂/(m₁+m₂)，较小天体质量与总质量之比

   拉格朗日点
      旋转坐标系中的五个平衡点 (L1-L5)

   Jacobi 常数
      CR3BP 中唯一的运动积分

   halo 轨道
      围绕拉格朗日点的周期轨道

   状态转移矩阵
      描述轨道扰动传播的矩阵 (STM)

   微分修正
      通过迭代修正初始条件求解周期轨道

   延拓法
      从已知解追踪轨道族的方法

坐标系与模型相关术语
--------------------

.. glossary::

   坐标系
      由坐标轴与原点拼成的数学参考框架（``CoordinateSystem``），描述位置、速度等矢量。被选作运动基准时称参考系，被选作积分框架时称计算系。

   参考系
      被选作"描述运动参数基准"的坐标系。在 e2m2e 中即状态存放与传播的坐标系（``system.coordinate_system``）。

   计算系
      积分所发生的坐标系。回答"积分在哪个坐标系里执行"。在 e2m2e 中与参考系重合。

   参考系的名称
      ``ReferenceFrame`` 枚举（J2000、ICRF、ITRF93 等），命名一个参考坐标系；只标识不做变换。

   中间模型
      按保真度，模型分四级：二体模型 < CR3BP < 中间模型 < 高精度星历模型。中间模型是 CR3BP 加部分摄动改进，低于星历模型。

传播与积分
----------

.. glossary::

   传播
      把运动方程在计算系中积分推进、得到状态历史的整个过程；``Dynamics.propagate`` 的职责（含步长控制、结果提取）。传播调用积分。

   积分
      ``积分器`` 执行单步数值推进的底层动作。积分是单步，传播是全程；前者被后者调用。

运动方程与动力学方程
--------------------

.. glossary::

   运动方程
      描述运动学参数（位置、速度）与时间关系的方程（如 dr/dt = v）。对时间，不对力。

   动力学方程
      描述运动学参数与力关系的方程，一般为微分式（如 dv/dt = F/m）。完整的运动方程组 = 运动方程 + 动力学方程；``Dynamics`` 积分二者合成的状态导数 ODE。

Hamiltonian 正规化
------------------

.. glossary::

   Hamiltonian 正规化
      通过逐层坐标变换消去 CR3BP 平动点附近运动方程中的非线性耦合项，把轨道动力学化简为少数几个几乎不变的参数。e2m2e 用 ``NormalFormPipeline`` 一行调用完成，上下文在 ``NormalFormContext``、结果在 ``NormalFormResult``。代码 docstring 也称"法型化"，同义。

   rho 坐标系
      以平动点为原点的 6 维无量纲相对坐标系，状态 ``[ρ, ρ̇]``；正规化各步在此系内进行。

   动力学替代轨道
      受摄（星历）系统里"最接近周期"的轨线，由多重打靶在时间窗口内修正到首尾闭合。``DynamicalSubstituteCorrector`` 产出，见 ``DynamicalSubstituteResult.substitute_orbit``。

   生成函数 W
      连接 rho 坐标与 quasi-Floquet 坐标的近恒等变换的母函数，见 ``DynamicalSubstituteResult.W_poly``。

   quasi-Floquet 变换矩阵
      时变、保辛的矩阵 ``B(t)``，把替代轨道邻域的时变线性化化为常系数实标准形（一双曲方向 + 两中心方向），满足 ``BᵀJB = J``。``QuasiFloquetReducer`` 求解，见 ``QuasiFloquetResult``。

   中心流形
      消去双曲-中心耦合后、只剩中心运动的不变流形，其上轨道不沿双曲方向逃逸。``CenterManifoldReducer`` 经高阶 Lie 变换消耦，生成函数在 ``CenterManifoldResult.W_series``。

   表征参数
      正规化终点产物，作用量-角变量形式的 6 维约化坐标 ``(q1, p1, I2, θ2, I3, θ3)``，理想下为运动积分。与 rho 坐标互为正逆变换（``rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param``），由 ``LibrationCatalogTransformer`` 提供。

MBSE
----

.. glossary::

   MBSE
      基于模型的系统工程（Model-Based Systems Engineering）：以形式化模型为核心、贯穿需求/设计/分析/验证/确认全生命周期的系统工程方法，模型要素间关系显式可追溯。e2m2e 借鉴其思路做组件登记、需求追溯、数据模型与图表生成，见 ``docs/reference/mbse/``。
