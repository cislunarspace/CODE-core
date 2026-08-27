Glossary / 术语表
====================

[English](#glossary) | [简体中文](#中文)

English
-------

CR3BP terms
~~~~~~~~~~~~~~

.. glossary::

   CR3BP (Circular Restricted Three-Body Problem)
      Circular Restricted Three-Body Problem.

   Mass parameter (质量参数)
      μ = m₂/(m₁+m₂): the smaller body's mass over total mass.

   Libration points (平动点)
      Five equilibrium points (L1–L5) of the rotating frame, specific to the
      CR3BP model. L1–L3 are collinear; L4/L5 triangular. e2m2e says "libration
      point", not "Lagrange point".

   Jacobi constant (Jacobi 常数)
      The only integral of motion in CR3BP.

   Halo orbit (halo 轨道)
      Periodic orbit around a libration point.

   DPO (Distant Prograde Orbit)
      Distant Prograde Orbit — prograde distant family near lunar L2.

   Axial orbit (Axial 轨道)
      Collinear-point periodic family based on Gómez's Type B bifurcation,
      distinct from Halo's Type A mechanism.

   State transition matrix (状态转移矩阵)
      Matrix propagating orbital perturbations (STM).

   Differential correction (微分修正)
      Iteratively correcting initial conditions to solve for a periodic orbit.

   Continuation (延拓法)
      Tracing an orbit family from a known solution.

Frames & models
~~~~~~~~~~~~~~~

.. glossary::

   Coordinate system (坐标系)
      Mathematical reference frame assembled from axes + origin
      (``CoordinateSystem``), describing vectors such as position/velocity.
      When chosen as motion baseline it is called 参考系; as integration frame,
      计算系.

   Reference frame (参考系)
      Coordinate system chosen as the baseline for describing motion. In e2m2e:
      where states live and propagate (``system.coordinate_system``).

   Integration frame (计算系)
      The frame in which integration executes — coincident with the reference
      frame in e2m2e.

   Frame names (参考系的名称)
      ``ReferenceFrame`` enum (J2000, INERTIAL, ROTATING, SYNODIC…): names a
      frame; identifies only, performs no conversion.

   Intermediate model (中间模型)
      Four fidelity tiers: two-body < CR3BP < intermediate < high-fidelity
      ephemeris. Intermediate = CR3BP + partial perturbations, below ephemeris.

   ECOM (Empirical CODE Orbit Model)
      Empirical CODE Orbit Model: 9-coefficient empirical SRP model, more
      accurate than a simple cannonball model.

Transfer design
~~~~~~~~~~~~~~~

.. glossary::

   LGA (Lunar Gravity Assist)
      Lunar Gravity Assist — using lunar gravity to rotate the velocity vector,
      saving fuel.

   HMN (Hohmann Transfer)
      Hohmann transfer — minimum-energy two-impulse transfer between coplanar
      circular orbits.

   Lambert problem (Lambert 问题)
      Given two positions and flight time, solve the Keplerian arc joining them.

Propagation & integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Propagation (传播)
      The whole process of integrating equations of motion to a state history;
      ``Dynamics.propagate``'s duty (step control, result extraction).
      Propagation calls integration.

   Integration (积分)
      One low-level numerical step by an ``integrator``. Integration is
      single-step; propagation spans everything.

Equations of motion / dynamics equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Kinematic equations (运动方程)
      Equations relating kinematic variables (position, velocity) to time
      (dr/dt = v); about time, not force.

   Dynamic equations (动力学方程)
      Equations relating kinematics to force, typically dv/dt = F/m. Full set =
      kinematic + dynamic; ``Dynamics`` integrates their combined state-derivative ODE.

Normal form (标准形)
~~~~~~~~~~~~~~~~~~~~

.. glossary::

   Normal form (标准形)
      Layered coordinate transformations eliminating nonlinear couplings near
      CR3BP libration points, reducing dynamics to a few nearly-invariant
      parameters. One-line entry via ``NormalFormPipeline``; context in
      ``NormalFormContext``, results in ``NormalFormResult``.

   rho frame (rho 坐标系)
      6-dim nondimensional relative frame centered on a libration point, state
      ``[ρ, ρ̇]``; all reduction steps happen here.

   Dynamical substitute orbit (动力学替代轨道)
      Nearest-to-periodic trajectory in the perturbed (ephemeris) system,
      corrected to closure within a time window by multiple shooting. Produced
      by ``DynamicalSubstituteCorrector``; see
      ``DynamicalSubstituteResult.substitute_orbit``.

   Generating function W (生成函数 W)
      Generating function of the near-identity map between rho and quasi-Floquet
      coordinates; see ``DynamicalSubstituteResult.W_poly``.

   Quasi-Floquet transform matrix (quasi-Floquet 变换矩阵)
      Time-varying symplectic matrix ``B(t)`` turning time-varying linearization
      around the substitute into constant-coefficient real normal form (one
      hyperbolic + two center directions), satisfying ``BᵀJB = J``. Solved by
      ``QuasiFloquetReducer``; see ``QuasiFloquetResult``.

   Center manifold (中心流形)
      Invariant manifold of pure center motion after killing hyperbolic-center
      coupling; orbits on it don't escape along hyperbolic directions.
      ``CenterManifoldReducer`` decouples via high-order Lie transforms;
      generating functions at ``CenterManifoldResult.W_series``.

   Characterizing parameters (表征参数)
      Endpoint products of reduction — action-angle coordinates
      ``(q1, p1, I2, θ2, I3, θ3)``, ideally integrals of motion. Invertible chain
      with rho coordinates (``rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param``), provided by
      ``LibrationCatalogTransformer``.

MBSE
~~~~

.. glossary::

   MBSE (Model-Based Systems Engineering)
      Model-Based Systems Engineering: formal models spanning
      requirements/design/analysis/verification/validation, relations explicitly
      traceable. e2m2e borrows the mindset for component registration,
      requirement traceability, data models, and diagram generation — see
      ``docs/reference/mbse/``.

中文
----

CR3BP 相关术语
~~~~~~~~~~~~~~~

.. glossary::

   CR3BP
      圆型限制性三体问题 (Circular Restricted Three-Body Problem)

   质量参数
      μ = m₂/(m₁+m₂)，较小天体质量与总质量之比

   平动点
      旋转坐标系中的五个平衡点 (L1-L5)，仅属于 CR3BP 模型。L1、L2、L3 为共线平动点，L4、L5 为三角平动点。在 e2m2e 中用平动点而非拉格朗日点。

   Jacobi 常数
      CR3BP 中唯一的运动积分

   halo 轨道
      围绕平动点的周期轨道

   DPO
      Distant Prograde Orbit，月球 L2 附近的顺行远距离轨道族

   Axial 轨道
      基于 Gómez Type B 分岔的共线平动点周期轨道族，与 Halo（Type A）分岔机制不同

   状态转移矩阵
      描述轨道扰动传播的矩阵 (STM)

   微分修正
      通过迭代修正初始条件求解周期轨道

   延拓法
      从已知解追踪轨道族的方法

坐标系与模型相关术语
~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   坐标系
      由坐标轴与原点拼成的数学参考框架（``CoordinateSystem`` ），描述位置、速度等矢量。被选作运动基准时称参考系，被选作积分框架时称计算系。

   参考系
      被选作描述运动参数基准的坐标系。在 e2m2e 中即状态存放与传播的坐标系（``system.coordinate_system`` ）。

   计算系
      积分所发生的坐标系。回答积分在哪个坐标系里执行。在 e2m2e 中与参考系重合。

   参考系的名称
      ``ReferenceFrame`` 枚举（J2000、INERTIAL、ROTATING、SYNODIC 等），命名一个参考坐标系；只标识不做变换。

   中间模型
      按保真度，模型分四级：二体模型 < CR3BP < 中间模型 < 高精度星历模型。中间模型是 CR3BP 加部分摄动改进，低于星历模型。

   ECOM
      Empirical CODE Orbit Model，9 系数经验光压模型，比简单炮弹模型更精确地描述太阳光压对航天器的作用

转移设计
~~~~~~~~~

.. glossary::

   LGA
      Lunar Gravity Assist，月球引力辅助转移，利用月球引力改变航天器速度矢量以节省燃料

   霍曼转移（HMN）
      Hohmann Transfer，霍曼转移，共面圆轨道间的最小能量双脉冲转移

   Lambert 问题
      给定两端位置和飞行时间，求解连接两点的开普勒轨道弧段

传播与积分
~~~~~~~~~~

.. glossary::

   传播
      把运动方程在计算系中积分推进、得到状态历史的整个过程；``Dynamics.propagate`` 的职责（含步长控制、结果提取）。传播调用积分。

   积分
      ``积分器`` 执行单步数值推进的底层动作。积分是单步，传播是全程；前者被后者调用。

运动方程与动力学方程
~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   运动方程
      描述运动学参数（位置、速度）与时间关系的方程（如 dr/dt = v）。对时间，不对力。

   动力学方程
      描述运动学参数与力关系的方程，一般为微分式（如 dv/dt = F/m）。完整的运动方程组 = 运动方程 + 动力学方程；``Dynamics`` 积分二者合成的状态导数 ODE。

标准形（Normal Form）
~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   标准形（Normal Form）
      通过逐层坐标变换消去 CR3BP 平动点附近运动方程中的非线性耦合项，把轨道动力学化简为少数几个几乎不变的参数。e2m2e 用 ``NormalFormPipeline`` 一行调用完成，上下文在 ``NormalFormContext`` 、结果在 ``NormalFormResult`` 。

   rho 坐标系
      以平动点为原点的 6 维无量纲相对坐标系，状态 ``[ρ, ρ̇]`` ；标准形化简各步在此系内进行。

   动力学替代轨道
      受摄（星历）系统里最接近周期的轨线，由多重打靶在时间窗口内修正到首尾闭合。``DynamicalSubstituteCorrector`` 产出，见 ``DynamicalSubstituteResult.substitute_orbit`` 。

   生成函数 W
      连接 rho 坐标与 quasi-Floquet 坐标的近恒等变换的母函数，见 ``DynamicalSubstituteResult.W_poly`` 。

   quasi-Floquet 变换矩阵
      时变、保辛的矩阵 ``B(t)`` ，把替代轨道邻域的时变线性化化为常系数实标准形（一双曲方向 + 两中心方向），满足 ``BᵀJB = J`` 。``QuasiFloquetReducer`` 求解，见 ``QuasiFloquetResult`` 。

   中心流形
      消去双曲-中心耦合后、只剩中心运动的不变流形，其上轨道不沿双曲方向逃逸。``CenterManifoldReducer`` 经高阶 Lie 变换消耦，生成函数在 ``CenterManifoldResult.W_series`` 。

   表征参数
      标准形化简终点产物，作用量-角变量形式的 6 维约化坐标 ``(q1, p1, I2, θ2, I3, θ3)`` ，理想下为运动积分。与 rho 坐标互为正逆变换（``rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param`` ），由 ``LibrationCatalogTransformer`` 提供。

MBSE
~~~~

.. glossary::

   基于模型的系统工程（MBSE）
      基于模型的系统工程（Model-Based Systems Engineering）：以形式化模型为核心、贯穿需求/设计/分析/验证/确认全生命周期的系统工程方法，模型要素间关系显式可追溯。e2m2e 借鉴其思路做组件登记、需求追溯、数据模型与图表生成，见 ``docs/reference/mbse/`` 。
