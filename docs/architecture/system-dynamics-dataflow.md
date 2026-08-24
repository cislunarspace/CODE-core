# System 与 Dynamics：两棵类层次的数据流

`e2m2e/algorithm/dynamics/` 下有两棵类层次：System 一侧描述这是什么系统
（μ、特征尺度、平动点、星历、单位、坐标系），Dynamics 一侧描述怎么把它积分
出来（积分器、容差、STM、事件、结果缓存）。包 docstring 概括为 System（数据
上下文）+ Dynamics（传播编排）（e2m2e/algorithm/dynamics/\_\_init\_\_.py:1）。
本文按数据流动的方向，把这两个家族各自是什么、谁读它们的什么成员、一次传播里
数据走哪条路，逐一讲清楚。文中所有行号以当前代码为准。

## 从一次传播说起

先看最常见的一个场景：传播一条地月 CR3BP 轨迹。测试套件里它的标准形态是
（tests/conftest.py:13-25）：

```python
system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()
dynamics = CR3BP_Dynamics(system=system)
result = dynamics.propagate(state0, (0.0, 6.3))
```

数据在这条链上走四段路。

**第一段，构造 System。** `CR3BP_System.__init__` 只收质量参数 μ、两个天体名和
可选的天体半径，特征长度、特征时间、五个平动点全部置为 None
（e2m2e/algorithm/dynamics/cr3bp_system.py:71-128）。此时的系统还不能参与计算：
`DU`/`TU`/`VU` 属性在尺度未设时抛系统未初始化
（cr3bp_system.py:194-212）。`_with_default_scales()` 按天体对补上特征尺度
（cr3bp_system.py:130-160），平动点则推迟到首次需要时才解算：
`get_libration_point` 发现没算过就先调 `compute_libration_points`
（cr3bp_system.py:293-317）。System 的构造因而是两段式的：先定哪个系统，
再定用什么尺度量化它。

**第二段，构造 Dynamics。** `CR3BP_Dynamics(system)` 做的事很轻：存下 system
引用，填一套默认积分器配置（RK45、rtol/atol 1e-12、max_step 0.01），把结果缓存
置空（e2m2e/algorithm/dynamics/dynamics.py:78-97、544-553）。它不复制系统的任何
参数。μ 在每次算加速度时经 `self.system.mu` 现取（dynamics.py:593）。

**第三段，propagate。** 无事件时 CR3BP 走 Rust 快速路径，Python 侧只把 `mu`、
时间区间、初值和积分配置这组标量传过 FFI（dynamics.py:851-859）；轨迹在 Rust
侧算完，以 `{"time", "states"}` 字典回来，同时写进 `self.last_trajectory` 缓存
（dynamics.py:861-870）。这一段的细节见后文propagate 内部一节。

**第四段，结果进数据容器。** 设计链路把返回字典装进 `Orbit`，并把 system 引用
一并塞进去：`Orbit(states=result["states"], times=result["time"],
system=dynamics.system)`（e2m2e/algorithm/design/design_orbit.py:550）。从此这组
浮点数带着自己的单位与坐标系解释者旅行；之后谁想再做计算，可以从
`orbit.system` 重新长出一个 dynamics，`StabilityAnalysis` 正是这样按需重建的
（e2m2e/algorithm/stability.py:95-98）。

四段路合起来就是两个家族的分工：System 是长期持有的模型上下文，Dynamics 是
围着它转的、带配置与缓存的传播者。

## System：模型上下文

### 基类只承诺三个成员

`System` ABC 定义的最小接口只有三项：`frame`（坐标框架）、`unit_system`
（单位系统）、`gravitational_parameter(body)`（引力参数）
（e2m2e/algorithm/dynamics/system.py:15-51）。docstring 点名了什么刻意不进基类：
`mu`、`body_state(body, t)`、`coordinate_system` 属于特定实现的概念
（system.py:25-27）。基类另有一个非抽象的 `get_body_position`，默认抛
NotImplementedError：星历专属能力，放在基类只是给一个明确的报错位置
（system.py:54-66）。

接口薄是有后果的：想对两种系统多态的代码，只能依赖这三个成员，其余都得用
`getattr`/`hasattr` 现探。下文消费面一节会看到这是真实发生的模式。

### CR3BP_System：无量纲、自治、两段式初始化

构造注入五样东西：`mu`、主/次天体名、可选的主/次天体半径
（cr3bp_system.py:71-80）。构造时做两道校验：半径若给必须为正
（cr3bp_system.py:93-98），μ 必须落在 (0, 0.5)（cr3bp_system.py:102-106）。
构造完成后还有两步初始化可选：`set_characteristic_scales(distance, period)` 由
距离与周期推出特征长度/时间/速度并置 `is_initialized`（cr3bp_system.py:214-235）；
`compute_libration_points()` 用 `fsolve` 解三个共线点、解析给出两个三角点
（cr3bp_system.py:237-291，其中 264-266 数值解、268-274 解析解）。

对基类三成员，它的回答是：会合（旋转）坐标系、无量纲单位、约定总质量为 1 时
primary 的 GM 是 `1 - mu`、secondary 的是 `mu`（cr3bp_system.py:163-191）。

初始化之后，这个对象向外提供四类数据：

- 特征尺度：`DU`（km）、`TU`（天）、`VU`（m/s）三个属性，外加
  `characteristic_length/time/velocity` 原始字段（cr3bp_system.py:108-110、194-212）。
- 平动点：`L1`~`L5` 与 `L_points` 字典（cr3bp_system.py:112-117）。
- Jacobi 常数：`get_jacobi_constant(state)`，Parker 约定
  （cr3bp_system.py:319-354）。
- 单位换算与稳定性：`dimensionless_to_physical` / `physical_to_dimensionless`
  （cr3bp_system.py:356-396）、`compute_stability_index`（cr3bp_system.py:398-442）。

半径字段（`primary_radius_km`/`secondary_radius_km`）本身不参与动力学，是碰撞
检测的数据来源，流向见事件与碰撞小节。

### EphemerisSystem：SPICE 查询的统一入口

构造注入五样东西：天体名列表、已完成内核加载的 `SPICEManager`、参考原点
（默认 "EARTH"）、坐标框架（默认 J2000）、可选的 `CoordinateSystem`
（e2m2e/algorithm/dynamics/ephemeris_system.py:35-55）。对基类三成员，它的回答
是：构造时给定的框架、物理单位、GM 直通 `spice.get_gm`
（ephemeris_system.py:59-103）。

与 CR3BP 的两段式初始化不同，这里的第二步是给 `coordinate_system` 赋值，且
发生在构造之后、经 property setter 完成（ephemeris_system.py:69-75）。编排层
的标准写法是先构造再补：`system = EphemerisSystem(...)` 接着
`system.coordinate_system = CoordinateSystem(...)`
（e2m2e/algorithm/propagation.py:118-123）。力模型传播依赖这个字段，不设会被
ForceModel 拒绝（e2m2e/algorithm/forces/force_model.py:57-58）。

星历数据经四个查询方法流出：`gravitational_parameter`/`get_gm`（单体 GM）、
`get_gm_values`（按 bodies 顺序的 GM 数组）、`get_body_position`（位置）、
`get_body_state`（六维状态）（ephemeris_system.py:94-150）。另有一个
`update_coordinate_systems(t, state)` 用于推进动态坐标系
（ephemeris_system.py:77-92）；ForceModel 传播下沉 Rust 后，逐步更新由 Rust
内部完成，e2m2e/ 内已无调用点，仅测试保留
（tests/numerical/forces/container/test_force_model_dynamic_axes.py:143-186）。

### BCR4BPSystem：继承 CR3BP_System，叠加太阳

`BCR4BPSystem` 继承 `CR3BP_System`
（e2m2e/algorithm/dynamics/bcr4bp_system.py:23），构造时多收四个太阳参数：
`sun_mass`、`sun_distance`、`sun_angular_rate`、`sun_phase0`；前两个缺省时按
DE440 常量和日地平均距离推导（bcr4bp_system.py:50-106，缺省推导在 89-92）。
`sun_angular_rate` 特殊：它依赖特征时间，直接构造不给时暂存 None，由
`set_characteristic_scales` 覆写方法按儒略年公转推导（bcr4bp_system.py:148-161）；
因此标准入口是类方法 `BCR4BPSystem.earth_moon()`，一步完成构造与尺度设置
（bcr4bp_system.py:109-132）。

太阳位置不查星历，是时间 t 的解析函数 `sun_position(t)`，即会合系里的共面圆周
（bcr4bp_system.py:163-183）。`gravitational_parameter` 在 "primary"/"secondary"
之外多接受 "sun"（bcr4bp_system.py:185-193）。注意 docstring 的提醒：BCR4BP 无
Jacobi 积分，`compute_libration_points` 给出的是对应 CR3BP 的平动点，仅作参考
位置（bcr4bp_system.py:41-42）。

## Dynamics：传播编排者

### 基类持有的三类状态

`Dynamics.__init__` 只收一个 `system`（dynamics.py:78-84），随后在实例上放三类
东西：

1. **system 引用**：`self.system`，传播全程只读不写（dynamics.py:84）。
2. **积分器配置**：`integrator`、`rtol`、`atol`、`max_step`，默认值挂在类常量上
   （dynamics.py:86-91）。这些是公开字段，调用方构造后直接改：并行搜索按任务
   改容差与步长（e2m2e/algorithm/transfer/search_parallel.py:837-842），多重打靶
   构造后立刻覆写三项（e2m2e/algorithm/solver/multiple_shooting.py:97-101），测试
   fixture 为提速放宽星历传播的容差（tests/conftest.py:92-100）。
3. **结果缓存**：`last_trajectory`、`last_stm`，每次传播结束时覆写
   （dynamics.py:93-94；写入点见后文）。`CR3BP_Dynamics` 再加
   `jacobi_history`/`jacobi_error` 两个 Jacobi 监测缓存（dynamics.py:552-553）。

也就是说，Dynamics 是有态工人：配置与最近一次的结果都留在实例上，供调用方
事后取（如 `compute_state_transition_matrix` 内部就是一次传播后取末态 STM，
dynamics.py:986-999）。

### 模板方法：propagate 是骨架，子类填钩子

`propagate()` 是模板方法：基类定算法骨架（参数规范化 → 校验 → 分发 → 结果
装配），子类经两组钩子参与：`_get_eom_func(with_stm)` 给出 ODE 右端，
`_get_max_step(t_span)` 给步长上限（dynamics.py:100-125）。分发只有两支：
`_propagate_with_stm`（42 维增广状态）与 `_propagate_state_only`（6 维）
（dynamics.py:223-229）。基类的两支实现都走 scipy `solve_ivp`；三个子类各把
这两支覆写为 Rust 快速路径优先（事件场景的处理因子类而异，见后文）。

### 三个子类，两种继承选择

`CR3BP_Dynamics`（dynamics.py:522）与 `EphemerisDynamics`
（e2m2e/algorithm/dynamics/ephemeris_dynamics.py:47）分别实现自治无量纲 CR3BP
方程与含时物理单位 N 体方程。`BCR4BP_Dynamics` 直接继承 `Dynamics`
（e2m2e/algorithm/dynamics/bcr4bp_dynamics.py:42），不继承 `CR3BP_Dynamics`，
尽管它的方程就是 CR3BP 加一项太阳摄动。代码里能看到这个选择的理由：

- 雅可比签名不同：CR3BP 自治，`compute_jacobian_A(state)`（dynamics.py:611）；
  BCR4BP 含时，`compute_jacobian_A(t, state)`（bcr4bp_dynamics.py:123）。
- Jacobi 语义反转：CR3BP_Dynamics 构造即带 Jacobi 监测缓存
  （dynamics.py:552-553），而 BCR4BP 是时间周期系统、无 Jacobi 积分，必须把
  `compute_jacobi_constant` 与 `_handle_jacobi` 都实现为抛 NotImplementedError
  （bcr4bp_dynamics.py:457-463），继承来的能力要逐个点掉。
- STM 入口多一个参数：含时系统的 Φ 依赖起止时刻，
  `compute_state_transition_matrix` 多一个 `t0`（bcr4bp_dynamics.py:438-455）。
- Rust 入口不同：BCR4BP 的传播函数要多带四个太阳参数
  （bcr4bp_dynamics.py:293-303）。

继承 CR3BP_Dynamics 意味着几乎每个公开方法都要覆写、还要压掉 Jacobi 机制。
仓库里有过同型教训：ForceModel 一度形式上继承 Dynamics 只为复用几个数据属性，
实则全部重写 `propagate` 并对 STM/Jacobi 抛错，被认定为 LSP 违反（假继承）而
改为独立类（e2m2e/algorithm/forces/force_model.py:30-38）。

## 消费面：System 出了这个包，被谁读

分离是否承重，取决于 System 被多少不构造 Dynamics 的代码消费。逐个核实如下。

### 力模型侧：读 coordinate_system、spice、origin、gravitational_parameter

`ForceModel` 持有 system（类型标注就是 `Any`），构造时强制
`system.coordinate_system` 已设置（force_model.py:45-58）；传播时把每个力模型
序列化为 Rust 元组（`force.to_rust_spec(self.system)`），并读 `system.origin`
作为 observer 传入（force_model.py:281-293）；`system.spice` 是否存在被用作
资源缺失还是能力缺失的分流依据（force_model.py:226-233）。它从不构造
Dynamics，也明确不继承 Dynamics（force_model.py:30-38）。

`PhysicalModel._resolve_mu(system)` 是力模型侧对 System 最直接的消费：显式 μ
缺失时调 `system.gravitational_parameter(self._body)`
（e2m2e/algorithm/forces/physical_model.py:29-41）。`ConicalShadowModel.
flux_factor(t, state, system)` 经 `require_inertial_frame` 从 system 取出
coordinate_system、spice、原点三件事，再查太阳与遮挡体位置算光照份额
（e2m2e/algorithm/forces/shadow.py:166-197；physical_model.py:69-84）。

### 转移与求解侧：读 origin 与 gravitational_parameter

低推力三件套都把 system 当参数包用。`qlaw_guess(system, ...)` 只从中解析中心体
μ：先查力模型里的 PointMassGravity，查不到再
`system.gravitational_parameter(origin)`，`origin` 本身用 `getattr` 兜底成
"EARTH"（e2m2e/algorithm/transfer/qlaw.py:210、265-282）。
`LowThrustShooting` 与 `LowThrustCollocation` 构造时各做两件事：把力模型
`to_rust_spec(system)` 预序列化，把 `origin` 存为 observer
（e2m2e/algorithm/transfer/lowthrust_shooting.py:125-162；
e2m2e/algorithm/transfer/lowthrust_collocation.py:59-88）。

`NormalFormContext` 从 system 只取 μ，且是 `getattr(system, "mu", None)` 探测：
CR3BP_System 有就取，没有就回退固化常量
（e2m2e/algorithm/normal_form/context.py:64、114、196-201）。

### 坐标转换侧：读 mu、特征尺度、spice

`SynodicJ2000System` 持有 `CR3BP_System` 与 spice：逐点转换读 `mu`（质心平移）
与 `characteristic_time`（时间量纲化），批量路径把 `mu` 与时间单位两个标量传给
Rust（e2m2e/algorithm/coordinate/synodic_j2000.py:28-41、52、96-97）。
`rho_bridge` 的一组函数以 `EphemerisSystem` 为参：读 `system.spice` 构造
SynodicAxes、读 `system.get_body_state("MOON", et)` 取月球状态
（e2m2e/algorithm/coordinate/rho_bridge.py:47-63、110）。

### 轨道保持与预报：构造并配置 EphemerisSystem

`control_orbit` 构造十体 EphemerisSystem、赋 coordinate_system，然后交给
蒙特卡洛流程（e2m2e/algorithm/station_keeping/controller.py:228-247）；工作进程里
按参数重建同一个系统（e2m2e/algorithm/station_keeping/monte_carlo.py:644-650），
另建一个仅带特征尺度的 CR3BP_System 供会合系转换用（monte_carlo.py:66-72、
707-716）。`propagate_orbit` 同样先构造 EphemerisSystem、补 coordinate_system，
再 `ForceModel.from_config(force_config, system)`（propagation.py:118-124）。
这两个模块消费 System，但传播由 ForceModel 完成，不经过 Dynamics。

### 其余散点

`PoincareSection.periapsis(center, system)` 读 `mu` 与主/次天体名来定截面中心
（e2m2e/algorithm/manifold/sections.py:202-228）。`design_orbit` 里同一个 CR3BP
system 实例同时喂给四路消费：构造 `CR3BP_Dynamics`、直接调
`get_jacobi_constant`、读 `characteristic_time` 做时间换算、构造
`SynodicJ2000System` 做坐标转换（design_orbit.py:983-987、1030-1032）。

### 按 System 抽象签名的多态函数

签名标注 `system: System` 的入口有：`PhysicalModel._resolve_mu` /
`to_rust_spec`（physical_model.py:29、45）、`ConicalShadowModel.flux_factor`
（shadow.py:170）、`qlaw_guess` 与 `make_shooter_for_qlaw`（qlaw.py:210、290）、
`LowThrustShooting.__init__` 与 `LowThrustCollocation.__init__`
（lowthrust_shooting.py:125；lowthrust_collocation.py:59）、
`NormalFormContext.__init__`（normal_form/context.py:64）。

把这些入口实际读到的成员摊开看，真正对 CR3BP 与星历两种系统都成立的多态只
剩基类那一项：`gravitational_parameter`，它在 CR3BP 侧接受 "primary"/"secondary"
返回无量纲值，星历侧接受 SPICE 天体名返回 km³/s²（system.py:42-51 的 docstring
把这个双语义写明）。其余入口虽标着 `System`，实际读的都是实现侧成员
（`origin`、`coordinate_system`、`spice`、`mu`），靠 `getattr`/`hasattr` 兜底。
契约测试把双系统同一接口这件事固定下来：同一组断言跑在 CR3BP_System 与
EphemerisSystem 两个实现上（tests/algorithm/dynamics/test_system_contract.py:53-64）。

### 鸭子类型：名义 System，实际结构

类型标注之下，真实契约比注解更薄，三处实证：

- 数据层的 `Orbit`/`OrbitFamily` 把 system 存为 `Any`，docstring 明说数据层
  不依赖算法层，只用 `hasattr(system, "get_jacobi_constant")` 判断能力
  （e2m2e/data/types/orbit.py:8-12、50、418-429）。
- 低推力测试直接传 `SimpleNamespace(origin="EARTH")` 当 system 用
  （tests/algorithm/transfer/test_lowthrust_collocation.py:23；
  tests/algorithm/transfer/test_qlaw_failure.py:37）。
- `RelativeDynamics.linear_model` 用 try/except 在
  `compute_jacobian_A(t, state)` 与 `compute_jacobian_A(state)` 两种签名间适配
  （e2m2e/algorithm/proximity/relative_dynamics.py:145-150）。

## propagate 内部：一次调用里数据怎么动

### 入口编排

`Dynamics.propagate` 本体（dynamics.py:144-233）按固定次序处理输入：

1. `initial_state` 转 ndarray，`max_step` 经钩子取得（dynamics.py:199-200）。
2. 事件规范化：单个 callable 包成列表；空列表等价于无事件，直接置 None
   （dynamics.py:201-205）。
3. 若开碰撞检测，先把碰撞事件造出来并入事件列表，同时检查初始状态是否已在
   某天体半径内（dynamics.py:207-209）。
4. `backend` 校验：只许 "scipy"/"rust"，有事件时必须显式给，否则报错
   （dynamics.py:211-214）。
5. 初始即在半径内的，短路返回单点轨迹加即时碰撞标记：scipy 事件不会对
   g<0 的起点触发，必须显式处理（dynamics.py:217-221、474-493）。
6. 按 `with_stm` 分发到两支（dynamics.py:223-229）。
7. 若开碰撞检测，从结果的事件段提取碰撞信息（dynamics.py:231-232、495-510）。

### scipy 与 Rust 两条后端路径

**scipy 路径**（基类实现）：`_propagate_state_only` 拿 `_get_eom_func(False)`
的 ODE 右端调 `solve_ivp`，`result.y` 转置成 (n, 6)；失败或空结果抛
`PropagationFailure`，不允许拿空轨迹伪装成功（dynamics.py:291-325）。
`_propagate_with_stm` 先把 6×6 单位阵展平拼成 42 维增广状态再积分，回来按
前 6 维/后 36 维拆开（dynamics.py:254-271）。两条 scipy 路径都在返回前写
`last_trajectory`（含 STM 时连 `last_stm`）（dynamics.py:273-274、327）。

**Rust 路径**（子类覆写）：无事件时三个子类都要求 Rust 扩展可用，缺失即抛错、
不降级 scipy。传过 FFI 的东西逐类不同：

- CR3BP：只有 `mu` 一个系统参数，加时间区间、t_eval、初值、容差、步长
  （dynamics.py:851-859；STM 版 dynamics.py:782-790）。
- BCR4BP：`mu` 加太阳四参数 `mu_sun`/`sun_distance`/`sun_angular_rate`/
  `sun_phase0`（bcr4bp_dynamics.py:360-370；STM 版 293-303）。
- 星历：`bodies`、`origin`、`gm_values` 三个从 system 现取的序列
  （ephemeris_dynamics.py:124-126、133-142）。天体位置不经过 Python 对象传递：
  Rust 侧在积分内环直接查进程内 SPICE（先查星历缓存，未命中回退 cspice）
  （crates/e2m2e-spice/src/spk_accel.rs:46-56）。在这条链上，EphemerisSystem
  的角色收敛为一个参数包。

Rust 路径回来后都做长度防御校验（返回点数不等于请求点数即抛错），再写
`last_trajectory`（dynamics.py:792-804、861-870）。

**事件时的第三支**：有事件且 `backend="rust"` 时，CR3BP/BCR4BP 走通用 Rust
积分器 `solve_ivp_events`，ODE 右端仍以 Python 回调形式传入，事件函数被
折算成 `(g, terminal, direction)` 三元组（dynamics.py:700-756、721-723；
bcr4bp_dynamics.py:225-271）。EphemerisDynamics 不支持事件：events 非 None
直接 NotImplementedError（ephemeris_dynamics.py:85-113、162-189）。

### 返回字典的键在何时出现

`time`/`states` 恒有。其余键的出现条件：

- `stm`：`with_stm=True` 时（dynamics.py:276-280）。
- `status`/`cause`：只在纯状态路径出现：基类 scipy 版与 CR3BP 的两个 Rust
  纯状态分支（dynamics.py:329-334、872-877、926-931）。STM 路径、
  EphemerisDynamics 与 BCR4BP_Dynamics 的 Rust 分支都不带这两个键。
- `jacobi`/`jacobi_error`：`with_jacobi=True` 且为 CR3BP 时。基类
  `_handle_jacobi` 是 no-op（dynamics.py:345-358），CR3BP_Dynamics 覆写为逐点
  计算并写缓存（dynamics.py:1012-1022），BCR4BP 覆写为抛错
  （bcr4bp_dynamics.py:461-463）。
- `t_events`/`y_events`：传了 events 时，逐事件的触发时刻与状态
  （dynamics.py:282-284、336-338）。
- `collision`：`collision_detection=True` 时，未碰撞为 None，否则是
  `{"body", "t", "state"}`（dynamics.py:231-232、507-509）。

### 事件与碰撞检测

事件函数是 scipy 语义的 `g(t, state) -> float`，可挂 `terminal`/`direction`
属性；`PoincareSection.event(...)` 是主要的构造者
（tests/algorithm/dynamics/test_events.py:21-26）。`with_stm=True` 时事件函数
收到的是 42 维增广状态（test_events.py:86-97）。

碰撞检测把撞天体翻译成终端事件：`_collision_specs` 从 system 读
`primary_radius_km`/`secondary_radius_km`（都没注入则报错）、读 `mu` 定两天体
在会合系的固定位置 [-μ,0,0] 与 [1-μ,0,0]（dynamics.py:398-428）；
`_setup_collision_detection` 再读 `DU` 把半径从 km 折成无量纲，构造
`g = |r - center| - R` 的 terminal 事件并追加到用户事件之后
（dynamics.py:447-472）。碰撞事件在事件列表末尾这一顺序被
`_extract_collision` 反向利用：按列表尾部 n 个索引回每个天体的触发记录
（dynamics.py:501-509）。这就是碰撞数据流全程读 system 的三个字段：半径、mu、
DU。

### 三条传播链的差异

- **CR3BP**：无量纲、自治。EOM 不显含 t；时间区间、t_eval 都是无量纲量；
  系统侧只贡献 `mu`。
- **BCR4BP**：在 CR3BP 右端上叠加太阳直接项与间接项，太阳位置由
  `system.sun_position(t)` 解析给出，EOM 显式含时
  （bcr4bp_dynamics.py:72-121）。雅可比左下块在伪势能 Hessian 上再加太阳项
  偏导（bcr4bp_dynamics.py:123-159）。无 Jacobi 积分。
- **星历**：物理单位（km、km/s、et 秒）、含时。EOM 对每个天体查 GM、对非原点
  天体查星历位置，原点天体出中心项、其余出第三体摄动加间接项
  （ephemeris_dynamics.py:235-293、256-291）。`max_step` 默认 60 秒且按传播
  时长自适应收紧（ephemeris_dynamics.py:65-83）。Python 侧单步算加速度要逐
  天体过 SPICE；Rust 快速路径把这个查询挪进 Rust 内环，Python 只递参数包。

## 这份分离保护了什么

### 一个 System 服务多个 Dynamics 与别的消费者

System 实例被共享的实证在测试与生产两侧都有。测试侧，BCR4BP 对照实验把坐标
转换器持有的 system 直接拿来再建一个动力学：
`CR3BP_Dynamics(spice_syn_j2000.cr3bp_system)`
（tests/algorithm/dynamics/test_bcr4bp_model.py:168）。生产侧，design_orbit 里
同一个 system 实例同时喂动力学、Jacobi 计算、时间换算、坐标转换四路
（design_orbit.py:983-1032）；`StabilityAnalysis` 与不变流形各自从 orbit 上挂的
system 重建自己的 `CR3BP_Dynamics`（stability.py:95-98；
e2m2e/algorithm/manifold/manifolds.py:114）。反向的共享同样成立：同一
`earth_moon_system` fixture 上既可以挂 `earth_moon_dynamics`
（tests/conftest.py:13-25），也可以被任何只要系统参数的测试直接消费。

与之对照，Dynamics 的配置是按任务改的：同一个搜索任务族内，dynamics 构造后
被覆写 integrator/rtol/atol/max_step（search_parallel.py:837-842）。若配置留在
System 上，共享 system 的消费者之间会互相踩配置；分离让模型参数与本次积分
配置各有其主。

### 多态缝与契约测试

`System` ABC 是 CR3BP 与星历两个世界之间唯一的类型级接缝，缝宽三个成员
（system.py:15-51）。消费面普查显示：真正跨缝多态的调用几乎都收敛到
`gravitational_parameter` 一项，其余访问都是 getattr/hasattr 探测实现侧成员。
两个契约测试文件把这个格局固定下来：`test_system_contract.py` 对两个实现跑
同一组接口断言（tests/algorithm/dynamics/test_system_contract.py:53-64），
`test_dynamics_contract.py` 对 propagate 的输出形状断言 (n, 6) 与 (n, 6, 6)
（tests/algorithm/dynamics/test_dynamics_contract.py:28-38）。

### 生命周期的差异

System 的寿命跟着数据走：`Orbit` 持有 system 引用，序列化再加载后引用还在
（orbit.py:50、253、275），稳定性分析、流形计算可以随时从它再长出 dynamics。
Dynamics 的寿命跟着任务走：构造、改配置、传播、读缓存，然后被丢弃；
`_corrected_dro_cached` 与 `dro_corrector` 两个 fixture 各自建独立的
system+dynamics 对，互不共享（tests/algorithm/conftest.py:62-88）。两类对象的
生命周期差一个量级，这是分离在运行时最直观的体现。

### 两侧的真实重叠

分离并不彻底，有三处已知的重叠，各有来由：

- **Jacobi 常数双入口**。`CR3BP_System.get_jacobi_constant` 是定义所在
  （cr3bp_system.py:319-354）；`CR3BP_Dynamics.compute_jacobi_constant` 一行委托
  （dynamics.py:1001-1010），供 `_handle_jacobi` 在传播后逐点调用。只想要 Jacobi
  值、不想碰传播的调用方走 system 侧（design_orbit.py:987；
  e2m2e/algorithm/family/axial_initial_guess.py:166）。测试断言两入口数值一致
  （tests/algorithm/dynamics/test_cr3bp_model.py:66-69）。
- **A 矩阵两处构造**。`CR3BP_System.compute_stability_index` 内部拼一份 6×6
  线性化矩阵用于平动点特征值分析（cr3bp_system.py:418-423）；
  `CR3BP_Dynamics.compute_jacobian_A` 拼同构矩阵供 STM 变分方程与延拓模块
  复用（dynamics.py:611-638）。两者共用 `pseudo_potential_hessian`
  （e2m2e/algorithm/dynamics/potential.py:14-58），Hessian 只有一份，拼装各
  归各的语境：一个在系统性质里，一个在传播配套里。
- **碰撞半径与 DU 的跨层读取**。碰撞检测是 Dynamics 的职责，但半径存在
  System 上、折无量纲要用 System 的 `DU`（dynamics.py:398-428、447-472）。
  半径是天体属性而非积分配置，故数据留在 system，事件构造留在 dynamics。

## 附：文件地图

- System 侧：`system.py`（ABC）、`cr3bp_system.py`（CR3BP_System +
  LibrationPoint）、`ephemeris_system.py`（EphemerisSystem）、
  `bcr4bp_system.py`（BCR4BPSystem）。
- Dynamics 侧：`dynamics.py`（Dynamics 基类 + CR3BP_Dynamics +
  `propagate_state_at_orbit_time`）、`ephemeris_dynamics.py`
  （EphemerisDynamics）、`bcr4bp_dynamics.py`（BCR4BP_Dynamics）。
- 共享：`potential.py`（伪势能 Hessian，供 CR3BP/BCR4BP 的雅可比与平动点
  稳定性分析共用）。
- 行为写照：`tests/algorithm/dynamics/`（契约、事件、碰撞、变分方程）。
