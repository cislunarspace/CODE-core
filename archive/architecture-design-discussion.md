# 架构设计讨论记录（已完成）

> 本文记录 e2m2e 架构设计的讨论共识，随讨论推进更新。最终形态见《架构设计.md》用户草案 + 本文共识；代码落地时据此留模板。
>
> 状态：讨论已完成（2026-08），核心架构决策已全部落地。共识条目标 ✅。

## 〇、总体定位（已确认）

**最终形态的 e2m2e 是一个独立的地月轨道设计库，不是"DFH 的复刻品"。** DFH 只是开发历程中的一个参考坐标系，验证不依赖它、运行时不留痕、将来可以不提。五个能力（轨道设计/轨道保持/转移设计/轨道预报/时空转换）是 e2m2e 自己的领域能力，与 DFH 无关。DFH 格式互操作（io/）是临时脚本，最终不进 e2m2e。

## 一、总体分层（已确认）

按依赖方向由内向外，五层。**本设计描述最终形态，不描述过渡路线**。

| 层级 | 名称 | 职责 | 实现语言 |
|:---|:---|:---|:---|
| 第1层 | 数据层 `data/` | 星历数据管理、时空参考系数据、数据模板、通用数据类型 | Python（含 SPICE/r2s2 适配器） |
| 第2层 | 数值计算层 `crates/` | 积分器、打靶、牛顿/延拓迭代、Lambert、STM/7D 传播 | Rust |
| 第3层 | 算法层 `algorithm/` | 轨道族种子/初猜、站保控制律、流形、转移编排、名义轨道 | Python（调 Rust） |
| 第4层 | 接口层 `api/` | Facade 门面、配置、Pydantic 模型、MCP、CLI | Python |
| 第5层 | 工具层 `tools/` | 可视化、日志、测试夹具 | Python |

**依赖规则**：上层可调用下层，下层不感知上层。Rust 数值层不依赖 Python 运行时、不吃 SPICE 句柄（只吃注入数据）。

### 分层哲学（最终形态）

- **Rust = 一切"喂进数字就迭代"的数值方法**：积分、打靶、牛顿/延拓迭代、Lambert、STM/7D 传播。迭代的"问题定义"（约束、自由变量、目标）从 Python 传入，Rust 只管迭代到收敛。
- **Python 算法层 = 一切"需要领域知识构造问题"的编排**：它不做数值迭代，只做三件事——①构造问题（选轨道族、定约束、选流形方向）②调 Rust 迭代器 ③解释结果（转 Orbit、算物理量、判收敛）。
- 牛顿/延拓迭代骨架**直接归 Rust**（最终形态，不是过渡）。

## 二、各层详细设计

### 第1层 数据层 `data/`

```
e2m2e/data/
├── kernels/           # SPICE 内核管理
│   ├── manager.py     # 加载/缓存/校验（源：core/spice.py SPICEManager）
│   └── provider.py    # EphemerisProvider 抽象（SPICE/r2s2 实现，含时间尺度）
├── frames/            # 时空参考系【只留数据】
│   ├── eop.py         # EOP 文件解析（源：core/coordinate/gmat_eop.py）
│   ├── leap_seconds.py# 闰秒表
│   ├── r2s2.py        # r2s2 库适配器（句柄管理）
│   └── spice_frames.py# SPICE 帧查询
├── templates/         # 数据模板
│   ├── seed.py        # 轨道族种子参数（源：dfh/cr3bp_orbits.py 常量）
│   ├── systems.py     # 地月系统标准参数 μ/DU/TU（源：core/constants.py）
│   ├── perturbations.py # 摄动开关默认（源：io/inputs_dac.py DEFAULT_*）
│   ├── force_config.py# 力模型配置 schema（纯数据，源：core/forces/force_config.py）
│   └── enums.py       # 领域枚举（OrbitFamilyType/ReferenceFrame，源：mbse/data/enums.py）
└── types/             # 通用数据类型
    ├── state.py       # 状态向量、轨道根数（类型别名）
    ├── epoch.py       # 时间类型（UTC/TDB/TAI，类型别名）
    ├── orbit.py       # Orbit 数据容器（states/times/system/period）
    ├── trajectory.py  # 轨迹数据容器、NominalOrbit（+Interpolator）
    └── ephemeris.py   # EphemerisTable【通用容器，非 DFH 专属】
```

**已共识**：
- ✅ `frames/` **只留数据**（EOP、闰秒、历表句柄）；转换**算法**归 `algorithm/coordinate/`。
- ✅ `EphemerisTable` 是**通用星历容器**（UTC+GCRS+会合系位置），归 `data/types/`；DFH 读写格式是临时脚本适配器。
- ✅ **data/types/ 类型系统**：State/Epoch 是**类型别名**（`State = npt.NDArray[float64]` 形状 (6,)，Epoch 别名 + 时间尺度说明），Trajectory 是**真容器类**（EphemerisTable/NominalOrbit 有结构）。区分标准：单值→别名，多字段/多列→类。算法层保持 numpy 不强制包装，类型标注可选。
- ✅ **Orbit 归 `data/types/orbit.py`**（纯数据容器，与 EphemerisTable/NominalOrbit 同类）：states/times/metadata + 可选 `system` 绑定（解释单位/坐标系）+ 手动设 `period`（不自动算，需要时由调用方设）。
- ✅ **coord转换最终形态 = 强化现有 Axes/Origin/CoordinateSystem 抽象**（不新增 Frame 抽象）。所有坐标系（含 synodic↔J2000、GCRS↔EBCRS）表达为 Axes + Origin + CoordinateSystem；时空间联合转换（GCRS↔EBCRS 同时换时间尺度）作为 CoordinateSystem 扩展方法。要补：时间尺度作为参考系的一部分统一（现在散在 SPICEManager.utc_to_et/r2s2/gmat_time）。草案 `data/frames/base.py` Frame 抽象**不建**。
- ✅ DFH 输入输出格式（io/）是**临时脚本，最终不进 e2m2e**。
- ✅ **EphemerisProvider 抽象**（`data/kernels/provider.py`）：对上层屏蔽数据来源。单点 + 批量两类方法；时间（utc_to_et/et_to_utc/utc_to_tai）、状态（body_position/body_state/body_rotation）、帧（pxform）三类。SPICE 和 r2s2 分别实现。Rust 侧"注入数据"（ephem_cache 样条表）从批量查询构建。源：现有 SPICEManager 接口化。
- ✅ **时间尺度并入 EphemerisProvider**（不单独 TimeSystem 类）。时间转换（utc_to_tdb/tdb_to_utc/utc_to_tai/tai_to_tt/tt_to_tdb/jd_tdb_to_et）是 EphemerisProvider 方法，走 SPICE 或 r2s2/de440t。
- ✅ **TDB 作动力学统一时间**：算法层/数值层内部统一用 ET(TDB) 或 JD_TDB；只有接口边界（api/Pydantic/DFH 格式）才转 UTC。Soffel：TCB 有 0.47s/年漂移，TDB 只剩 <2ms 周期项。
- ✅ **力模型配置 schema 归 `data/templates/force_config.py`（纯数据）**，`ForceModel.from_config/to_config` 构建逻辑留算法层 forces/。ADR 0004 的 schema（version/forces/type/params）不变，只挪文件位置 + 拆"schema 数据"与"构建逻辑"。DFH 摄动开关映射（io/force_mapping.py）是临时脚本，但输出配置字典是通用契约。
- ✅ **力模型类归 `algorithm/forces/`**（ForceModel/PhysicalModel 子类/FiniteBurn/ImpulsiveBurn）：Python 类是"力模型定义"（参数验证 + to_rust_spec 序列化 + 无 Rust 时的兜底计算），与 Rust `e2m2e-forces` 的 CompiledForce 枚举对应。三层：data/templates/force_config.py（schema）→ algorithm/forces/（ForceModel + from_config）→ Rust e2m2e-forces（加速计算）。

### 第2层 数值计算层 `crates/`

```
crates/
├── propagation/       # 积分器族 + Lambert + solve_ivp（现有 e2m2e-propagation）
├── forces/            # 力模型 + STM + 7D 增广（现有 e2m2e-forces；ECOM 光压待补 #253）
├── integrators/       # pyo3 绑定 + 打靶 + 同伦 + solver/（牛顿/延拓，待下沉）
└── spice/             # CSPICE FFI + 缓存（现有 e2m2e-spice）
```

**已共识**：
- ✅ 牛顿/延拓迭代骨架**直接归 Rust**（最终形态）。
- ✅ **迭代求解器并入 `e2m2e-integrators` crate，不新开 solver crate**：新增 `solver/` 子模块（newton.rs/continuation.rs），与现有 multiple_shooting/segmented_shooting/homotopy 同类集中。代码模板阶段建骨架（docstring 说明最终职责 + 待下沉），Python 的 DifferentialCorrection/Continuation 保持现有实现直到下沉完成。
- ✅ 转移/流形的**编排**留 Python（NLP 是 Python 强项），但 Lambert、7D 增广 + 灵敏度、STM 传播已归 Rust。
- ✅ 流形种子/截面留 Python（领域知识）。

### 第3层 算法层 `algorithm/`

```
e2m2e/algorithm/
├── design/            # 任务轨道设计（三段编排：family→ephemeris_correction→propagation）
├── family/            # 轨道族生成（源：dfh/cr3bp_orbits.py 六类初猜拆入；含 halo_family、strategies、轨道族注册表）
├── station_keeping/   # 轨道保持（controller.py + special_point/target_point/error_models/monte_carlo）
├── transfer/          # 转移设计（transfer_orbit.py 编排器 + lambert/three_body_lambert/multi_impulse/low_energy/low_thrust/search/optimize/porkchop）
├── dynamics/          # System + Dynamics（CR3BP/Ephemeris/BCR4BP）
├── forces/            # 力模型类（ForceModel/PhysicalModel 子类/推力）
├── propagation.py     # 轨道预报薄壳（单文件模块）
├── coordinate/        # 坐标转换算法（IAU2006/synodic↔J2000/GCRS↔EBCRS，Axes 子类方法）
├── manifold/          # 不变流形 + 庞加莱截面（manifolds.py + sections.py）
├── proximity/         # 相对运动（relative_dynamics/phasing/safety，主题 3）
├── stability.py       # 稳定性（StabilityAnalysis，依赖 Rust STM）
├── normal_form/       # 正规化（可选依赖 [normal-form] extra）
└── nominal_orbit/     # 名义轨道（NominalOrbit + Interpolator，FR1↔FR2 契约）
```

**已共识**：
- ✅ **propagate 和 spacetime_transform 是单段能力、不建独立编排器**。`algorithm/propagation.py`（单文件模块，不是目录）= propagate_orbit 薄壳（配 ForceModel + 调 propagate + 输出 EphemerisTable）；spacetime_transform 在 algorithm/coordinate/ 提供统一转换入口。api/ Facade 直接薄封装，无需 transfer_orbit.py 式编排器。
- ✅ **mbse 保留独立顶层 `e2m2e.mbse`**（SysML 文档产物，非运行时架构组件）：需求注册表/追溯矩阵是"架构文档的活链接"，不归 tools/（避免工具层变杂）、不移出（不断需求↔代码链接）。
- ✅ **领域枚举归 `data/`**（OrbitFamilyType/ReferenceFrame 等，源 mbse/data/enums.py 迁移）：枚举是数据，最终归 data/types/ 或 data/templates/。
- ✅ **control_orbit 编排进 station_keeping/controller.py**（新文件，与 special_point/target_point/error_models/monte_carlo 平级）：读输入星历、选控制律（1/2/3 模式）、配理论/真实双力模型、汇总 SK/MANEUVERS/受控星历输出。与 design/ 的"编排器独立"精神一致，不并入 monte_carlo.py（避免其膨胀）。
- ✅ `design_orbit` 单独 `algorithm/design/`。
- ✅ **algorithm/design/ 持有三段编排**：family（初猜）→ ephemeris_correction（CR3BP→星历修正，标准/两层/同伦）→ propagation（高精度预报）。api/ 只做 Pydantic 校验 + 薄调用，编排逻辑不进 api/。
- ✅ 六类初猜函数（design_dro/design_halo/design_nrho/design_lissajous/design_triangular）从 cr3bp_orbits.py **拆进 family/**。
- ✅ **轨道族注册表 = 函数形态**（`_REGISTRY: dict[str, Callable]`，注册表值 = 设计函数 `design_xxx(params) -> Orbit`），不强制类化。`design_orbit` 查注册表按族分发。新族 = 写一个设计函数 + 注册，核心不硬编码族类型（草案 plugin 化精神，避免函数改类过度抽象）。
- ✅ family/ 回答"一条轨道/一族轨道怎么收敛出来"；design/ 回答"任务参数怎么变成标称轨道"。
- ✅ **System + Dynamics 都归 `algorithm/dynamics/`**（System/CR3BP_System/EphemerisSystem + Dynamics/CR3BP_Dynamics/EphemerisDynamics/BCR4BP_Dynamics）：System 和 Dynamics 是一对（system 解释 Orbit、Dynamics 用 system 传播），拆开会割裂"动力学"概念。标准参数数据（μ/DU/TU/平动点值）归 `data/templates/systems.py`——参数数据与模型类分开。
- ✅ **algorithm/transfer/ 按数学类型组织**：脉冲路径（lambert/three_body_lambert/multi_impulse）、自然动力学路径（low_energy/manifold，覆盖 LGA/WSB 数学内核）、低推力路径（low_thrust/）、任务层（search/optimize/porkchop）。DFH 的 HMN/LGA/WSB 是任务语义，对应到 transfer/ 是数学能力组合；api/ 的 transfer_design 接收 transfer_type 内部组合。
- ✅ **transfer_design 编排器 = algorithm/transfer/transfer_orbit.py**：接收 transfer_type（HMN/LGA/WSB/小推力）+ 目标轨道 + TLI 参数，按枚举选路径组合底层数学模块（HMN→lambert+打靶、LGA/WSB→low_energy/three_body_lambert）。与 design/control 编排对称，api/ Facade 薄封装。
- ✅ **NominalOrbit 是 FR1↔FR2 数据契约**（Gómez 8.2.3）：等间距历元状态表 + Floquet 基 + 投影因子表 + 高次插值器（Lagrange r=5~6）。归 `data/types/trajectory.py`。
- ✅ **Floquet 基 + 投影因子由 FR1 预计算**（`design_orbit` 产出自带），控制全程插值不复算。`control_orbit` 控制律从"现算 STM"演进为"插值投影因子"。
- ✅ **proximity（相对运动）归 `algorithm/proximity/`**（relative_dynamics/phasing/safety，主题 3）：领域算法与 station_keeping/family 同类，不是独立顶层包。属二档三档扩展位（如 relative_motion 可能算二档）。
- ✅ **最终形态留 Python 的领域知识模块**：family 种子（Richardson/线性模态/族行走）、station_keeping 控制律（特征点约束/LQR 权重）、manifold（Floquet 模/种子/截面）、transfer 编排（NLP/搜索/路径组合）、design 编排（三段组合）、coordinate 转换（IAU2006/synodic↔J2000/GCRS↔EBCRS）、normal_form 约化（可选）、nominal_orbit（名义轨道+插值）。
- ✅ **coordinate 转换算法最终留 Python**（`algorithm/coordinate/`，Axes 子类方法）；Rust 下沉是后续性能优化，非最终架构必须项。
- ✅ **下沉 Rust 的算法，Python 侧保留同名薄封装**（DifferentialCorrection/Continuation/MultipleShooting 类名保留）：Python 类是"问题构造入口"（约束/自由变量/目标配置），迭代循环/收敛判断在 Rust solver/。算法层编排（design/transfer）不感知下沉，156 测试引用类名不变。
- ✅ **algorithms/ 独立算法归位**：stability → `algorithm/stability.py`（分析工具，依赖 Rust STM）；sections → `algorithm/manifold/`（庞加莱截面是流形一部分）；halo_family → `algorithm/family/`；strategies → `algorithm/family/`（修正策略与轨道族相关）；ephemeris_correction/ → `algorithm/` 注册表（CR3BP→星历修正调度，design 编排的一段）。
- ✅ **normal_form 归位 `algorithm/normal_form/`**（算法层子模块）+ 可选依赖（`[normal-form]` extra）+ 惰性导入 + 不注册 MCP（三档/辅助）。它是分析工具，不是任务能力。
- ✅ `core/coordinate/` 拆两半：转换算法 → algorithm/coordinate/；数据/适配器 → data/frames/。
- ✅ **代码模板 = 未实现能力建占位函数 + NotImplementedError**，只对"对外承诺的能力"建（FR3 三方式、角动量管理、ECOM 光压等）；纯内部算法优化项不建占位。占位模块 docstring 写清实现状态、数学内核、待补内容；Facade 方法也占位（如 transfer_design(transfer_type="LGA") 抛 NotImplementedError，Agent 收到结构化"未实现"错误）。与现有 #253/#261 的 NotImplementedError 一致。

### 第4层 接口层 `api/`

```
e2m2e/api/
├── facade.py          # Facade 门面：唯一公开顶级入口，粗粒度任务方法
├── config.py          # 配置：只管运行环境（内核路径/精度阈值/日志）
├── models.py          # 公开数据模型（Pydantic：输入/输出/错误）
├── mcp/               # MCP 服务
│   ├── server.py      # MCP 服务器（LLM 工具入口）
│   └── tools.py       # 工具注册（由 Facade 方法自动派生）
└── cli/               # 命令行（人类入口）
    └── main.py        # e2m2e 命令
```

**已共识**：
- ✅ Facade 是唯一入口，方法对应"任务级能力"（粗粒度），算法层保留细粒度 API。
- ✅ MCP 只注册粗粒度工具（Agent 不该被 50 个工具淹没），细粒度留给高级用户。
- ✅ 两层粒度：Facade 粗 + algorithms 细。
- ✅ Pydantic 只在 api/ 边界，不进算法层（算法层保持 numpy + 异常）。
- ✅ **Pydantic 模型全部手写**（一档二档三档统一），输入/输出/错误模型精雕参数单位、默认值、取值域（DFH 参数域如 perilune_height∈[100,10000]、duration∈(0,20]），为后续维护质量。
- ✅ 错误在 api/ 翻译：异常 → 结构化错误码（OrbitError 含 code/message/details）。
- ✅ **Facade 返回专属 Pydantic 模型**（DesignOrbitResponse/ControlOrbitResponse 等各任务定义）；**MCP 传输层包统一信封**（{status, data, error, meta}），状态在信封层、数据在专属模型。LLM 看到"调用 orbit_design → {status, data:{...}}"，错误时 {status:"error", error:{code, message}}。
- ✅ MCP tools 由 Facade 方法自动派生（契约单一来源）。
- ✅ **纯派生 + 元数据标记**：MCP 工具 = Facade 方法全集；Facade 方法带 `mcp_exposed: bool` 元数据（一档二档 True、三档/辅助 False）。注册逻辑统一扫 Facade 方法，清单单一来源，一档也会增加。
- ✅ 一档五个任务级工具是稳定骨架；二档/三档后续会补充。
- ✅ **MCP 部署形态 = 进程内库为主体 + CLI 薄包装 mcp-serve**：`api/mcp/server.py` 提供 `create_server(facade)` 函数（进程内、可测试），CLI 加 `e2m2e mcp-serve` 子命令薄启动。一个 Facade 实例 = 一个 server（MCP 工具 = facade 上 mcp_exposed=True 的方法），每 server 独立配置/内核句柄，不共享全局状态（r2s2 进程单例除外，已知限制）。
- ✅ **CLI 方案 A**：CLI 子命令 = Facade 方法（`mcp_exposed=True` 的），参数从同一份 Pydantic 模型生成。CLI 与 MCP 完全对称，加能力 = 加 Facade 方法 + 手写模型。
- ✅ **transfer-orbit-design 保留独立仓库 + 只留 GUI**（废弃 tod/generates 算法脚本层——被 e2m2e CLI 覆盖）。它是 e2m2e 的桌面前端，GUI 参数表单从 e2m2e Pydantic 模型生成。e2m2e 重命名只改 GUI 调用点，不波及其算法脚本（它们要删）。三层：e2m2e api（Facade/CLI/MCP）→ transfer-orbit-design（GUI）→ 用户。
- ✅ config.py 构造注入 Facade（`Facade(config=Config(...))`），内部默认从环境变量读；SPICEManager 全局句柄、r2s2 进程单例作为已知限制用 Config 显式管理。
- ⏳ MCP 工具清单一档二档边界（三梯队）——已提议待确认：
  - 一档任务：orbit_design / orbit_control / transfer_design / orbit_propagation / spacetime_transform
  - 二档子任务：orbit_family_generation / orbit_stability / transfer_search / low_thrust_design / manifold_analysis / low_energy_transfer
  - 三档底层：porkchop / normal_form / visualize（不注册）/ dfh 格式读（不注册）
- ✅ **原则修正：只讨论最终形态，不问中间状态**。transfer_orbit 进一档是最终形态的必然（FR3 完整实现），不再作为待定。

### 第5层 工具层 `tools/`

```
e2m2e/tools/
├── viz/               # 可视化（可选依赖）
└── logging/           # 结构化日志
```

**已共识**：
- ✅ 可视化归 tools/ 且标可选依赖（MCP 无头部署不需要）。
- ✅ **visualization 整体迁 `tools/viz/`**（PlotConfig/OrbitVisualizer/FamilyPlotter 等，matplotlib 标 `[viz]` extra）。
- ✅ **integrators shim 保留顶层 `e2m2e.integrators`**（不迁入任何目录）：数值层（Rust）对 Python 的门面（rk_step/solve_ivp_events/Lambert 等），与 `_integrators`（Rust 绑定）的薄封装区分保留。
- ✅ **不需要 golden 比较器**（golden 是他人引入概念，用户认为不需要）。测试夹具直接放 tests/。
- ✅ **标准 logging + 关键事件键值对、零新依赖**（不引入 structlog）：算法层保持 `logger.info`，打靶/延拓迭代等关键数值事件用键值对（`logger.info("correction_iter", iter=3, error=1e-8)`）；`tools/logging/` 提供配置工厂（Formatter 把键值对转 key=val 或 JSON）。api/config.py 控制级别和 handler。
- ✅ **日志 ≠ 结果**：算法最终结果（收敛标志/迭代次数/残差历史）在结果对象（EphemerisCorrectionResult/OrbitDesignResult）里，日志只记过程事件（调试用），结果对象不因日志改变。

## 三、验证策略（按定义完成任务，不依赖外部对照）

- ✅ **顶层结构（最终形态）**：`e2m2e/data`（第1层）+ `e2m2e/algorithm`（第3层，**单数随草案**）+ `e2m2e/api`（第4层）+ `e2m2e/tools`（第5层）+ `e2m2e/integrators.py`（数值层门面，保留顶层）+ `e2m2e/mbse`（SysML 文档产物，保留独立顶层）+ `_integrators`（Rust 绑定内部）。**core 拆散后顶层无 core**；`e2m2e.algorithms` → `e2m2e.algorithm`（旧名 sys.modules 别名过渡）。
- ✅ **Rust 绑定模块保留 `e2m2e._integrators`**（内部模块名不对外承诺，用户经 e2m2e.integrators 门面调用）；不改 `_core`（core 在新架构已拆散，改名易混淆）。
- ✅ **extras 依赖分组**：`[normal-form]`（sympy/joblib，保留现状）+ `[viz]`（matplotlib 等，tools/viz）+ `[mcp]`（MCP 协议层，部署 MCP 服务器时装）。核心依赖轻量（numpy/scipy/pydantic/r2s2/spiceypy/pyerfa/tqdm 必需），MCP 是服务化封装可选项，CLI mcp-serve 缺依赖时给清晰报错。
- ✅ **文档结构（最终形态）**：Sphinx（现状）docs/index + getting-started + data/ + algorithm/ + api/ + tools/ + reference/ + adr/。**README 加"能力与实现状态"表**（每个能力标 已实现/部分/未实现），Sphinx 模块 docstring 标注实现状态，占位函数 docstring 写清"实现状态：未实现，待补 XXX"。**docs/architecture/**（新）放架构说明 + 讨论记录。
- ✅ **命名规范**：不统一类名风格（`CR3BP_System`/`CR3BP_Dynamics` 保留——三体问题文献惯例，且重命名已是大变更）；新五层包名全小写 + 下划线（data/algorithm/api/tools）；新增代码遵循 CLAUDE.md（snake_case 函数、PascalCase 类）。
- ✅ **依赖方向规则（硬规则）**：api/ → algorithm/ + data/；algorithm/ → data/ + _integrators；data/ → 仅外部库（SPICE/r2s2/numpy）；integrators.py 门面 → _integrators；tools/ → 任意（辅助，核心不 import tools/）。**算法层不 import api/**、**数据层不 import algorithm/**。Pydantic 只在 api/ 边界，算法层用 numpy/dataclass。CI 跑 import 检查强制。
- ✅ **模块骨架模板**：每个最终形态模块文件都有——①模块 docstring（职责 + 依赖 + 实现状态 已实现/部分/未实现）②公共 API 签名完整（未实现时函数体抛 NotImplementedError）③完整类型标注（mypy 可过）④测试存根。
- ✅ **占位函数建抛错测试（B）+ 按模块一个占位测试文件**：占位测试只断言"调用抛 NotImplementedError + 错误信息含能力名"；实现完成后改成正常行为测试。按模块一个 test_xxx_placeholder.py，不逐个函数建。
- ✅ **proximity 的 MCP 归属**：`relative_motion` 标二档（mcp_exposed=True，交会接近是 Agent 能直接用的任务级能力）；`safety` 标三档（保持点安全偏分析，不注册）。
- ✅ **设计树走完（48 问）**：核心架构决策已全部确认，进入整理阶段——把讨论记录整理成正式架构文档 + ADR + 模板骨架。

## 三、验证策略（续）——正确性与测试标准

- ✅ **正确性由物理定义裁决**：解析解对照（二体传播闭合、圆轨道半径不变、Jacobi 常数守恒、STM 行列式=1 辛性质、霍曼转移 Δv 匹配理论值）+ 物理不变量，这些是"定义"，算对了自然满足。
- ✅ **测试标准允许文献公式/解析值，不允许其他软件运行输出**。Vallado 公式、Richardson 系数等是"轨道力学公理"（定义的一部分）；跟 DFH 等软件跑一遍比对是"别的软件输出"，不需要。
- ✅ **不需要 golden 对照，不需要与 DFH 强制对比**。golden 是他人引入概念，按定义能完成任务即可。
- ✅ DFH 仅作**开发期交叉参考**（本地手动跑，诊断量级/系统性偏差，如"光压差异 97% 来自 ECOM vs 炮弹"），比对脚本放 scripts/ 不进 CI、不进发布包。
- ✅ 现有代码中"数值对拍"类 golden（generate_dfh_golden.py、_dro/_nrho_golden_*.py）随 io/ 一起不入最终架构；"格式契约"类（inputs-dac.golden 逐字节比对）属 DFH 格式临时脚本，随 io/ 走。
- ✅ **测试分层**：Rust 单元（数值方法 vs 解析解）→ Python 算法单元（种子形状/控制律解析解/误差模型统计）→ 集成（跨层链路 + 物理量）→ 物理不变量贯穿（Jacobi 守恒/STM 辛/闭合性）。

## 四、迁移路径（过渡路线，非最终形态）

- ✅ 激进式全量重命名：现有 core/algorithms/transfer/dfh/io/visualization → 新五层。
- ✅ 用 sys.modules 别名过渡（旧路径不破坏既有 import）。
- ✅ 按依赖序分批重命名（先 data → 数值/算法 → api/tools），每批一 commit、跑通该批测试再动下一批（避免"大爆炸"）。
- ✅ 现有 HEAD（FR1-FR5 全绿）是产品基线，重命名后每批回归。
- ✅ 未实现功能（ECOM 光压 #253、角动量管理 #261、LGA/WSB、星历转移闭合）：代码留模板 + 用法文档说明 + README 说明。

## 五、参考书要点（设计依据）

### Soffel《时空参考系》
- 时间尺度链：UTC→TAI(+闰秒)→TT(TAI+32.184s)→TDB(Fairhead-Bretagnon/de440t)。TDB 作动力学统一时间（TCB 有 0.47s/年漂移）。
- GCRS 与 ICRF 差 ~20mas 框架偏差（现有 SynodicJ2000System 混用两者，地月轨道约 70m 差异）。
- 数据/算法分离：EOP、闰秒表、历表是数据绝不能硬编码进内核；转换链算法归内核。
- SOFA/NOVAS 是参考系算法事实标准，自研 Rust 应以 SOFA 为算法参照 + 交叉验证。

### Gómez《平动点任务设计》vol I（共线）
- 模型层阶：CR3BP → 椭圆 RTBP → 受扰解析 → 真实星历，每层是上一层的扰动。
- 多重打靶病态闭合条件：条件数 ~λ1^N，必须反馈绑定（首尾 x 位移、投影因子）。
- NRHO 不能靠解析展开（EM 情形收敛域外），必须数值延拓。
- 名义轨道 = 等间距历元状态表 + Floquet 基 + 投影因子表 + 插值（FR1↔FR2 数据契约）。
- 收敛参数基线：打靶 1e-6、细化 1e-7、数值微分步长 1e-6、RK78 局部误差 1e-13。

### Gómez vol II（三角平动点）
- L4/L5 五级模型链：RTBP→双圆→中间→简化→真实。级数展开不可行（§4.6），必须数值。
- L4/L5 是"动力学替身"问题：太阳周期强迫产生 m·T_S 周期轨道。
- 地月 μ 紧邻共振边界 μ₃=0.013516，数值困难。
- A/B/C/F/G 轨道初值与特征值是独立 golden 基准。

### 软件工程三书
- 架构 = 需求 + 功能架构 + 物理架构；结构单元充分规约后才编码。
- FFI 只暴露 POD（扁平数组 + 错误码），不穿透内部类型。
- 测试 70/20/10；小型测试隔离无依赖；golden 只证回归不证正确。
- 性能是设计问题不是测试问题；性能基准在架构阶段建立。
- 每次变更后回归（原则 196）；三档基线（需求/分配/产品）。
- 信息隐藏/封装（原则 65/80）；低耦合高内聚（原则 73）。

## 六、待定项清单

以下项在追问过程中已解决，全部转为✅（见上文对应条目）：MCP 工具清单一档/二档边界（一档稳定、二档三档会扩）；transfer_orbit 进一档（最终形态必然）；golden 不建（无待定位置）；Facade 覆盖二档（纯派生）；logging 用标准 logging + 键值对（不用 structlog）；文档结构（Sphinx 新五层 + README 能力状态表）；模板形态（占位函数 + 抛错测试）。

以下三项在后续实现阶段均已解决：
1. ✅ 迁移的**分批顺序细目**——见 `archive/migration-to-five-layer.md`（迁移已完成，旧包已删）
2. ✅ 每个 Facade 方法的 Pydantic 模型字段清单——`e2m2e/api/models.py` 已落地（DesignOrbitRequest/Response 等）
3. ✅ 未实现能力的完整清单——ECOM 光压、角动量管理、LGA/WSB/HMN 转移、低推力均已实现，见 README 能力表
