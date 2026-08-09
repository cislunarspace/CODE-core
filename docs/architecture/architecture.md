# e2m2e 架构设计

> 本文描述 e2m2e 的架构（五层已落地，旧包 core/algorithms/transfer/dfh/io/visualization/proximity 已删除）。设计过程记录见 `docs/architecture-design-discussion.md`，逐项架构决策见 `docs/adr/`。

## 总体定位

e2m2e 是一个**独立的地月空间轨道设计库**，不是任何现有软件的复刻品。它面向"LLM+Agent"自主任务规划范式：大模型负责理解任务意图、分解与编排，e2m2e 负责提供精确可靠的轨道计算工具，通过 MCP 协议被调用。

五个核心能力，是 e2m2e 自己的领域能力，与外部软件无关：

1. **任务轨道设计**——从任务参数（轨道类型、高度、振幅、历元、力模型开关）产出标称轨道。
2. **轨道保持**——沿标称轨道施加脉冲控制，仿真测定轨与控制误差，蒙特卡洛评估。
3. **转移轨道设计**——从出发条件到目标轨道的转移路径（脉冲/小推力/低能量）。
4. **轨道预报**——给定初值与力模型的高精度数值外推。
5. **时空坐标转换**——TDT+GCRS ↔ TDB+EBCRS 等参考系与时间尺度转换。

## 总体分层

按依赖方向由内向外，五层。**内层不感知外层**。

| 层级 | 名称 | 职责 | 实现语言 |
|:---|:---|:---|:---|
| 第1层 | 数据层 `data/` | 星历数据管理、时空参考系数据、数据模板、通用数据类型 | Python（含 SPICE/r2s2 适配器） |
| 第2层 | 数值计算层 `crates/` | 积分器、打靶、牛顿/延拓迭代、Lambert、STM/7D 传播 | Rust |
| 第3层 | 算法层 `algorithm/` | 轨道族、站保控制律、流形、转移编排、名义轨道 | Python（调 Rust） |
| 第4层 | 接口层 `api/` | Facade 门面、配置、Pydantic 模型、MCP、CLI | Python |
| 第5层 | 工具层 `tools/` | 可视化、日志 | Python |

**分层哲学**：

- **Rust = 一切"喂进数字就迭代"的数值方法**：积分、打靶、牛顿/延拓迭代、Lambert、STM/7D 传播。迭代的"问题定义"（约束、自由变量、目标）从 Python 传入，Rust 只管迭代到收敛。
- **Python 算法层 = 一切"需要领域知识构造问题"的编排**：不做数值迭代，只做三件事——①构造问题（选轨道族、定约束、选流形方向）②调 Rust 迭代器 ③解释结果（转 Orbit、算物理量、判收敛）。
- 牛顿/延拓迭代骨架**直接归 Rust**（最终形态）。

## 顶层结构

```
e2m2e/
├── data/          # 第1层 数据层
├── algorithm/     # 第3层 算法层（单数，随设计草案）
├── api/           # 第4层 接口层
├── tools/         # 第5层 工具层
├── integrators.py # 数值层对 Python 的门面（保留顶层）
├── mbse/          # SysML 文档产物（保留独立顶层）
└── _integrators/  # Rust 绑定（内部）
```

`core` 拆散后顶层无 `core`。旧路径（`e2m2e.core`、`e2m2e.algorithms` 等）已删除。

## 第1层 数据层 `data/`

```
e2m2e/data/
├── kernels/           # SPICE 内核管理
│   ├── manager.py     # 加载/缓存/校验
│   └── provider.py    # EphemerisProvider 抽象
├── frames/            # 时空参考系【只留数据】
│   ├── eop.py         # EOP 文件解析
│   ├── leap_seconds.py# 闰秒表
│   ├── r2s2.py        # r2s2 库适配器
│   └── spice_frames.py# SPICE 帧查询
├── templates/         # 数据模板
│   ├── seed.py        # 轨道族种子参数
│   ├── systems.py     # 地月系统标准参数
│   ├── perturbations.py # 摄动开关默认
│   ├── force_config.py# 力模型配置 schema
│   └── enums.py       # 领域枚举
└── types/             # 通用数据类型
    ├── state.py       # 状态向量（类型别名）
    ├── epoch.py       # 时间类型（类型别名）
    ├── orbit.py       # Orbit 数据容器
    ├── trajectory.py  # 轨迹容器、NominalOrbit
    └── ephemeris.py   # EphemerisTable
```

关键设计：

- **EphemerisProvider 抽象**（`data/kernels/provider.py`）：对上层屏蔽数据来源。单点 + 批量两类方法；时间（utc_to_tdb/et_to_utc/utc_to_tai/tai_to_tt/tt_to_tdb/jd_tdb_to_et）、状态（body_position/body_state/body_rotation）、帧（pxform）三类。SPICE 和 r2s2 分别实现。Rust 侧"注入数据"（星历缓存样条表）从批量查询构建。
- **frames 只留数据**（EOP、闰秒、历表句柄）；转换算法归 `algorithm/coordinate/`。时空转换**算法**不在数据层。
- **TDB 作动力学统一时间**：算法层/数值层内部统一用 ET(TDB) 或 JD_TDB；只有接口边界（api/Pydantic/输出格式）才转 UTC。
- **类型系统**：State/Epoch 是类型别名（`State = npt.NDArray[float64]` 形状 (6,)），Trajectory 是真容器类（EphemerisTable/NominalOrbit）。单值→别名，多字段/多列→类。算法层保持 numpy 不强制包装。
- **Orbit** 归 `data/types/orbit.py`：states/times/metadata + 可选 system 绑定 + 手动设 period。
- **NominalOrbit** 归 `data/types/trajectory.py`：FR1↔FR2 数据契约（等间距历元状态表 + Floquet 基 + 投影因子表 + 高次插值器）。
- **领域枚举**归 `data/templates/enums.py`（OrbitFamilyType/ReferenceFrame 等）。

## 第2层 数值计算层 `crates/`

```
crates/
├── propagation/       # 积分器族 + Lambert + solve_ivp
├── forces/            # 力模型 + STM + 7D 增广
├── integrators/       # pyo3 绑定 + 打靶 + 同伦 + solver/
└── spice/             # CSPICE FFI + 缓存
```

关键设计：

- **迭代求解器并入 `e2m2e-integrators` crate**，新增 `solver/` 子模块（newton.rs/continuation.rs），与现有 multiple_shooting/segmented_shooting/homotopy 同类集中。不新开 crate。
- **力模型算法归 Rust**（`e2m2e-forces`），Python 侧 `algorithm/forces/` 保留同名类（参数验证 + to_rust_spec 序列化 + 无 Rust 时的兜底计算）。
- **下沉 Rust 的算法，Python 侧保留同名薄封装**（DifferentialCorrection/Continuation/MultipleShooting 类名保留）：Python 类是"问题构造入口"，迭代循环/收敛判断在 Rust。
- **Rust 不吃 SPICE 句柄**，只吃注入数据（星历缓存样条表）。

## 第3层 算法层 `algorithm/`

```
e2m2e/algorithm/
├── design/            # 任务轨道设计（三段编排）
├── family/            # 轨道族生成（含 halo_family、strategies、注册表）
├── station_keeping/   # 轨道保持（controller + 三控制律 + 误差模型 + 蒙特卡洛）
├── transfer/          # 转移设计（transfer_orbit 编排器 + 数学模块）
├── dynamics/          # System + Dynamics
├── forces/            # 力模型类
├── propagation.py     # 轨道预报薄壳（单文件模块）
├── coordinate/        # 坐标转换算法
├── manifold/          # 不变流形 + 庞加莱截面
├── proximity/         # 相对运动
├── stability.py       # 稳定性
├── normal_form/       # 正规化（可选依赖）
└── nominal_orbit/     # 名义轨道
```

关键设计：

- **轨道族注册表 = 函数形态**：`_REGISTRY: dict[str, Callable]`，注册表值 = 设计函数 `design_xxx(params) -> Orbit`。`design_orbit` 查注册表按族分发。新族 = 写一个设计函数 + 注册。
- **algorithm/design/ 持有三段编排**：family（初猜）→ ephemeris_correction（CR3BP→星历修正）→ propagation（高精度预报）。
- **algorithm/transfer/ 按数学类型组织**：脉冲（lambert/three_body_lambert/multi_impulse）、自然动力学（low_energy/manifold）、低推力（low_thrust/）、任务层（search/optimize/porkchop）。`transfer_design` 编排器按 transfer_type 组合。
- **System + Dynamics 都归 algorithm/dynamics/**（一对，拆开割裂"动力学"概念）；标准参数数据归 data/templates/systems.py。
- **最终形态留 Python 的领域知识模块**：family 种子、站保控制律、manifold 种子/截面、transfer 编排、design 编排、coordinate 转换、normal_form 约化、nominal_orbit。

## 第4层 接口层 `api/`

```
e2m2e/api/
├── facade.py          # Facade 门面：唯一公开顶级入口
├── config.py          # 配置（只管运行环境）
├── models.py          # 公开数据模型（Pydantic）
├── mcp/               # MCP 服务
│   ├── server.py      # MCP 服务器
│   └── tools.py       # 工具注册
└── cli/               # 命令行
    └── main.py        # e2m2e 命令
```

关键设计：

- **Facade 是唯一入口**，方法对应"任务级能力"（粗粒度）。算法层保留细粒度 API（专家用）。
- **纯派生 + 元数据标记**：MCP 工具 = Facade 方法全集；Facade 方法带 `mcp_exposed: bool` 元数据（一档二档 True、三档/辅助 False）。
- **Pydantic 模型全部手写**：输入/输出/错误模型精雕参数单位、默认值、取值域（如 perilune_height∈[100,10000]、duration∈(0,20]）。
- **Facade 返回专属 Pydantic 模型**；MCP 传输层包统一信封（{status, data, error, meta}）。
- **CLI 子命令 = Facade 方法**（mcp_exposed=True 的），参数从同一份 Pydantic 模型生成。CLI 与 MCP 完全对称。
- **MCP 部署形态 = 进程内库为主体 + CLI 薄包装 mcp-serve**：`create_server(facade)` 函数 + `e2m2e mcp-serve` 子命令。一个 Facade 实例 = 一个 server。
- **config.py 构造注入**：`Facade(config=Config(...))`，内部默认从环境变量读；SPICEManager 全局句柄、r2s2 进程单例作为已知限制用 Config 显式管理。

### MCP 工具清单

一档任务级（稳定骨架，会增）：`orbit_design` / `orbit_control` / `transfer_design` / `orbit_propagation` / `spacetime_transform`。

二档子任务级（会增）：`orbit_family_generation` / `orbit_stability` / `transfer_search` / `low_thrust_design` / `manifold_analysis` / `low_energy_transfer` / `relative_motion`。

三档辅助（不注册）：`porkchop` / `normal_form` / `safety` / `visualize` / 格式读写。

## 第5层 工具层 `tools/`

```
e2m2e/tools/
├── viz/               # 可视化（可选依赖 [viz]）
└── logging/           # 结构化日志
```

关键设计：

- **标准 logging + 关键事件键值对、零新依赖**：算法层保持 `logger.info`，打靶/延拓迭代等关键数值事件用键值对；`tools/logging/` 提供配置工厂。
- **日志 ≠ 结果**：算法最终结果在结果对象里，日志只记过程事件。

## 依赖方向规则

```
api/ → algorithm/ + data/
algorithm/ → data/ + _integrators
data/ → 仅外部库（SPICE/r2s2/numpy）
integrators.py → _integrators
tools/ → 任意（辅助，核心不 import tools/）
```

**硬规则**：算法层不 import api/；数据层不 import algorithm/。Pydantic 只在 api/ 边界。CI 跑 import 检查强制。

## 验证策略

- **正确性由物理定义裁决**：解析解对照（二体传播闭合、圆轨道半径不变、Jacobi 常数守恒、STM 行列式=1 辛性质、霍曼转移 Δv 匹配理论值）+ 物理不变量。
- **测试标准允许文献公式/解析值，不允许其他软件运行输出**。Vallado 公式、Richardson 系数是"轨道力学公理"；其他软件输出不是定义。
- **不需要 golden 对照，不需要与其他软件强制对比**。
- **按功能类目组织测试**（`theory`/`integrator`/`force`/`data`/`orchestration`/`interface`/`aux`，见 ADR 0021），正确性由物理定义裁决（ADR 0013）；CI 维持静态门，测试在 release 前跑全量。

## 文档结构

```
docs/
├── index.rst            # 总入口
├── getting-started/     # 安装/快速开始/可视化
├── data/                # 数据层文档
├── algorithm/           # 算法层文档
├── api/                 # 接口层文档
├── tools/               # 工具层文档
├── architecture/        # 架构说明（本文）
├── reference/           # 术语表、ADRs
└── adr/                 # ADR
```

README 加"能力与实现状态"表（每个能力标 已实现/部分/未实现）。占位函数 docstring 写清实现状态。

## 迁移记录（已完成）

迁移已于 2026-08 完成，过程见 `docs/architecture/migration-to-five-layer.md`（历史任务指令）。旧包 core/algorithms/transfer/dfh/io/visualization/proximity 已删除，sys.modules 别名已移除。迁移时留作占位的能力（ECOM 光压、角动量管理、LGA/WSB/HMN 转移、低推力等）现已落地，见 README 能力表。

## 依赖与 extras

- 核心依赖轻量：numpy/scipy/pydantic/r2s2/spiceypy/pyerfa/tqdm。
- `[normal-form]`：sympy/joblib（正规化）。
- `[viz]`：matplotlib 等（可视化）。
- `[mcp]`：MCP 协议层（部署 MCP 服务器时）。
