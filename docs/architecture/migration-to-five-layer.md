# 迁移任务：e2m2e 五层架构落地

> 本指令交给 Codex 执行。执行前先读以下文件，它们是迁移的完整依据：
> - `docs/architecture/architecture.md`（最终形态架构）
> - `docs/architecture-design-discussion.md`（48 问共识，每个模块标注"源：xxx 迁入"）
> - `docs/adr/0011-five-layer-architecture.md`（迁移决策）
> - `docs/adr/0012-dependency-direction.md`（依赖规则）
> - `docs/adr/0013-verification-by-definition.md`（验证策略）
> - `docs/adr/0015-nominal-orbit-coordinate.md`（NominalOrbit 契约）

## 背景

e2m2e 现有顶层结构是历史演进的结果：`core`、`algorithms`、`transfer`、`dfh`、`io`、`visualization`、`proximity`、`mbse`、`integrators`。已完成五层架构设计（ADR 0011），需把现有代码**激进式全量迁移**到新结构。骨架已就位（`e2m2e/data/`、`e2m2e/algorithm/`、`e2m2e/api/`、`e2m2e/tools/`，占位函数 + NotImplementedError）。

## 目标结构（最终形态）

```
e2m2e/
├── data/          # 第1层 数据层（星历/时空数据/模板/类型）
├── algorithm/     # 第3层 算法层（领域编排，单数）
├── api/           # 第4层 接口层（Facade/MCP/CLI）
├── tools/         # 第5层 工具层（viz/logging）
├── integrators.py # 数值层门面（保留顶层）
├── mbse/          # SysML 文档产物（保留独立顶层）
└── _integrators/  # Rust 绑定（内部）
```

**核心：`core` 拆散后顶层无 core；`dfh/` 拆散；`io/` 最终不进 e2m2e；`algorithms` → `algorithm`；`visualization` → `tools/viz`。**

## 硬规则（必须遵守）

1. **依赖方向**（ADR 0012）：api/ → algorithm/ + data/；algorithm/ → data/ + _integrators；data/ → 仅外部库。算法层不 import api/，数据层不 import algorithm/。
2. **Pydantic 只在 api/ 边界**：算法层保持 numpy + 异常，算法函数签名不接受 Pydantic 对象。
3. **类名不统一**：`CR3BP_System`/`CR3BP_Dynamics` 等保留原样，不改 PascalCase/snake_case 风格。
4. **TDB 作动力学统一时间**：算法层/数值层内部用 ET(TDB) 或 JD_TDB，接口边界才转 UTC。
5. **验证靠物理定义**（ADR 0013）：不改用 golden 对照、不与其他软件对拍。测试断言来自解析解/守恒量/文献公式。
6. **日志 ≠ 结果**：算法最终结果在结果对象里，日志只记过程事件。

## 迁移策略

**激进式全量迁移 + sys.modules 别名过渡**：

- 每个新模块实现完整后，在旧包位置建立 **shim**（re-export 新模块），使旧路径 `e2m2e.core`、`e2m2e.algorithms`、`e2m2e.transfer`、`e2m2e.dfh` 等在迁移期间保持可用（169 个测试引用旧路径，不能一次断掉）。
- 具体机制：迁移一个子模块后，旧包同名文件改为 `from <new_path> import *`（或明确的 re-export），旧路径仍能 import。
- 全部迁移完成、测试全绿后，**删除旧包**（core/algorithms/transfer/dfh/io/visualization/proximity），顶层只留新结构。
- 分批次迁移，每批一个 commit、跑通测试再动下一批。

## 分批迁移顺序

### 第 1 批：数据层 `data/`（从 core/ + io/ 迁入）

| 新模块 | 源 | 内容 |
|---|---|---|
| `data/kernels/manager.py` | `core/spice.py` | SPICEManager（加载/缓存/校验/时间转换/状态查询） |
| `data/kernels/provider.py` | 接口化 SPICEManager | EphemerisProvider 抽象（时间/状态/帧三类，单点+批量），SPICE 与 r2s2 两实现 |
| `data/frames/eop.py` | `core/coordinate/gmat_eop.py` | EOP 文件解析 |
| `data/frames/leap_seconds.py` | `core/coordinate/gmat_data.py` 闰秒部分 | 闰秒表 |
| `data/frames/r2s2.py` | `core/coordinate/gcrs_ebcrs.py` 句柄管理部分 | r2s2 适配器 |
| `data/frames/spice_frames.py` | `core/spice.py` 帧查询部分 | SPICE 帧旋转查询 |
| `data/templates/seed.py` | `dfh/cr3bp_orbits.py` 常量 | EARTH_MOON_MU/CHAR_LENGTH/种子 |
| `data/templates/systems.py` | `core/constants.py`、`core/cr3bp_system.py` 参数 | 物理常量/系统标准参数 |
| `data/templates/perturbations.py` | `io/inputs_dac.py` DEFAULT_* | 摄动开关默认 |
| `data/templates/force_config.py` | `core/forces/force_config.py` schema 部分 | 力模型配置 schema（纯数据） |
| `data/templates/enums.py` | `mbse/data/enums.py` | OrbitFamilyType/ReferenceFrame 等 |
| `data/types/state.py` | （已有骨架） | State 类型别名 |
| `data/types/epoch.py` | （已有骨架） | Epoch 类型别名 |
| `data/types/orbit.py` | `core/orbit.py` | Orbit 数据容器（完整实现） |
| `data/types/trajectory.py` | `io/ephemeris.py` EphemerisTable + 新建 | EphemerisTable + NominalOrbit（含插值器） |

**验证**：`uv run pytest tests/architecture/ tests/core/test_orbit.py -q`（新增 data 测试），旧 `tests/core/` 测试仍过（shim 保路径）。

### 第 2 批：算法层 `algorithm/`（从 algorithms/ + core/ + transfer/ + dfh/ 迁入）

| 新模块 | 源 | 内容 |
|---|---|---|
| `algorithm/family/` | `dfh/cr3bp_orbits.py` 六类初猜 + `algorithms/halo_family.py` + `algorithms/halo_initial_guess.py` + `algorithms/lissajous_initial_guess.py` + `algorithms/triangular_initial_guess.py` + `algorithms/strategies/` | 六类初猜函数 + 族行走 + 轨道族注册表（函数形态） |
| `algorithm/design/` | `dfh/design_orbit.py` 链路外壳 | 任务轨道设计三段编排（family → ephemeris_correction → propagation） |
| `algorithm/station_keeping/` | `dfh/control_orbit.py` 编排 + `algorithms/station_keeping/`（special_point/target_point/error_models/monte_carlo） | controller.py 编排 + 三控制律 + 误差模型 + 蒙特卡洛 |
| `algorithm/transfer/` | `transfer/`（25 文件：lambert/three_body_lambert/multi_impulse/low_energy/lowthrust_shooting/lowthrust_collocation/qlaw/porkchop/transfer_search/transfer_optimization/nlp_*/config/cost/propulsion/terminal/mission_assessment/nsga2/solution_database/search_*/transfer.py）+ `algorithms/three_body_lambert.py` | transfer_orbit.py 编排器 + 数学模块（按数学类型组织，ADR 0011） |
| `algorithm/dynamics/` | `core/system.py` + `core/cr3bp_system.py` + `core/ephemeris_system.py` + `core/dynamics.py` + `core/bcr4bp_system.py` + `core/bcr4bp_dynamics.py` + `core/ephemeris_dynamics.py` | System + Dynamics 家族 |
| `algorithm/forces/` | `core/forces/`（force_model/point_mass/third_body/gravity_field/srp/drag/relativistic/thrust/force_config 构建部分） | ForceModel + PhysicalModel 子类 + 推力 |
| `algorithm/propagation.py` | `core/forces/force_model.py` ForceModel.propagate 薄封装 | propagate_orbit 单段能力 |
| `algorithm/coordinate/` | `core/coordinate/` 算法部分（axes/coordinate_system/dynamic_axes/iau_2006/synodic_j2000/rho_bridge/gcrs_ebcrs 转换逻辑） | Axes/Origin/CoordinateSystem + 转换算法 |
| `algorithm/manifold/` | `algorithms/manifolds.py` + `algorithms/sections.py` | InvariantManifold + PoincareSection |
| `algorithm/proximity/` | `proximity/` | relative_dynamics/phasing/safety |
| `algorithm/stability.py` | `algorithms/stability.py` | StabilityAnalysis |
| `algorithm/normal_form/` | `algorithms/normal_form/`（20 文件） | 约化流水线（可选依赖） |
| `algorithm/nominal_orbit/` | 新建 + `algorithms/` 插值相关 | NominalOrbit 插值器 |
| `algorithm/ephemeris_correction/` | `algorithms/ephemeris_correction/`（standard/two_level/homotopy/types） | 星历修正注册表（design 编排的一段，注册表分发保留） |
| `algorithm/solver/`（薄封装） | `algorithms/continuation.py` + `algorithms/differential_correction.py` + `algorithms/multiple_shooting.py` + `algorithms/two_level_multiple_shooting.py` | 下沉 Rust 的算法 Python 侧保留**同名薄封装**（问题构造入口，迭代在 Rust solver/，ADR 0011）。**本期只迁移文件位置 + 保持现有实现**，下沉 Rust 是后续独立工作 |

**验证**：`uv run pytest tests/algorithms/ tests/transfer/ tests/core/ -q`（旧路径 shim 保测试），新增 algorithm/ 对应测试。

### 第 3 批：接口层 `api/`（从 dfh/ 编排 + 新建）

| 新模块 | 源 | 内容 |
|---|---|---|
| `api/facade.py` | （已有骨架） | Facade 门面方法逐个接入 algorithm/ 编排器 |
| `api/config.py` | （已有骨架） | Config 定稿 |
| `api/models.py` | （已有骨架） | Pydantic 模型手写（DesignOrbitRequest/Response 等 + OrbitError） |
| `api/mcp/server.py` | 新建 | create_server(facade)，依赖 [mcp] extra |
| `api/mcp/tools.py` | 新建 | MCP 工具纯派生（扫 Facade 方法 + mcp_exposed 元数据） |
| `api/cli/main.py` | 新建 | e2m2e 命令（子命令 = Facade 方法） |

**验证**：`uv run pytest tests/architecture/ -q`（Facade 占位测试改为行为测试）。

### 第 4 批：工具层 `tools/`（从 visualization/ 迁入）

| 新模块 | 源 | 内容 |
|---|---|---|
| `tools/viz/` | `visualization/` 全部 | PlotConfig/OrbitVisualizer/FamilyPlotter 等 |
| `tools/logging/` | 新建 | 日志配置工厂 |

**验证**：`uv run pytest tests/visualization/ -q`（旧路径 shim）。

### 第 5 批：清理

- 全部迁移完成后，删除旧包：`core/`、`algorithms/`、`transfer/`、`dfh/`、`io/`、`visualization/`、`proximity/`。
- 更新 `e2m2e/__init__.py` 顶层导出为新结构。
- 更新 `docs/` Sphinx 文档引用新路径。
- 更新 `pyproject.toml` extras（`[normal-form]`/`[viz]`/`[mcp]`）。
- 新增 CI import 检查（ADR 0012）：脚本检查 algorithm/ 不 import api/、data/ 不 import algorithm/。

## io/ 的处理（关键）

`io/`（DFH 格式读写）**最终不进 e2m2e**，但当前被 `dfh/design_orbit.py`、`dfh/control_orbit.py`、`algorithms/station_keeping/monte_carlo.py` 引用。处理方式：

1. `EphemerisTable` 等**通用容器**抽到 `data/types/`（第 1 批完成）。
2. DFH 专有格式读写（`inputs_dac.py`、`maneuvers.py`、`sk_statistic.py`、`results.py` 的 parse/write）**保留在 io/ 作临时脚本**，迁移期间继续被 dfh/ 模块引用。
3. 全部迁移完成后，dfh/ 模块已迁入 algorithm/，io/ 的引用消失，**删除 io/**（或移到 `scripts/` 作开发期参考）。
4. `force_mapping.py` 的映射逻辑（DFH 摄动开关 → 配置字典）：输出配置字典是通用契约，映射表可作临时脚本随 io/ 处理。

## 未实现能力（保留占位）

以下对外承诺能力保持占位函数 + NotImplementedError，**不实现**，只确保占位存在且抛错信息含能力名：
- 角动量管理（原 #261，`algorithm/station_keeping/momentum_management`）
- ECOM 光压（原 #253，`algorithm/forces/ecom_solar_radiation_pressure`）
- LGA/WSB 引力辅助转移（`algorithm/transfer/transfer_orbit` 的 transfer_type）
- MCP/CLI 协议层（api/，依赖 [mcp] extra）

## 完成标准

1. 新五层结构完整，旧包删除，顶层无 core/algorithms/transfer/dfh/io/visualization/proximity。
2. 全部测试通过（`uv run pytest -n auto -q`），无回归。
3. 新代码遵循依赖方向规则（CI import 检查通过）。
4. `uv run ruff check .`、`uv run ruff format --check .`、`uv run mypy e2m2e/ --ignore-missing-imports`、`cargo fmt --all -- --check`、`cargo clippy --workspace -- -D warnings` 全过。
5. docs/ 文档引用新路径，README 能力状态表更新。
6. 未实现能力保持占位 + 抛错测试。

## 禁止事项

- 不改架构决策（五层结构、依赖规则、验证策略已定，见 ADR）。
- 不改 `CR3BP_System`/`CR3BP_Dynamics` 等类名风格。
- 不引入 golden 对照、不与其他软件对拍。
- 不加新依赖（除已规划 [mcp] extra）。
- 不顺手重构无关代码（精准改动，diff 最小）。
