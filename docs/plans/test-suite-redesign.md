# 测试套件系统重设计

> 起源 issue #358（简化 `test_low_thrust_end_to_end.py`）；经 grilling 升格为整套测试套件的系统重设计。
> 本文是 **loop-go 的执行依据**：§一 是决策记录（ADR 0021，PR② 落 `docs/adr/`），§二–§六 是执行细节与验收。
> 范围：仅测试侧（`tests/` + 测试相关文档 + pytest 配置）；**不改任何被测源码**。

## 零、背景与共识（一句话）

测试套件不是"端到端太多"，而是"没有系统设计"——速度分层（L1–L4）是摆设（全仓 layer 标记仅 43 处、`l2` 仅 2 次）、`e2e`/`l3`/`slow` 三标记重叠、CI 不跑任何测试、`tests/core` 与 `tests/algorithms`（复数）对应已删除的源包。新原则：**按"验证什么"归类（功能类目），目录镜像源结构，CI 维持静态门、测试在 release 前跑全量。**

## 一、ADR 0021（决策记录，PR② 落 `docs/adr/0021-test-suite-functional-categories.md`）

```markdown
# ADR 0021：测试套件按功能类目组织，废除速度分层

**状态**：建议（待评审）
**日期**：2026-08-09
**关联**：ADR 0013（验证策略）、ADR 0011（五层架构）

## 背景

ADR 0013 定下"正确性由物理定义裁决"，并附一句"测试分层：Rust 单元→Python 单元→集成→物理不变量"。该分层未落地：

- 全仓 ~203 个测试文件，显式标层级仅 43 处，`l2` 只标 2 次；"默认 L1+L2"是空话。
- `e2e`/`l3`/`slow` 三标记描述同一概念（慢/集成），散用。
- CI 不跑任何测试（仅 lint+mypy+层间 import 检查），分层只服务本地手感。
- `tests/core/`、`tests/algorithms/`（复数）对应的源包在五层迁移中已删除，测试按死结构组织。

## 决策

1. **分类轴从"速度/集成深度"换成"验证什么"**：封闭 7 类——`theory`（数理/物理理论）、`integrator`、`force`、`data`（数据层：内核/帧/类型/IO/模板 + 坐标转换）、`orchestration`（层3 算法编排）、`interface`（层4 门面）、`aux`（工具/辅助）。每测试恰好 1 主类。
2. **目录镜像源结构**（导航用），**标记标功能类**（验证什么），两轴分离。`slow`/`spice` 正交于功能类。
3. **废 `l1`/`l2`/`l3`/`l4`/`e2e`**；`addopts` 改 `-m "not slow"`。
4. **CI 维持静态门**（格式/风格/类型/层间 import），**测试在 release 前跑全量**。
5. **`tests/` 按五层重排**（`data/`、`numerical/`、`algorithm/`、`api/`、`tools/`、`mbse/`、`_meta/`），消除死结构。

## 理由

1. 速度不是正确性类别——跑多慢不改变证明了什么。
2. 分层为"per-PR 跳过慢测试"而生；CI 既不跑测试，分层失存在理由。
3. 目录镜像源 → "模块 X 的测试在哪"可预测；功能类交标记 → 不散射同一模块的测试。
4. 编排器 API（`transfer_orbit`/`design_orbit`）的"API 正确"天然需一次真调用；按断言归 `orchestration`/`interface`，`e2e` 作为类别解散（ADR 0013 反 mock，无中间态）。
5. 数据层（容器/IO/模板）验证的是"数据结构与默认值"，独立于物理与编排，单列 `data` 类。

## 结果

- `pyproject.toml` markers 换 7 类 + `slow`/`spice`；`architecture.md §验证策略` 删 L1–L4 段；本 ADR 取代 ADR 0013 中"测试分层"那句（0013 其余不变）。
- 迁移分三 PR：①`git mv` 纯移动（保历史、不改逻辑、不换标记）；②逐文件打功能类标记、去 l1–l4/e2e；③清理结构债（私有符号测试、golden/gmat/dfh 术语、#358 归类）。
```

## 二、分类判据（rubric，驱动 203 个文件的归类）

1. **主类 = 被测对象，不是附带计算。** `transfer_orbit` 测试即便断言质量守恒，主类仍是 `orchestration`（编排器是被测对象，守恒是产物上的健全性断言）；直接构造 `LowThrustShooting` 验质量守恒 = `theory`。
2. **层3 vs 层4 边界**：测 `algorithm/` 内部编排（族/转移/设计/校正/站保）= `orchestration`；测 `api/facade`/`mcp`/`cli`/Pydantic 模型（对外契约）= `interface`。
3. **数值层 vs 算法层**：力模型/动力学/积分器即便经 Python 薄封装调用，主类仍归对应物理类（`force`/`integrator`/`theory`）；薄封装本身的契约测试才归 `orchestration`。
4. **正交标记独立**：恰好 1 个功能类标记 + 任意个 `slow`/`spice`。

**源模块 → 默认主类**（机械分配用，有冲突按 rubric 1 覆盖）：

| 源模块 | 默认主类 |
|---|---|
| `crates/propagation`、`algorithm/propagation` | `integrator` |
| `crates/forces`、`algorithm/forces` | `force` |
| `crates/spice`、`data/kernels`、`data/frames` | `data` |
| `algorithm/coordinate` | `data` |
| `data/types`、`data/templates`、文件 IO | `data` |
| `algorithm/dynamics`（势/Jacobi/定义） | `theory` |
| `algorithm/dynamics`（STM 传播） | `integrator` |
| `algorithm/normal_form`、`algorithm/stability` | `theory` |
| `algorithm/{family,transfer,design,station_keeping,manifold,proximity,solver,ephemeris_correction}` | `orchestration` |
| `api/{facade,mcp,cli,models}` | `interface` |
| `tools/`、`mbse/`、架构 import 检查、ABI/版本元测试 | `aux` |

## 三、目录迁移图（PR① `git mv`）

| 旧位置 | → 新位置 | 规则 / 说明 |
|---|---|---|
| `tests/integrators/` | `tests/numerical/integrators/` | 1:1 |
| `tests/core/dynamics/` + `core/system/` + `core/test_{bcr4bp,dynamics_events,potential}.py` | `tests/numerical/dynamics/` | 动力学/系统/CR3BP 势合并 |
| `tests/core/forces/` + `tests/forces/` | `tests/numerical/forces/` | 所有力模型合并（Rust 实质） |
| `tests/core/spice/` | `tests/data/kernels/` | SPICE 内核=数据层 |
| `tests/core/atmosphere/` | `tests/data/atmosphere/` | 密度模型=数据/模型 |
| `tests/core/orbit/` | `tests/data/types/` | `Orbit` 容器=数据类型 |
| `tests/core/coordinate/` **拆分** | `test_gmat_fixtures`→`tests/data/frames/`；其余→`tests/algorithm/coordinate/` | 帧数据解析 vs 坐标转换算法 |
| `tests/algorithms/coordinate/` | `tests/algorithm/coordinate/` | 与上合并 |
| `tests/algorithms/` 族测试（axial/dpo/horseshoe/lpo/spo/strategies） | `tests/algorithm/family/`（+`/strategies/`） | |
| `tests/algorithms/` 校正/同伦/打靶（differential_correction/homotopy/multiple_shooting/two_level/patch_point/continuation/pal/`*_ephemeris_correction`/dispatch） | `tests/algorithm/correction/` | 校正方法族 |
| `tests/algorithms/{manifolds,sections}` | `tests/algorithm/manifold/` | |
| `tests/algorithms/{stability,stability_branches}` | `tests/algorithm/stability/` | |
| `tests/algorithms/{normal_form,station_keeping}/` | `tests/algorithm/{normal_form,station_keeping}/` | 1:1 |
| `tests/algorithms/{propagation,spacetime_convert}.py` | `tests/algorithm/{propagation,coordinate}/` | |
| `tests/transfer/` | `tests/algorithm/transfer/` | 1:1（含 #358 文件） |
| `tests/orbit_design/` | `tests/algorithm/design/`（`scenarios/` 保留） | |
| `tests/proximity/` | `tests/algorithm/proximity/` | 1:1 |
| `tests/api/` | `tests/api/` | 不动 |
| `tests/visualization/` + `tests/tools/` | `tests/tools/viz/` + `tests/tools/` | |
| `tests/format/` | `tests/tools/format/` | |
| `tests/mbse/` | `tests/mbse/` | 不动 |
| `tests/architecture/` + `tests/test_init_version.py` + `tests/test_rust_abi.py` | `tests/_meta/` | 元测试（import 检查/ABI/版本） |

两个 grab-bag（`core/`≈66、`algorithms/` 根≈32）按上表规则机械归位；除 `core/coordinate/` 一处拆分需逐文件看，其余整目录归位。`conftest.py` 随其目录一起移动。

## 四、文档同步（PR②）

| 文件 | 当前内容 | 改成 |
|---|---|---|
| `pyproject.toml` `[tool.pytest.ini_options]` | `markers` 含 l1/l2/l3/l4/e2e/slow/spice；`addopts = ["-m","not l3 and not slow"]` | markers 换 7 类 + `slow`+`spice`；`addopts = ["-m","not slow"]`；更新默认排除注释 |
| `CLAUDE.md` 构建与测试段（line 48） | "默认只跑 L1+L2…单跑某层 `-m l1/l2/l3/l4`" | "默认排除 `slow`；按功能类选跑 `-m theory/data/...`；release 前全量 `-m ""`；spice 默认需 `make setup`" |
| `docs/architecture/architecture.md` §验证策略（line 187–192） | "测试分层：Rust 单元→Python 单元→集成→物理不变量" | 删 L1–L4 段，换"按功能类目 + 按定义（ADR 0013）+ release 跑全量"，指向 ADR 0021 |
| `docs/adr/0013-verification-by-definition.md` | 含"测试分层"那句 | 该句标注"已被 ADR 0021 取代"（0013 其余不变） |
| `README.md` §测试与代码规范（line 166–171） | `make test` / `uv run pytest tests/` | 加一行功能类目与 ADR 0021 指引（轻量） |
| `docs/adr/0021-test-suite-functional-categories.md` | （不存在） | 新建，内容 = §一 |

**CONTEXT.md 不存在**（CLAUDE.md 政策"根 CONTEXT.md"未落地）；本设计不创建它。若后续要建，另开任务。

## 五、PR 分解与验收

> 本 worktree 缺 CSPICE、无法构建 Rust 扩展，**测试无法实跑**。验收以"可静态核验"为准：`pytest --collect-only`（无 import 错、新位置可发现）+ `ruff`/`mypy`/`cargo fmt`/`clippy` + 标记审计（grep）。实跑留到 release 前或主 checkout。

### PR①  目录迁移（纯 `git mv`）

**做**：按 §三 表 `git mv`；`conftest.py`/`__init__.py` 随目录移动；不动任何逻辑、不换标记、不改 import（pytest 的 `pythonpath=["tests"]` + 各 conftest 保持相对引用可用；`kernel_helpers` 仍可导入）。

**验收**：
- `find tests/core tests/algorithms -name '*.py'` → 空（死目录消失）。
- `uv run pytest --collect-only -q` → 无 collection error，测试数 = 迁移前。
- `uv run ruff check .` / `mypy` / `cargo fmt --check` / `cargo clippy` → 全绿。
- diff 仅 `git mv`（`git diff --stat` 全是 rename，无内容改动）。

### PR②  标记换轨 + 文档同步 + 落 ADR

**做**：
1. 逐文件按 §二 rubric + 默认表打**恰好 1 个**功能类标记（`pytestmark = pytest.mark.<类>` 或文件级），删 `l1`/`l2`/`l3`/`l4`/`e2e`；保留 `slow`/`spice` 正交标记。
2. 改 `pyproject.toml` markers + addopts（§四）。
3. 落 `docs/adr/0021-...md`（§一）；同步 CLAUDE.md / architecture.md / 0013 / README（§四）。

**验收**：
- `grep -rE "pytest\.mark\.(l1|l2|l3|l4|e2e)|markers.*\b(l1|l2|l3|e2e)\b" tests/ pyproject.toml` → 空。
- 每个测试文件**恰好 1 个**功能类标记（脚本核验：统计每文件 `{theory,integrator,force,data,orchestration,interface,aux}` 标记数 = 1）。
- `pytest --collect-only` 全绿；`-m theory`/`-m orchestration` 等各类可独立选中非空子集。
- 文档同步项（§四 六处）全部落地。

### PR③  清理结构债

**做**（每项独立，可拆子 PR）：
1. **#358 归类**：`test_low_thrust_end_to_end.py`（现 `tests/algorithm/transfer/`）— `converges` 留作 `orchestration` smoke（module fixture 去重，3×→1×）；质量单调/throttle∈[0,1] 下沉到 `test_lowthrust_shooting`（`theory`）或并入 smoke；dv 对比归 `theory`。
2. **私有符号测试**：把钻 `_ephemeris_to_dict`/`_design_result_to_dict`/`_validate_params` 等私有转换器的测试改测 Facade 公开面；无法避免的记录为例外并注释理由。
3. **术语统一**：`golden`/`gmat`/`dfh` 命名收口——`test_gmat_fixtures`（实为标准数据解析）等按真实归属正名；注释里"DFH"仅作方法来源，不作 oracle。

**验收**：
- #358 文件不再重复 3× 编排（grep `transfer_orbit(` 在该文件出现 ≤1 次于 fixture）。
- 私有符号 import 数下降（除非注释豁免）。
- `make check`（ruff/mypy/cargo）全绿；`pytest --collect-only` 全绿。

## 六、风险与边界

- **不改进范围**：不改任何 `e2m2e/` 源码；不改 CI 工作流（维持静态门）；不建 CONTEXT.md。
- **import 完整性**：PR① 移动后 `tests/conftest.py` 与各 `conftest.py` 的相对路径、`kernel_helpers`（`pythonpath=["tests"]`）须仍可用——`pytest --collect-only` 是把关点。
- **PR① 不夹逻辑**：若移动中必须改某 import 才能跑通，停下报告（说明目录依赖比预期复杂），不悄悄改。
- **标记审计脚本**：PR② 验收建议写一个一次性脚本核验"每文件恰好 1 功能类标记"，避免手工漏标。
