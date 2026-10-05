# AGENTS.md

本指南面向 AI 助手与贡献者，记录当前代码、构建和验证契约。先读本文件、`CONTEXT.md` 与相关 ADR，再改动；以源码和配置为准，不以旧文档中的工具数量或版本示例为准。

## Agent skills

工程技能在会话中读取 `docs/agents/` 下的仓库级配置；改动这些文件即改变技能行为，需同步本文件。配置本身用中文书写，遵循本文件的语言约定。

- `docs/agents/issue-tracker.md`：issue 与规格的存放位置、`gh` 操作约定、AI 贡献标记、PR 是否作为请求渠道、发布说明的起草要求（以调用方为受众，把 CHANGELOG 对应节提炼成稿，中英双语单正文，条目末尾附 PR 与 issue 编号，小标题取 `CONTRIBUTING.md` 的分类统一表；禁止自动生成 notes 直接交付），以及 GitHub Project 的工作状态配置（Project ID、Status / Priority / Start Date 字段与选项 ID）。由 `/code-review`、`/github-project`、`/merge-pr`、`/open-pr` 按路径读取，`/triage` 读取其中的渠道标记与 Project 配置。
- `docs/agents/triage-labels.md`：五个分诊角色到本仓库标签字符串的映射，供分诊流程按角色取用；映射缺失时 `/triage` 会要求运行 `/setup-ouyangjiahong-skills`。
- `docs/agents/domain.md`：探索代码库时如何读 `CONTEXT.md` 与 `docs/adr/`、如何按术语表用词、如何标注 ADR 冲突。供探索代码库的技能使用；术语与决策落定时由 `/domain-modeling` 更新 `CONTEXT.md` 与 ADR。
- `docs/agents/multi-session-concurrency.md`：多会话并发约定。

## Project Overview

`e2m2e`（Earth to Moon, Moon to Earth）是地月空间算法工具集，覆盖 CR3BP 任务轨道与轨道族、星历传播、转移设计、轨道保持、时空坐标转换、轨道库和地月空间分区分析。Python API 编排领域逻辑，Rust/PyO3 扩展 `e2m2e._integrators` 承担数值密集路径。

调用方入口：

- 进程内：`from e2m2e.api import Facade`，并通过 `Facade().catalog`、`Facade().spatiography` 使用轨道库和分区能力。
- CLI：`e2m2e <工具名>`；部署命令为 `e2m2e mcp-serve` 和 `e2m2e serve-stdio`。
- MCP 与 GUI sidecar 的工具面、请求模型和描述均从同一接口元数据派生。

## Architecture & Data Flow

### 分层与依赖方向

源码由底至顶为：

1. `e2m2e/data/`：常数、SPICE 内核、帧数据、模板、通用类型和轨道库。
2. `crates/` 数值层：传播、力模型、SPICE FFI、水平集/HJB，以及 `e2m2e._integrators` 的 PyO3 绑定。
3. `e2m2e/algorithm/`：构造领域问题、选择轨道族和约束、调用数值后端、解释结果。
4. `e2m2e/api/`：Facade、Catalog、Spatiography、Pydantic 边界和传输适配器。
5. `e2m2e/tools/`：日志等辅助能力。

`e2m2e/mbse/` 是独立的系统工程模型子系统，不属于上述调用链。包根的 `exceptions.py`、`status.py`、`spice_ext.py`、`integrators.py` 是共享叶模块，不得导入任何业务层。

硬门禁在 `scripts/check_layer_imports.py`：`data` 不得依赖 `algorithm/api/tools/integrators/mbse`，`algorithm` 不得依赖 `api/tools/mbse`，`api` 不得依赖 `tools/mbse`。新增 import 前先检查这一边界。

算法层通常由 Python 构造和编排问题，积分、力模型、族生成、打靶和搜索等性能敏感路径下沉 Rust；仓库仍保留显式选择的 SciPy 路径，不要把“Rust 快速路径”误改成无条件替换。Rust 扩展要求在使用处校验 ABI 和符号，缺失时抛错，不得静默回退到 Python/scipy。

### 工具执行链

```mermaid
flowchart LR
    Caller[进程内调用方] --> F[Facade / Catalog / Spatiography]
    Transport[CLI / MCP / sidecar] --> E[execution.execute_tool]
    E --> V[mcp.envelope 参数校验与异常翻译]
    V --> F
    F --> A[algorithm 编排]
    D[data] --> A
    A --> I[integrators / spice_ext]
    I --> R[e2m2e._integrators PyO3]
```

- `e2m2e/api/facade.py` 的 `mcp_exposed`、`tool_inventory` 和 `resolve_tool_method` 是工具元数据唯一来源。MCP、CLI、sidecar 不得维护第二份工具清单或手写工具数量；描述取方法 docstring 首行。
- `e2m2e/api/execution.py::execute_tool` 是传输层工具调用的共享执行核心：规格查找、Pydantic 校验、调用、错误翻译和可选二进制帧抽取都在这里统一。进程内 API 仍可直接调用暴露类方法。
- `transfer_design`、`orbit_family_generation` 属于长任务，MCP/CLI/sidecar 经 `e2m2e.api.mcp.worker` 子进程执行；取消通过 kill 子进程，不要试图在线程中强行终止 Rust 计算。
- 统一文本信封为 `{status, data, error, meta}`。sidecar 的大数组使用 `e2m2e/api/frames.py` 唯一实现的 `E2M2` 帧，支持 `f32`/`f64`；不要在传输适配器中复制帧编码逻辑。

## Key Directories

| 路径 | 用途 |
|---|---|
| `e2m2e/data/` | 常数、内核、帧、模板、类型、catalog；`constants/constants.toml` 是物理常量源文件 |
| `e2m2e/algorithm/` | design、family、dynamics、forces、solver、transfer、coordinate、manifold、station_keeping、spatiography 等编排 |
| `e2m2e/api/` | `facade.py`、`models/`、`execution.py`、MCP、CLI、sidecar 和配置 |
| `e2m2e/tools/` | 日志等辅助工具 |
| `e2m2e/mbse/` | 独立的需求、架构、数据和图表模型 |
| `crates/` | Rust workspace；`cspice`/`cspice-sys` 绑定来自 git 依赖 cislunarspace/cspice-rs（ADR 0057），无本地 vendor |
| `tests/` | 按 data、numerical、algorithm、api、tools、mbse、`_meta` 镜像分层 |
| `scripts/` | 构建资源下载、架构检查、基线生成、benchmark 和手工诊断 |
| `docs/adr/` | 不可静默覆盖的架构决策记录；`CONTEXT.md` 是唯一领域术语表 |
| `kernels/` | 运行期 SPICE 内核；大型 `.bsp` 由 Git LFS 或 `download_kernels.py` 管理 |
| `datasets/catalog_baseline/` | 回归夹具和 Release 资产源，不随 wheel 分发，须显式导入 |

## Development Commands

源码开发使用 Makefile，不要手工拼接 editable 构建：

```bash
make dev            # setup + uv sync --group dev --no-install-project + maturin develop
make dev-release    # 同上，Rust 扩展以 --release 构建
make setup          # 下载 CSPICE 编译包和 SPICE 内核
make cspice         # 只准备 CSPICE_DIR
make kernels        # 只准备 kernels/
make test           # Rust workspace + Python 全量测试
make test-python    # pytest，默认 -n auto --dist loadscope
make test-rust      # cargo test --workspace，串行运行
make check          # fmt、clippy、ruff、层级/旧路径检查、mypy
make fmt            # Rust/Python 就地格式化
make docs           # Sphinx 零告警构建到 docs/_build/html
make catalog-baseline
make clean-tests
```

重要约束：

- 禁止裸跑 `uv sync`；它会触发 editable Rust 扩展构建并与 `make dev` 重复。`uv run` 一律带 `--no-sync`，例如 `uv run --no-sync python -m pytest ...`。
- 定向验证示例：`uv run --no-sync python -m pytest tests/api/test_cli.py::TestWorker::test_cancel`、`uv run --no-sync python -m pytest tests/algorithm -m "not spice"`、`cargo test -p e2m2e-forces srp_jacobian -- --test-threads=1`。
- 基线生成和回填脚本的 docstring 要求直接使用 `.venv/bin/python`，不要用会触发项目重建的 `uv run`：`.venv/bin/python scripts/generate_catalog_baseline.py`。
- 文档工具未随 dev group 安装时，先在虚拟环境安装 `sphinx myst-parser sphinx-autoapi shibuya`，再运行 `make docs`。文档构建不需要安装项目本体、Rust 或 SPICE。
- `make check` 是本地静态总门禁；PR CI 分为 `lint` 与 `typecheck`，不代替本地 Python 全量测试。修改 Makefile 或 CI 命令时同步检查两者。

## Code Conventions & Common Patterns

- Python 用 ruff：100 列、`target-version = "py310"`，规则集为 `E F W I UP B SIM`；Rust 用 `cargo fmt` 和 `cargo clippy --workspace -- -D warnings`；类型检查为 `mypy e2m2e/ --ignore-missing-imports`。
- Python 模块、函数和变量用 `snake_case`，类用 PascalCase；公共模块维护显式 `__all__`。请求/响应成对命名为 `*Request`/`*Response`；错误码用大写下划线。Rust FFI 函数通常以 `*_py` 命名，SPICE 包装以 `spice_*` 命名，但以实际导出表为准，不要强行重命名所有非 `*_py` 符号。
- 新的公开输入输出模型放 `e2m2e/api/models/` 对应主题子模块，并在 `models/__init__.py` re-export（公开导入路径 `e2m2e.api.models` 不变），继承公共 API 模型并保持 `extra="forbid"`；算法层使用 numpy、dataclass 和领域类型，不把 Pydantic 边界下沉到算法/data。字段描述是 CLI `--help` 和 MCP schema 的权威来源。
- 确定性错误使用 `E2M2EError` 层次；不可行搜索或部分成功使用 `(ConvergenceState, FailureCause, message)` 状态三元组，不用异常伪装软失败。API 信封将参数校验映射为 `INVALID_PARAMS`，保留领域 `OrbitError`，`E2M2EError` 层次映射为 `E2M2E_ERROR`，保留 message 并在 details 给出异常类型名（确定性领域错误的 cause 不丢），其余异常映射为 `INTERNAL_ERROR` 且不泄露 traceback。
- 核心同步；异步只放传输层。MCP 短任务用 `anyio.to_thread`，长任务用 worker 进程；进度回调形状为 `cb(fraction, message)`，进度回调失败不得中断计算。Rust 长计算用 `py.allow_threads`，可用 Rayon 并行。
- 通过 `Config` 注入运行环境，不新增全局单例。`Config.to_payload/from_payload` 是 worker 跨进程契约，未知字段必须拒绝。catalog 的 `records/*.json + *.npz` 是事实来源，`catalog.db` 是可重建索引；目录配置和自动入库都必须由调用方显式开启。
- 物理常量只改 `e2m2e/data/constants/constants.toml`，让 Python loader 和 Rust build script 同步生成；动力学基准使用研究级容差，筛选和测试使用筛选级容差，不把长弧/密网格塞入默认 pytest。
- 新增工具的顺序是：算法实现 → `Facade`/`Catalog`/`Spatiography` 方法加 `@mcp_exposed(request_model=...)` → 检查 `tool_inventory()` 派生面 → 补接口行为测试。不要在 MCP、CLI、sidecar 中复制业务逻辑。
- 注释、docstring、commit、Issue、PR 和 Agent brief 用中文；面向调用方的行为变化更新 `CHANGELOG.md`。ADR 是决策快照，后续变化追加修订或新开递增 ADR，不改写历史结论。
- Issue/PR 正文、AI 生成的评论、commit body 与 CHANGELOG 条目按 `CONTRIBUTING.md` 的“正文写作约定”写：完整叙述回答固定问题集、平实句子、少特殊符号、细节不进折叠区；开单模板已内联问题集引导。
- AI 标记的形式照仓库约定（issue/PR 标题 `[AI Generated][<类型>]` 前缀、评论首行 `> **[AI Generated]** 本评论由 AI 完成。`，见 `docs/agents/issue-tracker.md`）；标记里的执行者身份按**当前**工作环境如实写（如 `omp agent 会话`），不照抄既有 issue / PR / 评论里他次的身份，也不编造工具名。
- 文档（README、NOTICE、CHANGELOG、CONTEXT、ADR、`docs/` 与 `.out-of-scope/` 下所有页面，以及代码内中文注释与 docstring）不得使用直角引号「」『』（U+300C/U+300D/U+300E/U+300F），引用词语统一用弯引号“”（U+201C/U+201D），嵌套用弯单引号‘’（U+2018/U+2019），同一文件内保持一致；引号只在确为引用时使用，能直接叙述的就不要加引号；代码示例、行内代码与英文原文引述里的 ASCII 引号属代码语法或原样引述，保持不动。机器门禁为 `scripts/check_doc_quotes.py`，随 `make check` 运行。

## Important Files

- 组合根与入口：`e2m2e/__init__.py`、`e2m2e/api/facade.py`、`e2m2e/api/cli/main.py`、`e2m2e/api/mcp/server.py`、`e2m2e/api/sidecar/__init__.py`。
- API 契约：`e2m2e/api/models/`、`e2m2e/api/config.py`、`e2m2e/api/execution.py`、`e2m2e/api/mcp/envelope.py`、`e2m2e/api/frames.py`、`e2m2e/api/mcp/worker.py`。
- 共享内核：`e2m2e/exceptions.py`、`e2m2e/status.py`、`e2m2e/integrators.py`、`e2m2e/spice_ext.py`。
- 领域实现：`e2m2e/algorithm/design/design_orbit.py`、`algorithm/family/__init__.py`、`algorithm/dynamics/`、`algorithm/forces/`、`algorithm/transfer/`；数据和 catalog 在 `e2m2e/data/`。
- Python↔Rust 边界：`crates/e2m2e-integrators/src/lib.rs`、`crates/e2m2e-integrators/build.rs`、`crates/e2m2e-integrators/abi-version.txt`、各 crate 的 `Cargo.toml`。
- 构建与版本：`pyproject.toml`、`Cargo.toml`、`Makefile`、`rust-toolchain.toml`、`.python-version`、`uv.lock`、`Cargo.lock`（若存在，仅以配置和版本锁为准）。
- 门禁与测试基础设施：`scripts/check_layer_imports.py`、`scripts/check_deleted_dir_refs.py`、`scripts/check_doc_quotes.py`、`tests/conftest.py`、`tests/time_budget.py`、`tests/kernel_helpers.py`、`tests/_meta/`。
- 维护规则：`README.md`、`CONTRIBUTING.md`、`CONTEXT.md`、`CHANGELOG.md`、`docs/adr/README.md`；`docs/agents/` 下的配置见上文“Agent skills”节。

## Runtime/Tooling Preferences

- 运行时最低 Python `>=3.10`，开发环境由 `.python-version` 固定为 3.13；Rust 由 `rust-toolchain.toml` 固定为 1.98.0，并要求 rustfmt/clippy。包管理使用 uv，锁文件为 `uv.lock`；Python↔Rust 构建使用 maturin。
- 本地不需要 Node、Bun 或 Docker。CI 的 manylinux wheel job 使用容器是发布实现细节，不改变本地 `make dev` 流程。
- 普通构建的 CSPICE 来自 GitHub Release `cspice-v1`，SPICE 内核来自 `kernels-v1`；不要启用 `cspice-sys` 的官网下载路径。`make` 自动准备 `CSPICE_DIR`，bindgen 需要 `LIBCLANG_PATH`。支持的平台和资产规则以 `scripts/download_cspice.py` 为准。
- Rust 扩展缺失、ABI 过期或符号缺失时应明确报错并提示 `make dev`，不得静默降级。`crates/e2m2e-integrators/abi-version.txt`、生成的 `e2m2e/_rust_abi.py` 与 Rust `_py_abi_version()` 必须保持一致。
- 运行时常用配置为 `SPICE_KERNEL_DIR`、`E2M2E_CATALOG_DIR`、`E2M2E_CATALOG_ENABLED`；并行度和测试线程环境变量应遵循 Makefile、Rust 入口及 `tests/conftest.py` 的现有命名，不自行引入同义开关。
- 可选依赖按功能安装：`[mcp]` 提供 MCP/anyio，`[normal-form]` 提供 SymPy，`[docs]` 提供 Sphinx 文档工具。新增依赖前先复用标准库或现有依赖，并同步 `pyproject.toml` 与 `uv.lock`；Rust 依赖统一放 workspace。

## Testing & QA

- Python 使用 pytest、pytest-xdist、pytest-cov；Rust 使用 `cargo test`，测试既有 crate 内 `#[cfg(test)]` 也有 `crates/*/tests/` 集成测试。Rust 全量测试必须 `--test-threads=1`，因为 CSPICE 有进程级全局状态。
- 每个 pytest 用例恰好一个主功能标记：`theory`、`integrator`、`force`、`data`、`orchestration`、`interface`、`aux`；`spice`、`low_thrust` 是正交标记。不要新增 `slow`、`e2e` 等速度层标记，也不要让一个用例承担多个主类。
- 缺少外部能力才 skip：使用 `requires_spice`、`requires_native_symbols`、`pytest.importorskip` 或现有 fixture；代码错误必须 fail。SPICE fixture 用完卸载内核，catalog 测试用 `tmp_path` 隔离目录，昂贵轨道用 session/module 缓存并在函数级返回副本。
- 时间门禁机器强制 call 阶段默认每用例 10 秒；确实无法压缩时才用 `@pytest.mark.time_budget(seconds)` 并写明原因。单文件 60 秒是人工纪律，不是当前插件的机器门禁；长弧、密网格和生产图移到 `scripts/`。
- 按物理定义、解析解、守恒量、对称性和文献公式验收；不把外部软件运行时输出当 oracle。文献数值区分定义性（论文采用的常数、系数、可复算表值，可断言并记录复现陷阱）与结果性（他人仿真输出，仅 docstring 人工对照，禁入断言）；oracle 类别词汇见 ADR 0055。协议字节、闭式公式等固定值可以验证，但不要用脆弱的实现细节或复制同一实现的断言替代行为测试。
- Python 覆盖率配置为 branch coverage、`fail_under = 55`，需显式运行 `uv run --no-sync python -m pytest --cov` 才启用；`make test` 本身不带 `--cov`。修改行为时先补最小真实调用或回归测试，并运行受影响测试、`make check`；跨层契约变化再扩大到 `make test`。
- `_meta` 测试守护主标记、层级相关契约、常量来源、Rust ABI、共享叶 re-export、原生符号门和测试残留。`make check` 不会自动运行 `_meta`，完整 pytest 收集时才会覆盖这些门禁。

## 写作要求

所有面向人读的文本（注释、CONTEXT.md、ADR、issue 评论、PR 描述、agent brief、分诊记录、文档、Agent 回复）应当：

- 准确、清楚、简洁。先理解材料，再提炼结论。
- 按逻辑组织，区分相近概念。不用空泛、夸大的修饰语。
- 面向实际读者，从已知事实推到陌生结论。用分析说服，不装腔或堆砌。
- 语言统一中文，不中英混用。命令、代码、路径、专有名词保留原文。
- 概念不直译。没有通行译名的概念用功能描述，不生造译名。同一概念全文只用一个词：用“标准”不用“规范”，用“任务要求”不用“规格”，测试结果用“通过、失败”不用“红、绿”。
- 标点符号：中文语境用全角中文标点（句号“。”、逗号“，”、顿号“、”）。纯英文句子用英文标点。行内代码、文件名、命令后的标点根据所在句子语境选择。引号：中文用弯引号“”，英文用直引号""。列举用顿号（“、”）分隔，最后一项前不加“和”或“与”。
- 不用破折号（“——”、“—”）、分号、箭头（“→”）、markdown 加粗和表情符号。补充说明用括号，并列与分支写成独立句子，顺序与对应关系用文字表达。
- 单位用国标：带单位的数值用 GB 3100～3102 的法定计量单位。写单位符号时，数值与符号之间留一个空格（`200 ms`、`5 min`、`64 MB`），不写 `200MS`、`64MB` 这类变体。中文行文里用汉字单位名称（30 秒、5 分钟）同样合规。

## 编码准则

- 先理解再改动：完整阅读目标文件、相似实现和相关测试。不确定 API 或惯例时查源码或文档，不猜。
- 明确目标与决策：需求或验收条件不明确时先澄清。架构选择、假设和关键取舍要说明。
- 保持简单：只实现当前需求。复用已有模式。不为单一用例过早抽象、配置化或引入依赖。
- 精准修改：只改与任务直接相关的代码，贴合既有风格。删掉本次修改产生的废弃代码，不重格式化无关内容。
- 注释不留历史：删掉只对读过旧版本的人有意义的注释（“原来是 X”“改成 Y 是为了修 Z”）。注释解释当前代码为什么这样写，变更过程由 git 记录。
- 完整迁移：变更接口或行为时更新所有调用方、测试和文档。不保留无需求的兼容层。
- 按根因修复：先复现并读完整错误信息。一次处理一个原因，不用吞异常或特判掩盖问题。
- 验证行为：按影响范围运行相关检查。测试可观察行为、边界和错误路径，不测试实现细节。无法测试时说明原因并做可行的烟雾验证。
- 审慎依赖：优先现有依赖和标准库。新增依赖前确认必要性、维护状态和成本，并说明理由。
- 清楚沟通：说明做了什么、为什么、验证结果和已知风险。对不确定性给出具体事实，提交信息描述实际改动。
