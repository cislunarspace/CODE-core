# ADR 0021: Test suite organized by functional categories / 测试套件按功能类目组织，废除速度分层

[English](#adr-0021-test-suite-organized-by-functional-categories-speed-tiering-abolished) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-09
**Related**: ADR 0013 (verification strategy), ADR 0011 (five-layer
architecture)

### Context

ADR 0013 established correctness by physical definition, appending one clause
of test tiering: Rust units → Python units → integration → physical
invariants. The tiering never landed:

- Of ~203 test files repo-wide, only 43 carry explicit tier markers; `l2`
  appears just twice — "default L1+L2" is empty talk.
- `e2e`/`l3`/`slow` are three markers for the same concept (slow/integration),
  used sporadically.
- CI runs no tests (lint+mypy+inter-layer import checks only); tiering serves
  only local intuition.
- `tests/core/` and `tests/algorithms/` (plural) mirror source packages
  deleted during five-layer migration; tests organize around dead structure.

### Decision

1. **The classification axis changes from speed/integration depth to "what is
   verified"**: closed set of 7: `theory` (math/physics theory), `integrator`,
   `force`, `data` (data layer: kernels/frames/types/IO/templates + coordinate
   conversion), `orchestration` (layer-3 algorithm orchestration),
   `interface` (layer-4 facade), `aux` (tools/helpers). Exactly 1 primary class
   per test.
2. **Directories mirror source structure** (navigation), **markers state
   functional classes** (what's verified), unfinished features use independent
   markers to gate tests (e.g. `low_thrust`); no speed tiering is rebuilt.
3. **Abolish `l1`/`l2`/`l3`/`l4`/`e2e` and `slow` speed tiers**; `addopts`
   excludes only the not-yet-complete `low_thrust` feature.
4. **CI keeps static gates** (format/style/types/inter-layer imports);
   **full test suite runs before release**.
5. **`tests/` reorganized by the five layers** (`data/`, `numerical/`,
   `algorithm/`, `api/`, `tools/`, `mbse/`, `_meta/`), removing dead structure.

### Rationale

1. Speed is not a correctness category — how slowly something runs doesn't
   change what it proves.
2. Tiering existed to skip slow tests per PR; CI runs no tests, so tiering lost
   its reason to exist.
3. Directory-mirrors-source → predictable location of module X's tests;
   functional classes via markers → no scattering of a single module's tests.
4. Orchestrator APIs (`transfer_orbit`/`design_orbit`) naturally require one
   real call for API correctness; classify by assertion into
   `orchestration`/`interface`; `e2e` as a category dissolves (ADR 0013's anti-
   mock stance leaves no middle ground).
5. Data layer (containers/IO/templates) verifies data structures and defaults,
   independent of physics and orchestration — its own `data` class.

### Consequences

- `pyproject.toml` markers become 7 classes + `spice`; `addopts` excludes only
  `low_thrust`; no speed tiering; full run before release.
- Migration in three PRs: ① pure `git mv` moves (history kept, logic unchanged,
  markers untouched); ② per-file functional markers replacing l1–l4/e2e;
  ③ structural debt cleanup (private-symbol tests, golden/gmat/dfh terminology,
  #358 classification).

### Migration checklist (guarding against dead references)

Once directories under `tests/` are deleted, strings in sources and tests
pointing at them become dead references: `linked_tests` tracking, comments,
docstrings may all hide them. CI gates via
`scripts/check_deleted_dir_refs.py` (deleted-directory list maintained at that
script's top, `DELETED_DIRS`). Before/after deleting:

- Add the directory to `DELETED_DIRS`, run
  `uv run python scripts/check_deleted_dir_refs.py`, plus
  `grep -rn "tests/<old-dir>" e2m2e/ tests/` as double-check (the script gates
  code; documents need human review).
- Clean up or remap all old paths in source strings/comments/docstrings.
  Migration completion = zero old-path references.
- Requirement↔test tracking like `linked_tests` leaks most easily (#373 missed
  `tests/core`, #372 missed `tests/algorithms`); remap item-by-item against the
  migration table.
- Rewrite historical annotations ("ported from …") so they don't reference old
  paths (e.g., "located in the former core package pre-migration"), or delete.

### Revision (2026-08-14, #420)

`FiniteBurn` constant-mass Rust propagation, `VariableMassFiniteBurn`
variable-mass propagation, low-thrust shooting, collocation, and Q-law are all
implemented and in the default test set — `low_thrust` is no longer excluded by
default. The marker remains an orthogonal functional-classification marker for
explicitly selecting low-thrust regression sets.

`Facade.low_thrust_design` stays a separate interface-layer to-do and no longer
justifies gating low-level computation or algorithm tests. Wall-clock
comparisons belong to benchmark scripts outside pytest correctness judgments;
default-suite time bounds come from shrinking real problem sizes and removing
duplicated computation, not from restoring a `slow` category.

### Revision (2026-08-14, ADR 0026)

Decision 1 parenthesized the `data` class as kernels/frames/types/IO/templates
+ coordinate conversion. That "coordinate conversion" describes the
**functional class (what is verified)**, not code layering. Conversion
**algorithms** live at `algorithm/coordinate` (ADR 0011: algorithms belong to
the algorithm layer); conversion **data** (EOP/leap seconds/ephemerides) lives
in `data/frames`. Directories mirror source, markers follow functional classes
— two independent axes; don't infer from data-class membership that conversion
code belongs at the data layer. See ADR 0026.

## 中文

**状态**：已采纳
**日期**：2026-08-09
**关联**：ADR 0013（验证策略）、ADR 0011（五层架构）

### 背景

ADR 0013 定下正确性由物理定义裁决，并附一句测试分层：Rust 单元→Python 单元→集成→物理不变量。该分层未落地：

- 全仓 ~203 个测试文件，显式标层级仅 43 处，`l2` 只标 2 次；默认 L1+L2 是空话。
- `e2e`/`l3`/`slow` 三标记描述同一概念（慢/集成），散用。
- CI 不跑任何测试（仅 lint+mypy+层间 import 检查），分层只服务本地手感。
- `tests/core/`、`tests/algorithms/`（复数）对应的源包在五层迁移中已删除，测试按死结构组织。

### 决策

1. **分类轴从速度/集成深度换成验证什么**：封闭 7 类：`theory`（数理/物理理论）、`integrator`、`force`、`data`（数据层：内核/帧/类型/IO/模板 + 坐标转换）、`orchestration`（层3 算法编排）、`interface`（层4 门面）、`aux`（工具/辅助）。每测试恰好 1 主类。
2. **目录镜像源结构**（导航用），**标记标功能类**（验证什么），未完成功能用独立 marker 控制测试门（如 `low_thrust`）；不再建立速度分层。
3. **废 `l1`/`l2`/`l3`/`l4`/`e2e` 和 `slow` 速度分层**；`addopts` 只排除尚未完成的 `low_thrust` 功能。
4. **CI 维持静态门**（格式/风格/类型/层间 import），**测试在 release 前跑全量**。
5. **`tests/` 按五层重排**（`data/`、`numerical/`、`algorithm/`、`api/`、`tools/`、`mbse/`、`_meta/`），消除死结构。

### 理由

1. 速度不是正确性类别，跑多慢不改变证明了什么。
2. 分层为 per-PR 跳过慢测试而生；CI 既不跑测试，分层失存在理由。
3. 目录镜像源 → 模块 X 的测试在哪可预测；功能类交标记 → 不散射同一模块的测试。
4. 编排器 API（`transfer_orbit`/`design_orbit`）的 API 正确天然需一次真调用；按断言归 `orchestration`/`interface`，`e2e` 作为类别解散（ADR 0013 反 mock，无中间态）。
5. 数据层（容器/IO/模板）验证的是数据结构与默认值，独立于物理与编排，单列 `data` 类。

### 结果

- `pyproject.toml` markers 换 7 类 + `spice`；`addopts` 只排除 `low_thrust`；测试不再按速度分层，release 前跑全量。
- 迁移分三 PR：①`git mv` 纯移动（保历史、不改逻辑、不换标记）；②逐文件打功能类标记、去 l1~l4/e2e；③清理结构债（私有符号测试、golden/gmat/dfh 术语、#358 归类）。

### 迁移 checklist（防引用遗留）

`tests/` 下目录一旦删除，源码与测试代码里指向它的字符串就成了死引用：`linked_tests` 追踪、注释、docstring 都可能藏着。CI 由 `scripts/check_deleted_dir_refs.py` 把关（已删目录清单在该脚本顶 `DELETED_DIRS` 维护）。删目录前/后都应：

- 把目录加入 `DELETED_DIRS`，跑 `uv run python scripts/check_deleted_dir_refs.py`，并 `grep -rn "tests/<旧目录>" e2m2e/ tests/` 复核（脚本只拦代码，文档由人审）。
- 源码字符串/注释/docstring 里的旧路径全部清理或重映射。迁移完成的标志是无任何旧路径引用。
- `linked_tests` 一类的需求↔测试追踪最易漏（#373 漏了 `tests/core`、#372 漏了 `tests/algorithms`），按迁移表逐条重映射到新路径。
- 历史标注（ported from …）改写为不引用旧路径的表述（如迁移前位于旧 core 包），或删除。

### 修订（2026-08-14，#420）

`FiniteBurn` 恒质量 Rust 传播、`VariableMassFiniteBurn` 变质量传播、低推力打靶、
配点和 Q-law 已经实现并进入默认测试集，`low_thrust` 不再是默认排除项。该 marker
保留为功能分类的正交标记，供显式选择低推力回归集使用。

`Facade.low_thrust_design` 仍是独立的接口层待办，不再作为低层计算与算法测试门的
理由。墙钟性能比较属于 benchmark 脚本，不进入 pytest 的正确性判定；默认测试的
时间上界由缩小真实问题规模和消除重复计算保证，而非恢复 `slow` 分类。

### 修订（2026-08-14，ADR 0026）

决策 1 把 `data` 类括注为内核/帧/类型/IO/模板 + 坐标转换。此处的坐标转换
描述的是**功能类（验证什么）**，不是代码层级。坐标转换**算法**的代码位于
`algorithm/coordinate`（ADR 0011 既定：转换算法归 algorithm 层），坐标**数据**
（EOP/闰秒/历表）在 `data/frames`。目录按源码镜像、标记按功能类，两个轴独立；
不应由坐标转换属 data 类推出坐标转换代码应在 data 层。详见 ADR 0026。
