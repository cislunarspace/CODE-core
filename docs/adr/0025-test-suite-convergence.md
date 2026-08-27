# ADR 0025: Test suite convergence — external references removed, primary marker invariant, explicit backend selection / 测试套件收敛：外部参照清除、主标记守恒与后端显式选择

[English](#adr-0025-test-suite-convergence--external-references-removed-primary-marker-invariant-explicit-backend-selection) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-14
**Related Issue**: #425
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification strategy), ADR 0020 (failure policy),
ADR 0021 (test functional categories)

### Context

ADR 0021 switched the test classification axis from speed to "what is
verified," with two rules: directories mirror source structure; exactly one
primary class per test. Post-landing audit (`tests/algorithm`, 1249 cases)
showed the rules were words without gatekeepers; five structural debts
accumulated:

1. **Unclean verification sources**. qiao/DFH outputs entered in three forms:
   five skip-guarded empty regression placeholders (verifying nothing — task
   lists disguised as tests); one fixture cross-check depending on a personal
   workstation's absolute path (`/home/.../qiao/L1_EM_Hamilton.mat`);
   hardcoded external-software output values (e.g., Hamilton constant term
   `-862.50648692`). All violate ADR 0013 decisions 2/3: correctness judged by
   physical definition, never by other software's output.
2. **Primary markers not conserved**. Three cases in
   `test_differential_correction_stagnation.py` are collected into the default
   set without primary markers — no `-m <functional-class>` selects them.
   File-level `orchestration` markers cover cases actually verifying
   `interface`/`data` (`test_wsb_contract.py`; the `DesignOrbitRequest`
   validation sections in family files). "Exactly one primary class per test"
   lacked an executable constraint; missed markers were inevitable.
3. **Directories not mirroring**. `tests/algorithm/correction/` corresponds to
   source `e2m2e/algorithm/solver/`; `e2m2e/algorithm/nominal_orbit/` has no
   test-side counterpart; `tests/algorithm` mixes pure interface validation
   constructing only `e2m2e.api.models.DesignOrbitRequest`. Test-side layering
   diverges from ADR 0012.
4. **Environment drift**. SPICE availability checks have at least three
   implementations across test files; absolute-path fixtures tie results to
   specific machines.
5. **Production coupling**. `normal_form` frequency analysis's `prefer="auto"`
   automatic backend switching got cemented by test assertions (NAFF missing →
   FFT fallback; selected NAFF failing → FFT fallback). ADR 0020 decision 4
   forbids auto — tests were guarding a violating design.

### Decision

1. **Three-tier handling of external references**. Method-provenance comments
   (pointing at qiao Code10) stay — literature citation. qiao/DFH fixture
   cross-checks and hardcoded output values leave pytest, moving to `scripts/`
   for manual diagnostics (ADR 0013 decision 4's designated place). The five
   skip-placeholder regression tests are deleted; unfinished numeric regressions
   move to issue tracking. Deleted numeric assertions get definition-level
   replacements in the same commit: ephemeris-computed reference values,
   Hamilton equation structure, symplecticity (`BᵀJB=J`), round-trip identity.
2. **Primary-marker conservation gatekeeper**. Meta-tests in `tests/_meta/`:
   collect all cases, assert each has exactly one primary-class marker,
   listing violators' paths. ADR 0021 decision 1 turns from slogan into
   executable constraint.
3. **Directory-mirror convergence**. Rename `tests/algorithm/correction/` to
   `solver/`; add `nominal_orbit/`; pure `DesignOrbitRequest` validation moves
   to `tests/api/`; `tests/algorithm` no longer imports `e2m2e.api`.
4. **SPICE detection single-pointed**. Availability checks converge onto a
   single fixture in `tests/conftest.py` + unified `spice` marker skips;
   per-file implementations all deleted.
5. **Explicit backend selection (with production change)**. `normal_form`'s
   `prefer` drops `auto`, accepting only `naff`/`fft`; selected-NAFF failure
   raises. This implements ADR 0020 decision 4 rather than revising it.

**Six migration steps** (order-dependent, each independently verifiable):
① gatekeeper lands (currently red on the stagnation file — baseline evidence)
→ ② pure moves (syncing `linked_tests`, traceability matrix, `DELETED_DIRS`)
→ ③ marker corrections (gatekeeper green) → ④ external-reference removal
(three tiers, separate commits) → ⑤ explicit backend selection (independent
PR) → ⑥ SPICE detection convergence.

### Rationale

1. **Gatekeeper first**. ADR 0021's rules had no gatekeeper; the stagnation
   missed markers and file-level mislabeled categories are evidence. Make rules
   executable before migrating; the first red run doubles as a quantitative
   baseline of current state.
2. **External references sorted by form, not一刀切**. Method provenance is
   citation, not cross-checking — deleting loses traceability. Fixture checks
   and hardcoded outputs treating other software as standard must leave pytest.
   Skip placeholders verify nothing while inflating collection counts; their
   information (which numeric regressions await) belongs to issues, not the
   suite.
3. **Definition-level replacements leave no coverage vacuum**. Deleted numeric
   assertions (e.g., Hamilton constant) swap same-commit for references computed
   on-site from ephemerides and definitional formulas — correctness still
   judged by definition, oracle no longer tied to personal workstations.
4. **Kill `auto` in production rather than amend the ADR**. The alternative —
   adding an algorithm-equivalence exemption clause to ADR 0020 — was rejected:
   exemptions get abused, and research scenarios shouldn't tolerate silent
   backend swaps anyway. `auto` violates determinism (ADR 0020 rationale 3).
5. **Directory mirroring exists for navigational predictability**
   (ADR 0021 rationale 3). `correction/` vs `solver/` is name-reality mismatch;
   MBSE's traceability matrix already registers `e2m2e.algorithm.solver`. After
   alignment, "where are module X's tests" returns to a one-sentence answer.

### Consequences

#### Added

- `tests/_meta/` primary-marker conservation meta-tests.
- Directories `tests/algorithm/solver/`, `tests/algorithm/nominal_orbit/`.
- Manual qiao/DFH diagnostic scripts under `scripts/` (migrated from pytest).

#### Changed

- `tests/algorithm/correction/` deleted (merged into `solver/`); `DELETED_DIRS`,
  `linked_tests`, traceability matrix synced.
- Pure API-validation cases moved to `tests/api/`; `tests/algorithm` no longer
  imports `e2m2e.api`.
- `e2m2e/algorithm/normal_form/`: `prefer` drops `auto`; NAFF failure raises
  (production behavior change, independent PR).
- SPICE availability probing converged to `tests/conftest.py`.

#### Unchanged

- Seven primary classes + `spice`/`low_thrust` orthogonal markers; no speed
  tiering restored.
- ADR 0013 verification, ADR 0020 failure handling, ADR 0021 functional
  category decisions unchanged in text. This entry strengthens without
  revising.

#### Costs

- With hardcoded external values gone, regression protection of corresponding
  coefficients rests on construction quality of definition-level assertions;
  risk of wrongly built definitions falls to same-commit review.
- Moving ~20 API validation cases and updating the traceability matrix —
  one-time cost.
- Dropping `auto` is public behavior change: callers relying on default
  NAFF→FFT silent fallback must pass `fft` explicitly.

### Revision (2026-08-14, #425 implementation)

Decision 4's landing refined in two spots:

1. The universal probe lives in `tests/kernel_helpers.py`
   (`spice_kernels_available()` + `requires_spice` marker), not
   `tests/conftest.py`. Why: `SPICE_KERNEL_DIR` and kernel-loading helpers
   already live there — probing shares origin; and the five consolidated probes
   were module-level skipifs (for pytestmark lists), which fixtures can't serve
   during collection. Rule added: any case carrying `requires_spice` must also
   carry the orthogonal `spice` marker, or `-m spice` selection undercounts.
2. `test_dynamical_substitution.py`'s full-window acceptance kept runtime
   probing (calling `eval_params` to check kernel-pool load) — different
   semantics from file-existence probing; that arrangement was superseded by
   the #426 revision below.

### Revision (2026-08-16, #426 scope decision)

This revision replaces decision 1's requirement to migrate qiao/DFH cross-checks
into `scripts/`, without changing definition-level acceptance relying solely on
e2m2e and its supported SPICE kernels.

The qiao normal-form pipeline and its `.mat`/`.npz` intermediates are standalone
research tools outside e2m2e's operation contract, release contract, or
development maintenance scope. The repo maintains no qiao cross-check scripts
and treats no qiao intermediate results as oracles for pytest or other project
checks. SPICE kernels remain the project's supported standard runtime dependency.

`test_dynamical_substitution.py`'s full-window frequency-suppression acceptance
calls only e2m2e itself plus SPICE, reading no qiao data, so its runtime
kernel-pool probing stays. It is definition-level behavioral checking of e2m2e's
own ephemeris dynamics — not external-software cross-checking.

## 中文

**状态**：已采纳
**日期**：2026-08-14
**关联 Issue**：#425
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（验证策略）、ADR 0020（失败处理策略）、ADR 0021（测试套件功能类目）

### 背景

ADR 0021 把测试分类轴从速度换成验证什么，并定下目录镜像源结构、每测试恰好 1 主类两条规则。落地后的审计（`tests/algorithm`，1249 用例）发现规则只写了文字、没有守门员，五类结构债随之积存：

1. **验证来源不干净**。qiao/DFH 输出以三种形态混入：5 个 skip 守卫的空回归占位测试（不验证任何东西，是任务清单伪装成测试）；1 个依赖个人工作站绝对路径（`/home/.../qiao/L1_EM_Hamilton.mat`）的 fixture 对拍；硬编码外部软件输出值（如 Hamilton 常数项 `-862.50648692`）。三者都违反 ADR 0013 决策 2/3：正确性由物理定义裁决，不许拿别的软件输出当标准。
2. **主标记不守恒**。`test_differential_correction_stagnation.py` 3 个用例能被默认集收集，却没有主标记，任何 `-m <功能类>` 都选不中它。另有文件级 `orchestration` 标记盖住实际验证 `interface`/`data` 的用例（`test_wsb_contract.py`、family 文件中的 `DesignOrbitRequest` 校验段）。每测试恰好 1 主类缺少可执行约束，漏标必然发生。
3. **目录不镜像**。`tests/algorithm/correction/` 对应源码 `e2m2e/algorithm/solver/`；`e2m2e/algorithm/nominal_orbit/` 在测试侧无对应目录；`tests/algorithm` 内混入只构造 `e2m2e.api.models.DesignOrbitRequest` 的纯接口校验用例。测试侧的层间边界与 ADR 0012 不一致。
4. **环境漂移**。SPICE 可用性检查在各测试文件内至少三种实现；绝对路径 fixture 使测试结果依赖特定机器。
5. **生产耦合**。`normal_form` 频率分析的 `prefer="auto"` 自动后端切换被测试断言固化（NAFF 缺失回退 FFT、选定 NAFF 失败也回退 FFT）。ADR 0020 决策 4 明确不允许 auto，测试在守护一个违规设计。

### 决策

1. **外部参照三档处理**。方法出处注释（对应 qiao Code10）保留，属文献引用。qiao/DFH 的 fixture 对拍与硬编码输出值从 pytest 移除，迁 `scripts/` 手工诊断（ADR 0013 决策 4 的既定位置）。5 个 skip 占位回归测试删除，未完成的数值回归转 issue 跟踪。被删的数值断言在同 commit 用定义级断言替代：星历现场计算参照值、Hamilton 方程结构、辛性（`BᵀJB=J`）、往返一致性。
2. **主标记守恒守门员**。`tests/_meta/` 加元测试：收集全部用例，断言每个用例的主类标记恰为 1，违例列出文件路径。ADR 0021 决策 1 由此从口号变为可执行约束。
3. **目录镜像收敛**。`tests/algorithm/correction/` 更名 `solver/`；补 `nominal_orbit/`；纯 `DesignOrbitRequest` 校验用例迁 `tests/api/`；`tests/algorithm` 不再 import `e2m2e.api`。
4. **SPICE 探测单点化**。可用性检查收敛到 `tests/conftest.py` 单一 fixture + `spice` 标记统一 skip，各文件自写实现一律删除。
5. **后端显式选择（含生产改动）**。`normal_form` 的 `prefer` 废 `auto`，只收 `naff`/`fft`；选定 NAFF 失败即抛。这是 ADR 0020 决策 4 的落实，不是修订。

**迁移六步**（顺序有依赖，每步独立可验证）：①守门员落地（当前应因 stagnation 文件报红，作为基线证据）→ ②纯移动（同步 `linked_tests`、追溯矩阵、`DELETED_DIRS`）→ ③标记校正（守门员转绿）→ ④外部参照清除（三档分 commit）→ ⑤后端显式选择（独立 PR）→ ⑥SPICE 探测收敛。

### 理由

1. **守门员先行**。ADR 0021 的规则没有守门员，stagnation 漏标、文件级标记盖错类别都是证据。先让规则可执行，后续迁移才有判据；守门员第一轮的报红同时充当现状的量化基线。
2. **外部参照按形态分流而非一刀切**。方法出处是引用，不是对拍，删了反而损失可追溯性；fixture 对拍和硬编码输出值才是用别的软件当标准，必须离开 pytest。skip 占位测试不验证任何行为，还虚增收集数，它的信息（哪些数值回归待补）属于 issue，不属于测试套件。
3. **定义级替代不留覆盖真空**。被删的数值断言（如 Hamilton 常数项）同 commit 换成由星历与定义公式现场算出的参照，正确性仍由定义裁决，且 oracle 不再依赖个人工作站文件。
4. **废 `auto` 动生产而非修 ADR**。另一选项是在 ADR 0020 增补算法等价策略豁免条款。被拒绝：豁免口子容易被滥用，且研究场景本不该容忍后端悄悄替换。`auto` 违背确定性这个领域要求（ADR 0020 理由 3）。
5. **目录镜像是为了导航可预测**（ADR 0021 理由 3）。`correction/` 与 `solver/` 名实不符，MBSE 追溯矩阵已按 `e2m2e.algorithm.solver` 登记，测试侧对齐后，模块 X 的测试在哪重新能用一句话回答。

### 结果

### 新增

- `tests/_meta/` 主标记守恒元测试。
- `tests/algorithm/solver/`、`tests/algorithm/nominal_orbit/` 目录。
- `scripts/` 下的 qiao/DFH 手工诊断脚本（自 pytest 迁入）。

### 变更

- `tests/algorithm/correction/` 删除（并入 `solver/`），`DELETED_DIRS`、`linked_tests`、追溯矩阵同步。
- 纯 API 校验用例迁 `tests/api/`；`tests/algorithm` 不再 import `e2m2e.api`。
- `e2m2e/algorithm/normal_form/`：`prefer` 参数废 `auto`，NAFF 失败即抛（生产行为变更，独立 PR）。
- SPICE 可用性探测收敛至 `tests/conftest.py`。

### 不变

- 七主类 + `spice`/`low_thrust` 正交标记体系不变；不恢复速度分层。
- ADR 0013 验证策略、ADR 0020 失败处理、ADR 0021 功能类目的决策本文不变。本篇是落实与补强，无修订条款。

### 代价

- 删除硬编码外部输出值后，对应系数的回归保护依赖定义级断言的构造质量；定义断言构造错误的风险由同 commit 评审承担。
- 约 20 个 API 校验用例的移动与追溯矩阵更新为一次性成本。
- `prefer` 废 `auto` 是公开行为变更：依赖默认 NAFF→FFT 静默回退的调用方需显式传 `fft`。

### 修订（2026-08-14，#425 实施）

决策 4 的落点两处细化：

1. 通用探测实现在 `tests/kernel_helpers.py`（`spice_kernels_available()` +
   `requires_spice` 标记），不在 `tests/conftest.py`。理由：`SPICE_KERNEL_DIR`
   与内核加载助手本就在 kernel_helpers，探测与其同源；且被整合的五处原是
   模块级 skipif（供 pytestmark 列表使用），fixture 在 collection 期不生效，
   无法满足该用法。规则补一句：凡带 `requires_spice` 的用例必须同时携带
   `spice` 正交标记，否则 `-m spice` 选择集漏测。
2. `test_dynamical_substitution.py` 的完整窗口验收用例曾保留运行期探测
   （试调 `eval_params` 判内核池是否已加载），与文件存在性探测语义不同；
   该安排已由下列 #426 修订取代。

### 修订（2026-08-16，#426 范围决定）

本修订取代决策 1 中 qiao/DFH 对拍迁入 `scripts/` 的要求，但不改变
仅依赖 e2m2e 与其支持的 SPICE 内核的定义级验收。

qiao normal-form 流水线及其 `.mat` / `.npz` 中间结果是独立研究工具，不属于
e2m2e 的运行契约、发布契约或开发期维护范围。仓库不再维护 qiao 对拍脚本，
也不把 qiao 中间结果作为 pytest 或其他项目检查的 oracle。SPICE 内核仍是项目
支持的标准运行依赖。

`test_dynamical_substitution.py` 的完整窗口频率压制验收只调用 e2m2e 自身与
SPICE，不读取 qiao 数据，因此继续保留运行期内核池探测。它属于 e2m2e 星历
动力学的定义级行为检查，不属于外部软件对拍。
