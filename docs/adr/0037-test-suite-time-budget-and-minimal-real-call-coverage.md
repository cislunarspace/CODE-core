# ADR 0037: Test suite time budget, minimal real-call coverage, and e2e test boundaries / 测试套件时间预算、真调用最小覆盖与端到端测试边界

[English](#adr-0037-test-suite-time-budget-minimal-real-call-coverage-and-e2e-test-boundaries) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-23
**Related Issues**: #534, #536
**Related**: ADR 0013 (verification strategy), ADR 0020 (failure policy),
ADR 0021 (functional categories & time bounds), ADR 0025 (suite convergence)

### Context

While investigating full-suite runtime and failures (#534), audit found multiple
time-budget violations (single tests >30 s, files >4 min killed by timeout):

1. **Nine deleted orbit-family e2e test files** (2038 lines, ~35 min): directly
   calling `design_*` to generate multi-family real orbits on top of correctors
   at 1e-12 research tolerance + scipy STM propagation (#536's root cause);
2. **`tests/api/test_facade.py` smoke cases**: genuinely generating long-arc
   families one by one — horseshoe (76 s), nrho (46 s), lpo (38 s);
3. **`tests/algorithm/transfer/test_lga.py`**: pure-Python grid search
   (360 departure angles × 5 TOFs) exceeding 4 minutes;
4. **`tests/algorithm/design/` ephemeris-correction tests** (5+ files): direct
   SPICE + differential corrector calls, minutes per file.

The core contradiction producing these slow tests: **ADR 0013 opposes mocks and
ADR 0021 decision 4 requires orchestrators' "one real call", but no executable
standard existed for that call's scale ceiling or a per-test time budget** — so
test authors wrote production-scale parameter sweeps and long-arc family
generation straight into pytest.

### Decision

1. **Test suite time budget**:
   - **Per-test wall-clock ceiling: 10 seconds**;
   - **Per-file wall-clock ceiling: 60 seconds**;
   - Tests exceeding budget stay out of default pytest; shrink problem scale
     (small amplitudes/short arcs/coarse grids/screening tolerances) into budget;
     irreducible ones move to `scripts/` manual diagnostics or benchmarks.

2. **Minimal real-call coverage standard** (interpreting & refining ADR 0021
   rationale 4):
   - `orchestration`/`interface` orchestrator entries (`design_orbit`,
     `transfer_orbit`, `Facade`) **must keep exactly one minimal-scale real-call
     smoke test** proving chain connectivity and return-type contracts;
   - Minimal calls must pick the **cheapest parameter combinations** (small-
     amplitude Halo, planar Lyapunov, loose-perilune NRHO); horseshoe (months-long
     periods), long-arc LPO (T≈21) etc. are strictly forbidden as smoke samples;
   - Exhaustive multi-family generation and grid-density-sensitive physical
     searches aren't smoke material and never enter default suites.

3. **Test tolerance orthogonal to production tolerance**:
   - Default suites uniformly use **screening tolerances**
     (`rtol/atol ≈ 1e-9–1e-10`), never dynamics-benchmark integration's
     `DEFAULT_TOLERANCE = 1e-12`;
   - Correctors & search entries should support overriding propagation tolerance
     via parameters (#536's landing scope).

4. **Computation-sharing invariant**:
   - When multiple assertions in one file depend on the same generated numbers,
     share via `pytest.fixture(scope="module")` or `@functools.cache`; repeatedly
     invoking expensive generation inside test bodies is forbidden.

### Rationale

1. **Speed isn't a correctness category, but time bounds decide whether the
   regression gate is usable** (ADR 0021 #420 basis): half-hour tests dying to
   timeouts provide no deterministic pre-merge protection.
2. **Smoke verifies chain glue, not physical feasibility exhaustion**: horseshoe's
   physical closure and Halo's chain glue are isomorphic at the interface layer;
   smoke needs only the fastest walkable path.
3. **Eliminate ambiguity**: concrete 10s/60s numbers prevent future accumulation
   of e2e recomputation debt.

### Migration plan

1. **Phase 1 (now)**: delete `tests/algorithm/family/`'s nine heavy-compute
   files keeping registry contracts; WSB tests lower tolerance + cache
   (commits `92b798e`, `acb2037`); establish ADR 0037 & CONTEXT.md.
2. **Phase 2 (#536)**: once corrector tolerance configurability lands, restore
   one **<3-second minimal real-call coverage** per family under
   `tests/algorithm/family/`.
3. **Phase 3 (targeted optimization)**: shrink `tests/api/test_facade.py` smoke
   parameters (drop horseshoe/nrho long arcs for small-amplitude samples);
   evaluate sinking `test_lga.py` onto Rust search kernels (WSB pattern).

## 中文

**状态**：已采纳
**日期**：2026-08-23
**关联 Issue**：#534、#536
**关联**：ADR 0013（验证策略）、ADR 0020（失败处理）、ADR 0021（功能类目与时间上界）、ADR 0025（测试套件收敛）

### 背景

在排查全量测试套件耗时与失败（#534）的过程中，审计发现多处测试耗时违规（单测 >30s、单文件 >4min 超时被杀）：

1. **已删除的 9 个轨道族端到端测试文件**（2038 行，耗时 ~35 分钟）：直接调用 `design_*` 生成多族真轨道，叠加底层修正器 1e-12 研究级容差 + scipy STM 传播（#536 根因）；
2. **`tests/api/test_facade.py` 冒烟用例**：对 horseshoe（76s）、nrho（46s）、lpo（38s）等长弧族逐一真实生成；
3. **`tests/algorithm/transfer/test_lga.py`**：纯 Python 网格搜索（360 出发角 × 5 TOF）耗时超 4 分钟；
4. **`tests/algorithm/design/` 星历修正测试**（5+ 文件）：直接调用 SPICE + 微分修正器，每文件分钟级。

产生这些慢测试的核心矛盾在于：**ADR 0013 反对 mock、ADR 0021 决策 4 要求编排器“一次真调用”，但没有对“一次真调用”的规模上限和测试用例的时间预算给出可执行的标准**，导致测试作者将生产级的全量参数扫描、长弧族生成直接写入 pytest。

### 决策

1. **测试套件时间预算**：
   - **单测试用例墙钟时间上限为 10 秒**；
   - **单测试文件墙钟时间上限为 60 秒**；
   - 超过此预算的用例不得进入默认 pytest，须通过缩小问题规模（小振幅/短弧/粗网格/筛选级容差）降至预算内；无法降入预算的移至 `scripts/` 手工诊断或 benchmark。

2. **真调用最小覆盖标准**（解释并细化 ADR 0021 理由 4）：
   - `orchestration` 与 `interface` 层编排器入口（如 `design_orbit`、`transfer_orbit`、`Facade`）**必须且仅保留一条最小规模的真调用冒烟测试**，证明调用链路畅通与返回类型契约；
   - 最小真调用必须选用**计算成本最低的参数组合**（如小振幅 Halo、平面 Lyapunov、近月点高容差 NRHO），严禁用 horseshoe（数月周期）、长弧 LPO（T≈21）等长耗时族充当冒烟样本；
   - 遍历性多族生成、网格密度敏感的物理搜索不属于冒烟，不得放入默认测试集。

3. **测试容差与生产容差正交**：
   - 默认测试套件统一使用**筛选级容差**（`rtol/atol ≈ 1e-9 ~ 1e-10`），不使用动力学基准积分的 `DEFAULT_TOLERANCE = 1e-12` 研究级容差；
   - 微分修正器与搜索入口应支持通过参数覆盖传播容差（#536 落地范围）。

4. **重复计算守恒**：
   - 同一测试文件内多次断言依赖同一数值生成产物时，必须通过 `pytest.fixture(scope="module")` 或 `@functools.cache` 共享结果；严禁在测试函数体内重复调用昂贵生成。

### 理由

1. **速度不是正确性分类，但时间上界决定回归门禁是否可用**（ADR 0021 #420 依据）：测试如果动辄半小时被系统杀掉，则无法在合并前提供确定性防护。
2. **冒烟的目的在于验证链路粘合，而非穷举物理可行域**：horseshoe 的物理闭合与 Halo 的链路粘合在接口层是同构的；冒烟只需选最快的一条走通即可。
3. **消除模糊空间**：给出 10s/60s 具体数字，避免未来再次积累端到端重计算债。

### 迁移计划

1. **阶段 1（当前）**：删除 `tests/algorithm/family/` 9 个重计算文件，保留注册表契约；WSB 测试降容差 + 缓存（commit `92b798e`、`acb2037`）；建 ADR 0037 与 CONTEXT.md。
2. **阶段 2（#536 推进）**：修正器容差可配化落地后，在 `tests/algorithm/family/` 恢复每族一条 **<3 秒的最小真调用覆盖**。
3. **阶段 3（专项优化）**：`tests/api/test_facade.py` 冒烟参数缩小（去掉 horseshoe/nrho 长弧参数，换成小振幅样本）；`test_lga.py` 评估 Rust 搜索核下沉（参照 WSB 模式）。
