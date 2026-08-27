# ADR 0036: CR3BP baseline orbit-family dataset — precomputed full-family data shipped with the package / CR3BP 基线轨道族数据集：随包分发的预计算整族数据

[English](#adr-0036-cr3bp-baseline-orbit-family-dataset--precomputed-full-family-data-shipped-with-the-package) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-23
**Related**: ADR 0031 (orbit catalog), ADR 0029 (unified Rust family
generation), ADR 0014 (interface layer)

### Context

The repo previously shipped no precomputed orbit-family data: all families
computed at runtime (differential correction + continuation, Rust kernels).
Teaching scenarios demand out-of-the-box availability: after installing,
students should browse complete data for nine CR3BP orbit families immediately —
not first learn family-generation APIs.

Measured (2026-08-23, DE421 Earth-Moon, n_orbits=100 cap): single-family
generation takes ~1 s (halo L2 full family 1.5 s, DRO 0.7 s); nine families
total <15 s. `Orbit` members never carry whole trajectories anyway (`states` is
the (1,6) initial state); serializing all nine families by initial-state +
period + scalar diagnostics costs only ~120 KB. Generation is fast, data small —
but fast ≠ offline-available and version-consistent, and small makes packaging
free.

### Decision

#### 1. Package-distributed; data (JSON + NPZ) enters repo git

Baseline data (~3.3 MB total: JSON metadata + NPZ segments) commits to
`e2m2e/data/catalog_baseline/` as package data distributed with pip installs.
No Release downloads (too small to justify a distribution step).

> Revision (2026-08-23): the initial decision was no-git + regenerate before
> release; implementation switched to committing data directly. Why: JSON+NPZ
> must pair in-repo for fresh clones to build complete packages; size is small,
> change frequency low (regeneration only on algorithm changes) — git history
> cost acceptable.

#### 2. Content: initial states + periods + scalar diagnostics; no trajectories

Each member stores only x0, period, Jacobi, amplitude and other scalar
diagnostics (per ADR 0031 record format's cr3bp_segment). Full trajectory =
initial state + propagator's deterministic derivation; a 3601-point sample
computes on demand.

#### 3. Full families = each spec's maximal default coverage

Coverage doesn't grow via configuration: continuation stops naturally at ADR
0029 spec-builtin amplitude-window boundaries / folds / termination conditions;
actual member counts enter honestly (halo 79, DRO 42 are spec-determined, not
defects). Each record's metadata documents actual coverage: amplitude range,
member count, termination reason — fully auditable rather than verbally claimed.

#### 4. Scope: nine families × Earth-Moon DE421

HALO, NRHO, AXIAL, LISSAJOUS, DRO, DPO, SPO, HORSESHOE, LPO. Lyapunov has no
standalone family interface (ADR 0029 unregistered) — explicitly excluded;
NRHO/Axial calibrated seeds bind DE421 Earth-Moon, consistent with defaults of
other families. Other μ values await ADR 0029's calibration extension — out of
scope here.

#### 5. Shape: standard catalog records; first-use import into user library

The baseline is a batch of ADR 0031 catalog records (one per family), uniformly
tagged `tags: ["baseline"]`. Integration is first-use import: when user catalogs
lack the baseline or versions mismatch, copy from package into the user library
directory and rebuild index; storage engine untouched; everything thereafter uses
existing `catalog_query`. No read-only multi-source mounting: that changes the
engine, and package directories aren't writable/annotatable.

#### 6. Freshness: no CI gate; user reports + issues

Post-algorithm-change baseline regeneration relies on manual release-checklist
steps (`make catalog-baseline` rerunning generation & committing new data); no CI
recompute-compare gate. Data doubts go through issue reports → fix → re-release.
Known cost: windows where packaged data lags algorithm output due to forgotten
regeneration — accepted.

#### 7. Validation inside the generation flow

Generation script embeds assertions: per-member period-closure error, Jacobi
drift within verification tolerances, member-count floors, coverage metadata
completeness. Failed assertions write nothing. No separate test layer for file
validation.

#### 8. Teaching curation deferred

This round delivers full baselines + `baseline` tag only; curated cases &
annotations wait for real teaching consumption scenarios — no designing for
never-consumed requirements.

### Rationale

1. **Into git, shipped with package**: JSON+NPZ ~3.3 MB invisible inside
   packages; paired in-repo ensures any fresh clone builds complete package
   data; download links add distribution+verification steps for zero benefit.
   Algorithm-change regeneration goes `make catalog-baseline` + commit (see
   decision 6).
2. **No trajectories**: size grows ~120 KB → ~180 MB (float64 full sampling) to
   save 7 ms/member on-demand compute — disproportionate.
3. **First-use import over multi-source query**: catalog's value is unified
   querying; adding read-only second sources deep-changes the engine while import
   only copies files + rebuilds index (ADR 0031 guarantees full index
   rebuildability).
4. **No CI gate**: full generation <15 s makes gating technically cheap but CI
   tasks have maintenance costs; the project trusts release checklists +
   user-report (issue) loops.

### Consequences

#### Added

- `e2m2e/data/catalog_baseline/`: baseline records (JSON + NPZ, in git,
  package-distributed).
- `scripts/`: baseline generation script (embedded validation assertions;
  writes records + coverage metadata).
- Catalog first-use import logic: detect missing/mismatched baseline → import
  from package.
- Makefile target `catalog-baseline`.

#### Changed

- No interface changes; `catalog_query` consumption unchanged.

#### Unchanged

- ADR 0031 storage layout, record format, query interfaces — all untouched; the
  baseline is merely a batch of pre-generated records.
- ADR 0029 family-generation specs unchanged; baseline data snapshots their
  outputs.

### Trade-offs

- First-use import puts non-user-created records into user libraries; marked via
  `baseline` tag + baseline version — identifiable, re-importable.
- No CI gate leaves staleness windows (decision 6), mitigated by manual process.

## 中文

**状态**：已采纳
**日期**：2026-08-23
**关联**：ADR 0031（轨道库 catalog）、ADR 0029（统一 Rust 族生成）、ADR 0014（接口层）

### 背景

仓库此前不附带任何预计算轨道族数据：所有族运行时现算（微分修正 + 延拓，Rust 内核）。教学场景要求开箱即得：学生安装包后应能直接浏览九个 CR3BP 轨道族的完整数据，而不是先学会调族生成接口。

实测（2026-08-23，DE421 地月，n_orbits=100 上限）：单族生成 1 秒量级（halo L2 全族 1.5 s、DRO 0.7 s），九族全量 <15 s；`Orbit` 成员本就不携带整条轨迹（`states` 即 (1,6) 初态），九族全量按初态+周期+标量诊断序列化仅约 120 KB。生成快、数据小，但快不等于离线可得与版本一致，小则使随包分发没有代价。

### 决策

### 1. 随包分发，数据（JSON + NPZ）纳入仓库 git

基线数据（JSON 元数据 + NPZ 段数组，共约 3.3 MB）提交进
`e2m2e/data/catalog_baseline/`，作为 package data 随包分发，pip 安装即得。
不依赖 Release 下载（体积小，不值得引入分发环节）。

> 修订（2026-08-23）：初版决策是不进仓库 git、发版前重生成；实施时改为
> 数据直接入 git。理由：JSON 与 NPZ 必须成对入库才能保证 fresh clone 构建出
> 完整的包；且数据量小、变更频率低（仅算法调整时重算），git 历史成本可接受。

### 2. 内容：初态 + 周期 + 标量诊断，不存轨迹

每个族成员只存初态 x0、周期、Jacobi、振幅等标量诊断（沿 ADR 0031 记录格式的 cr3bp_segment）。整条轨迹是初态 + 传播器的确定性派生物，单条 3601 点采样按需现算。

### 3. 完整族 = 各族规格的最大默认覆盖

覆盖不通过配置扩张：延拓走到 ADR 0029 规格内置的振幅窗口边界 / 折叠点 / 终止条件自然停止，实际成员数如实入库（halo 79、DRO 42 这类数字是规格决定的，不是缺陷）。每族记录的元数据写明实际覆盖：振幅区间、成员数、终止原因，完整可审计而不口头宣称。

### 4. 范围：九族 × 地月 DE421

HALO、NRHO、AXIAL、LISSAJOUS、DRO、DPO、SPO、HORSESHOE、LPO。Lyapunov 无独立族接口（ADR 0029 未登记），明确排除；NRHO/Axial 的标定种子限 DE421 地月，与其他族的默认系统一致。其他 μ 值待 ADR 0029 的标定扩展，不在本篇范围。

### 5. 形态：标准 catalog 记录，首用导入用户库

基线就是一批 ADR 0031 catalog 记录（一族一条），统一打 `tags: ["baseline"]`。结合方式为首用导入：包内数据在用户 catalog 缺基线或基线版本不匹配时，复制进用户库目录并重建索引；存储引擎零改动，之后一切走既有 `catalog_query`。不做只读挂载多源查询：那要改存储引擎，且包目录不可写、无法打标注。

### 6. 新鲜度：不设 CI 守门，用户检查 + issue

算法变更后基线的重算靠发版清单里的人工步骤（`make catalog-baseline` 重跑生成脚本并提交新数据包），不设 CI 重跑比对守门。数据疑有问题走 issue 报告、修复重发。已知代价：存在忘了重算导致包内数据与算法输出短暂不一致的窗口，接受。

### 7. 校验在生成流程内

生成脚本内置断言：每成员周期闭合误差、Jacobi 漂移在验证容差内、成员数下限、覆盖元数据完整。断言失败不写包。不另建测试层校验文件。

### 8. 教学精选标注推迟

本次只交付全量基线 + `baseline` tag；精选案例与标注等真实教学消费场景出现后再做，避免为没人消费过的需求做设计。

### 理由

1. **入 git 并随包**：JSON + NPZ 约 3.3 MB，进包无感知；成对入库保证任意 fresh clone 构建的包数据完整；下载链接引入分发与校验环节，收益为零。算法变更的重算走 `make catalog-baseline` 后提交新数据（见决策 6）。
2. **不存轨迹**：体积从 ~120 KB 涨到 ~180 MB（float64 全采样），换来的只是 7 ms/条的现算延迟，不成比例。
3. **首用导入而非多源查询**：catalog 的价值在统一查询接口；给存储引擎加只读第二源是深改动，而导入只需复制文件 + 重建索引（ADR 0031 已保证索引可全量重建）。
4. **不设 CI 守门**：全量生成 <15 s，守门技术上便宜，但维护 CI 任务本身有成本；项目选择信任发版清单与用户报告（issue）回路。

### 结果

### 新增

- `e2m2e/data/catalog_baseline/`：基线记录文件（JSON + NPZ，入 git，随包分发）。
- `scripts/`：基线生成脚本（内置校验断言，写出记录与覆盖元数据）。
- catalog 首用导入逻辑：检测用户库基线缺失/版本不匹配时从包内导入。
- Makefile 目标 `catalog-baseline`。

### 变更

- 无既有接口变更；`catalog_query` 消费方式不变。

### 不变

- ADR 0031 存储布局、记录格式、查询接口全部不变；基线只是一批预生成的记录。
- ADR 0029 族生成规格不变；基线数据是其输出的快照。

### 取舍

- 首用导入使用户库目录出现非用户产生的记录；以 `baseline` tag 与基线版本号标记，可识别、可重导。
- 无 CI 守门留下数据陈旧窗口（见决策 6），以人工流程弥补。
