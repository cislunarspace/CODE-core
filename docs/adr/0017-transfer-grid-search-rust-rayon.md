# ADR 0017: Transfer grid search — pure-numerics kernel sunk to Rayon / 转移网格搜索纯数值内核下沉 Rayon

[English](#adr-0017-transfer-grid-search--pure-numerics-kernel-sunk-to-rayon) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0002 (Rust integrator core), ADR 0011 (five-layer
architecture), ADR 0012 (dependency direction), ADR 0013 (verification
strategy), ADR 0016 (EphemCache architecture)

### Context

The DRO→RO transfer search+optimize two-step method (Cui et al. 2025): step
one forward-integrates point-by-point over an α (tangential velocity ratio) ×
n_departure (departure epoch) grid with geometric feasibility screening,
producing a candidate set. Each grid point in the search phase is fully
independent with no cross-point state — a textbook pure-numerics kernel that
iterates on fed numbers.

Search previously parallelized with Python `ProcessPoolExecutor` /
`ThreadPoolExecutor`. The bottleneck wasn't integration itself (already Rust's
`propagate_cr3bp`) but two Python-side costs:

1. **Cross-process pickling**: the `processes` backend pickles config tuples +
   `departure_state` + `arrival_states` arrays into every subprocess, growing
   linearly with departure count.
2. **Per-α Python loop overhead**: inside `search_single_departure`'s inner α
   loop, `detect_local_minimum`'s Python for-loops, `dict` assembly,
   `np.concatenate`/`np.linalg.norm` per-point overhead are the real burden of
   the `threads` backend: they hold the GIL, so threads never truly
   parallelize.

Benchmarks confirmed it: at large grids (320 evaluations) `threads` was twice
as slow as `processes` (GIL serializing Python loop overhead plus thread-
switching cost); see benchmark data at the end.

### Decision

Sink the search phase's **6-step evaluation unit** (once per α, purely
numerical, no domain knowledge) plus **grid distribution** wholesale into
Rust, replacing Python process/thread pools with Rayon `par_iter`.

6-step evaluation unit (sinking list):

| Step | Content | Sinking notes |
|---|---|---|
| 1 Departure velocity composition | `v_mag`, tangential/normal unit vectors, `alpha·v_mag·t_hat` | nalgebra; precompute/cache α-independent quantities |
| 2 Forward propagation | `propagate_cr3bp` | **direct pure-Rust call** (not via GIL-holding `propagate_cr3bp_py`) |
| 3 Collision detection | distance to earth/moon centers | vectorized |
| 4 Distance series | n_traj×n_orbit broadcast + argmin | ndarray parallel |
| 5 Intersection/local minima | `detect_intersection` + `detect_local_minimum` | local minimum vectorized (compare neighbors) |
| 6 Result assembly | scalar/array/state aggregation | `TransferPointResult` pyclass |

**Parallel architecture** copies the order-preserving pattern from
`multiple_shooting.rs:465-531`:

```rust
#[pyfunction]
fn transfer_grid_search_py(...) -> Vec<TransferPointResult> {
    py.allow_threads(move || {
        (0..n_dep * n_alpha).into_par_iter()
            .map(|idx| evaluate_point(idx, ...))  // directly call pure-Rust propagate_cr3bp
            .collect::<Vec<_>>()
    })
}
```

- **GIL release**: `py.allow_threads` wraps the outer layer
  (`multiple_shooting.rs:810-828` template).
- **Direct pure-Rust call**: workers call
  `e2m2e_forces::cr3bp::propagate_cr3bp`, **never routing through
  `propagate_cr3bp_py`**: it holds the GIL and would make Rayon effectively
  serial (the easiest trap in this design).
- **Order-preserving bit-level identity**: `par_iter` + `collect` preserve
  order; `E2M2E_SEARCH_PARALLEL=0` forces serial mode for comparison
  (reusing `E2M2E_MS_PARALLEL`'s verification pattern).
- **CR3BP has no SPICE FFI**: pure math needs neither `multiple_shooting`'s
  `StrictGuard` nor `ephem_cache`; rayon safety preconditions are simpler than
  multiple shooting's.

### Precedents

- **`multiple_shooting.rs:528`** `(0..n_seg).into_par_iter()`: this repo's
  landed numerical-layer `par_iter` + `allow_threads` + env-var toggle pattern.
  This ADR's search sinking reuses that pattern, extending the env var from
  `E2M2E_MS_PARALLEL` symmetrically to `E2M2E_SEARCH_PARALLEL`.
- **ADR 0002 revision 2 (`propagate_compiled`)**: propagation entering Rust
  was a **special case** forced by the cspice kernel-pool singleton constraint
  (SPICE-related propagation must compile with force models into one
  extension). This ADR is the **other kind of precedent**: pure-numerics
  atomic sinking; the evaluation unit has no SPICE and sinks for performance
  (eliminating Python scheduling/loop overhead), not concurrency safety.

Together they extend ADR 0002's Rust-kernel boundary from single-step
integration / single-segment shooting to grid evaluation units.

### Boundaries (fixed)

Sinking touches only the search phase's pure-numerical evaluation units. The
following explicitly stay in Python:

- **Orchestration stays in Python**: `TransferSearch` (parameter management,
  `search`/`optimize` entries, feasibility filtering), `dispatch_grid_search`
  (backend dispatch), `set_parallel_backend` (backend validation/routing).
- **NLP optimization stays in Python**: SLSQP / COPT serial iteration is
  Python's strength (early architecture consensus); multi-candidate parallelism
  uses `ProcessPoolExecutor` — outside this ADR.
- **Geometry thin-wrappers retained**: the six on `TransferSearch`
  (`_forward_integrate` / `_check_collision` / `_compute_distance_series` /
  `_detect_intersection` / `_detect_local_minimum` /
  `_compute_min_distance`) are Python-side's **only dispatch seam**, kept for
  monkeypatch compatibility + numpy reference benchmarks.
- **CR3BP/BCR4BP pure-math paths need no ephem_cache**: `transfer_grid_search`
  and WSB's BCR4BP search call Rust pure-math propagators and can safely use
  Rayon; ephemeris paths (`EphemerisDynamics`) still need `ephem_cache` +
  `StrictGuard` (ADR 0016), handled under their own concurrency boundary.
- **low-thrust / porkchop / nsga2**: may reuse this infrastructure (same
  multiprocessing→rayon pattern) in later separate migrations; WSB was already
  sunk by #447 as an independent BCR4BP Rust/Rayon numeric kernel.

### Architecture compliance

- **ADR 0011 (five layers)**: the search evaluation unit is a pure-numerics
  atom within numerical-layer responsibility (integration + geometry), not
  algorithm orchestration sinking. Orchestration (`TransferSearch`) remains at
  the algorithm layer.
- **ADR 0002 (Rust boundary)**: what sinks is feed-numbers-and-iterate
  numerical evaluation, not orchestration. Citing the `multiple_shooting` +
  `propagate_compiled` precedents extends the Rust kernel boundary to grid
  evaluation units.
- **ADR 0012 (dependency direction)**: Python algorithm layer
  (`algorithm/transfer/`) calling Rust numerical layer (`crates/`) is the
  legal direction.
- **ADR 0013 (verification strategy)**: dual backends coexist; Python
  algorithm unit tests remain (thin-wrapper dispatch seam keeps monkeypatch
  tests unchanged); Rust gains equivalence comparisons
  (`test_rust_backend_equivalence` per-candidate per-field;
  `test_geometry_rust_vs_numpy` per geometry function) — no external software.

### Dual-backend coexistence and the monkeypatch seam

`search(parallel_backend='rust')` requires both: Rust extension built, and
geometry methods not monkeypatched. Either failing falls back to the Python
path automatically (correct results, just slower):

- **Monkeypatch seam**: tests inject synthetic trajectories via
  `monkeypatch.setattr(TransferSearch, "_forward_integrate", ...)`. The Rust
  kernel bypasses Python method dispatch so patches wouldn't apply;
  `_geometry_methods_monkeypatched` detects `__qualname__` deviation and falls
  back to Python, preserving test semantics.
- **Missing Rust extension**: `grid_search_rust` raises `RuntimeError`
  (`transfer_grid_search_py` is None); fall back to `processes`.

All 12 existing search tests (including 4 monkeypatches) pass unchanged.

### Benchmark data

Three scales uniformly `n_workers=4`; each scale/backend runs 3 times taking
median wall-time (48-core machine, CR3BP Earth-Moon, DOP853 rtol=atol=1e-9).
Benchmark script: `scripts/benchmark_transfer_search.py`, reproducible.

| Scale (dep×α) | Evals | processes(s) | threads(s) | rust(s) | rust vs processes |
|---|---|---|---|---|---|
| Small (2×3)   | 6   | 0.028 | 0.029 | 0.005 | 5.50× |
| Medium (8×10)  | 80  | 0.112 | 0.202 | 0.015 | 7.52× |
| Large (16×20) | 320 | 0.438 | 0.913 | 0.042 | 10.35× |

**Readings**:

- Rust speedup grows with grid size (5.5× → 7.5× → 10.4×). Python's per-α
  loop overhead grows linearly with points while rayon scheduling is near-zero
  overhead and loop-free — more points, more absolute time saved.
- `threads` slower than even `processes` at medium/large grids (0.55× / 0.48×):
  per-α Python loops hold the GIL, so threads never truly parallelize while
  adding switching cost. Confirms the bottleneck is Python loops, not
  integration.

### Consequences

#### Added

- Search evaluation unit's Rust implementation under `crates/e2m2e-forces/`
  (`transfer_geometry` geometric kernel + `transfer_grid_search` grid
  distribution + Rayon parallelism; pyfunction wrapper in `e2m2e-integrators`).
- Python thin wrappers `grid_search_rust` / `grid_search_rust_serial`
  (re-exported via `e2m2e.integrators`).
- `dispatch_grid_search` third branch `grid_search_rust_dispatch` (POD input
  flattening + monkeypatch fallback detection).
- `'rust'` added to `set_parallel_backend`'s validated set.
- Equivalence tests (`test_rust_backend_equivalence`,
  `test_geometry_rust_vs_numpy`, `test_rust_backend_via_search`).

#### Unchanged

- `TransferSearch` public API, `search()`/`optimize()` signatures & behavior.
- `processes` / `threads` / sequential Python execution paths (kept for
  testing and fallback).
- All 12 existing search tests (incl. 4 monkeypatches) unchanged.
- NLP optimization phase (SLSQP / COPT, staying in Python).

#### Trade-offs

- **New Rust maintenance surface**: `transfer_geometry` +
  `transfer_grid_search` are new Rust code to maintain; geometry functions must
  stay equivalent to numpy references (guarded by
  `test_geometry_rust_vs_numpy`).
- **Progress granularity regression**: the rust backend callbacks at departure
  granularity (per completed departure, not per α), unlike processes/threads'
  per-α tqdm: per-α FFI crossings would negate throughput.

### Revision (2026-08-12, ADR 0020 decision 4)

- **Explicitly chosen but unavailable rust now errors**: previously missing
  Rust extension → `grid_search_rust` raised `RuntimeError` and fell back to
  `processes`; now it errors outright (issue #378). Default
  `_default_parallel_backend` stays `rust` regardless of extension presence —
  the parallel model doesn't silently change; `processes`/`threads` only used
  when callers choose them explicitly.
- **Monkeypatch-seam exemption retained**: falling back to the Python path when
  geometry methods are monkeypatched (test `setattr` injection of synthetic
  trajectories) remains, restricted to test paths (`_geometry_methods_
  monkeypatched` detection); production paths never trigger it (ADR 0020
  decision 4's test-injection seam exemption).

## 中文

**状态**：已采纳
**日期**：2026-08-07
**关联**：ADR 0002（Rust 积分器内核）、ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（验证策略）、ADR 0016（EphemCache 架构）

### 背景

DRO→RO 转移的搜索+优化两步法（Cui et al. 2025），第一步是在 α（切向速度比）× n_departure（出发时刻）网格上逐点前向积分 + 几何可行性筛选，产出候选解集合。搜索阶段每个网格点完全独立、无跨点状态，是典型的喂进数字就迭代的纯数值内核。

此前搜索用 Python `ProcessPoolExecutor` / `ThreadPoolExecutor` 并行。瓶颈不在积分本身（积分早已是 Rust 的 `propagate_cr3bp`），而在 Python 侧两点：

1. **进程间 pickle**：`processes` 后端把配置元组 + `departure_state` + `arrival_states` 数组 pickle 进每个子进程，按出发点数线性增长。
2. **per-α 的 Python 循环开销**：`search_single_departure` 内层 α 循环里，`detect_local_minimum` 的 Python for 循环、`dict` 组装、`np.concatenate`/`np.linalg.norm` 等逐点 Python 开销，是 `threads` 后端的真正负担：它们持 GIL，线程无法真正并行。

基准证实了这一点：大网格（320 评估）下 `threads` 反而比 `processes` 慢一半（GIL 把 Python 循环开销串行化，又叠加线程切换），见文末基准数据。

### 决策

把搜索阶段的 **6 步评估单元**（每个 α 一次，全程纯数值、无领域知识）+ **网格分发**整体下沉 Rust，用 Rayon `par_iter` 替代 Python 进程/线程池。

6 步评估单元（下沉清单）：

| 步 | 内容 | 下沉要点 |
|---|---|---|
| 1 出发速度合成 | `v_mag`、切向/法向单位向量、`alpha·v_mag·t_hat` | nalgebra，与 α 无关量预计算缓存 |
| 2 前向积分 | `propagate_cr3bp` | **直接调纯 Rust**（不绕道持 GIL 的 `propagate_cr3bp_py`） |
| 3 碰撞检测 | 到 earth/moon 中心距离 | 向量化 |
| 4 距离序列 | n_traj×n_orbit 广播 + argmin | ndarray 并行 |
| 5 相交/局部极小 | `detect_intersection` + `detect_local_minimum` | 局部极小向量化（比较左右邻居） |
| 6 组装结果 | 标量/数组/状态聚合 | `TransferPointResult` pyclass |

**并行架构**照搬 `multiple_shooting.rs:465-531` 的保序并行范式：

```rust
#[pyfunction]
fn transfer_grid_search_py(...) -> Vec<TransferPointResult> {
    py.allow_threads(move || {
        (0..n_dep * n_alpha).into_par_iter()
            .map(|idx| evaluate_point(idx, ...))  // 直接调纯 Rust propagate_cr3bp
            .collect::<Vec<_>>()
    })
}
```

- **GIL 释放**：`py.allow_threads` 包外层（`multiple_shooting.rs:810-828` 模板）。
- **直接调纯 Rust**：worker 内部调 `e2m2e_forces::cr3bp::propagate_cr3bp`，**不绕道 `propagate_cr3bp_py`**：后者持 GIL，会让 Rayon 形同串行（本设计最易踩的坑）。
- **保序位级一致**：`par_iter` + `collect` 保序，`E2M2E_SEARCH_PARALLEL=0` 强制串行模式对照（复用 `E2M2E_MS_PARALLEL` 的串行验证范式）。
- **CR3BP 无 SPICE FFI**：纯数学，不需要 `multiple_shooting` 的 `StrictGuard` / `ephem_cache`，rayon 安全前提比多重打靶简单。

### 先例

- **`multiple_shooting.rs:528`** `(0..n_seg).into_par_iter()`：本仓已落地的数值层 `par_iter` + `allow_threads` + 环境变量开关范式。本 ADR 的搜索下沉是同一范式的复用，把环境变量开关从 `E2M2E_MS_PARALLEL` 对称扩为 `E2M2E_SEARCH_PARALLEL`。
- **ADR 0002 修订 2（propagate_compiled）**：传播进入 Rust 是被 cspice 内核池单例约束推动的**特例**（SPICE 相关传播必须和力模型编进同一扩展）。本 ADR 是纯数值原子下沉的**另一类先例**：搜索评估单元无 SPICE，下沉动机是性能（消除 Python 调度与循环开销），不是并发安全约束。

两者共同把 ADR 0002 的 Rust 内核边界从单步积分 / 单段打靶扩展到网格评估单元。

### 边界（固化）

下沉只触及搜索阶段的纯数值评估单元，下列职责明确留在 Python：

- **编排留 Python**：`TransferSearch`（参数管理、`search`/`optimize` 入口、可行性过滤）、`dispatch_grid_search`（后端分发）、`set_parallel_backend`（backend 校验与路由）。
- **NLP 优化留 Python**：SLSQP / COPT 串行迭代是 Python 强项（早期架构讨论共识）；多候选解并行用 `ProcessPoolExecutor`，不在本 ADR 范围。
- **几何方法 thin-wrapper 保留**：`TransferSearch` 上的 6 个 thin-wrapper（`_forward_integrate` / `_check_collision` / `_compute_distance_series` / `_detect_intersection` / `_detect_local_minimum` / `_compute_min_distance`）是 Python 端**唯一分发缝**，保留作 monkeypatch 兼容 + numpy 对照基准。
- **CR3BP/BCR4BP 纯数学路径无 ephem_cache**：`transfer_grid_search` 与 WSB 的 BCR4BP 搜索直接调用 Rust 纯数学传播器，可安全使用 Rayon；星历路径（`EphemerisDynamics`）仍需 `ephem_cache` + `StrictGuard`（ADR 0016），另按其并发边界处理。
- **low-thrust / porkchop / nsga2**：可复用本阶段基础设施（同模式 multiprocessing→rayon），后续单独迁移；WSB 已由 #447 下沉为独立 BCR4BP Rust/Rayon 数值核。

### 架构合规

- **ADR 0011（五层）**：搜索评估单元是数值层职责内的纯数值原子（积分 + 几何），不是算法编排下沉。编排（`TransferSearch`）仍留算法层。
- **ADR 0002（Rust 边界）**：下沉的是喂进数字就迭代的数值评估，不是编排。引用 `multiple_shooting` + `propagate_compiled` 两个先例，Rust 内核边界扩展到网格评估单元。
- **ADR 0012（依赖方向）**：Python 算法层（`algorithm/transfer/`）调 Rust 数值层（`crates/`），合法方向。
- **ADR 0013（验证策略）**：双后端共存，Python 算法单元测试保留（thin-wrapper 分发缝让 monkeypatch 测试零改动），Rust 用等价性对照（`test_rust_backend_equivalence` 逐候选逐字段、`test_geometry_rust_vs_numpy` 逐几何函数），不依赖外部软件。

### 双后端共存与 monkeypatch 缝

`search(parallel_backend='rust')` 生效需两个前提：Rust 扩展已构建、且几何方法未被 monkeypatch。任一不满足自动回退 Python 路径（结果正确，仅降速）：

- **monkeypatch 缝**：测试用 `monkeypatch.setattr(TransferSearch, "_forward_integrate", ...)` 注入合成轨迹。Rust 内核不经过 Python 方法分发，patch 不生效；`_geometry_methods_monkeypatched` 检测 `__qualname__` 偏离即回退 Python，保住测试语义。
- **Rust 扩展缺失**：`grid_search_rust` 抛 `RuntimeError`（`transfer_grid_search_py` 为 None），回退 `processes`。

现有 12 个搜索测试（含 4 个 monkeypatch）零改动通过。

### 基准数据

三档并行度统一 `n_workers=4`、每档每 backend 跑 3 次取中位 wall-time（48 核机器，CR3BP 地月，DOP853 rtol=atol=1e-9）。基准脚本为 `scripts/benchmark_transfer_search.py`，可复现。

| 规模 (dep×α) | 评估数 | processes(s) | threads(s) | rust(s) | rust vs processes |
|---|---|---|---|---|---|
| 小 (2×3)   | 6   | 0.028 | 0.029 | 0.005 | 5.50× |
| 中 (8×10)  | 80  | 0.112 | 0.202 | 0.015 | 7.52× |
| 大 (16×20) | 320 | 0.438 | 0.913 | 0.042 | 10.35× |

**读数**：

- rust 的加速比随网格规模增大（5.5× → 7.5× → 10.4×）。Python 侧的 per-α 循环开销按点数线性增长，rust 的 rayon 调度近零开销、无 Python 循环，点数越多，省下的绝对时间越多。
- `threads` 在中/大网格比 `processes` 还慢（0.55× / 0.48×）：per-α 的 Python 循环持 GIL，线程无法真正并行，反叠加切换开销。这反向印证瓶颈是 Python 循环而非积分。

### 结果

### 新增

- `crates/e2m2e-forces/` 下搜索评估单元的 Rust 实现（`transfer_geometry` 几何核 + `transfer_grid_search` 网格分发 + Rayon 并行，pyfunction 封装在 `e2m2e-integrators`）。
- Python 侧 `grid_search_rust` / `grid_search_rust_serial` 薄封装（`e2m2e.integrators` 重新导出）。
- `dispatch_grid_search` 第三分支 `grid_search_rust_dispatch`（POD 输入展平 + monkeypatch 回退检测）。
- `set_parallel_backend` 校验集合加 `'rust'`。
- 等价性对照测试（`test_rust_backend_equivalence`、`test_geometry_rust_vs_numpy`、`test_rust_backend_via_search`）。

### 不变

- `TransferSearch` 公开 API、`search()` / `optimize()` 签名与行为。
- `processes` / `threads` / sequential 三种 Python 执行路径（测试与降级路径保留）。
- 12 个现有搜索测试（含 4 个 monkeypatch）零改动。
- NLP 优化阶段（SLSQP / COPT，留 Python）。

### 取舍

- **新增 Rust 维护面**：`transfer_geometry` + `transfer_grid_search` 是新的需维护的 Rust 代码，几何函数须与 numpy 参照保持等价（由 `test_geometry_rust_vs_numpy` 兜底）。
- **进度粒度退化**：rust 后端用出发点粒度回调（每完成一个 departure 触发，不逐 α），与 processes/threads 的逐 α tqdm 不同：逐 α 跨 FFI 会抵消吞吐。

### 修订（2026-08-12，ADR 0020 决策 4）

- **显式选 rust 但 Rust 不可用改报错**：原先 Rust 扩展缺失时 `grid_search_rust` 抛 `RuntimeError`、回退 `processes`，现改为直接报错（issue #378）。默认 `_default_parallel_backend` 恒为 `rust`，不因扩展缺失悄然改变并行模型；`processes`/`threads` 仅在调用方显式选择时使用。
- **monkeypatch 缝豁免**：几何方法被 monkeypatch（测试 `setattr` 注入合成轨迹）时回退 Python 路径保留，但限定在测试路径（`_geometry_methods_monkeypatched` 检测），生产路径不触发（ADR 0020 决策 4 测试注入缝豁免）。
