# ADR 0016: EphemCache ephemeris cache architecture / EphemCache 星历缓存架构

[English](#adr-0016-ephemcache-ephemeris-cache-architecture) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-02
**Related**: ADR 0011 (five-layer architecture), ADR 0013 (verification
strategy)

### Context

e2m2e's force models (`GravityField`, `SRP`, SPK acceleration) query SPICE
ephemerides at every integration step: body positions (`spkpos`) and frame
rotations (`pxform`/`sxform`). The original implementation called cspice FFI
directly, with two problems:

1. **Performance**: multi-year multi-rev shooting (e.g. 2 years × 50 revs ×
   8 nodes × 50 iterations) crossing the FFI boundary for every SPICE lookup
   at every step of every force — call volumes unacceptable.
2. **Concurrency safety**: cspice maintains global state internally (the
   kernel pool); concurrent multithreaded calls trigger `DAFFRNOTFOUND` or
   panics, blocking parallel shooting (rayon `par_iter` segment integration)
   outright.

EphemCache pre-samples ephemerides into in-memory cubic-spline tables before
shooting; the whole integration then reads tables instead of calling cspice FFI.

### Cache coverage

EphemCache caches two kinds of SPICE queries:

| Kind | API | Cache key | Interpolation |
|------|-----|--------|----------|
| Body position | `spkpos` → `lookup_body_position` | `(target, observer, et)` | Cubic spline (`[f64; 3]`) |
| Frame rotation | `pxform`/`sxform` → `lookup_frame_matrix` / `lookup_sxform` | `(from, to, et)` | Per-element cubic spline (`[[f64; 3]; 3]` or `[[f64; 6]; 6]`) |

**Pre-sampling flow**: `EphemCache::build(bodies, frames, sxform_pairs,
et_start, et_end, dt)` samples via cspice over `[et_start, et_end]` at step
`dt` (default 3600 s), building cubic-spline tables. Queries outside the range
are misses.

### Known limitations

**Force models that don't use the cache** — these never query ephemerides:
- `PointMassGravity`: central-body two-body acceleration is analytic, no
  SPICE.
- Thrust models (`FiniteBurn` / `VariableMassFiniteBurn`): no SPICE
  dependency.

**Miss conditions**:
- Without `enable_ephem_cache`, the global cache is `None`; `lookup_*`
  returns `Ok(None)` and callers fall back to cspice (non-strict mode).
- Query epoch `et` outside the sampled `[et_start, et_end]` → miss.
- Requested target/observer or frame pair not in the sampling list → miss.
- Oversized `dt` degrading spline accuracy (current default 3600 s suffices
  for orbital mechanics).

**cspice fallback**: non-strict mode returns `Ok(None)` on miss and force
models fall back to cspice FFI per existing patterns; strict mode returns
`Err(CacheMissError)` propagated upward by callers' `?`.

### Relationship to parallel shooting

Parallel shooting (`multiple_shooting.rs`) is EphemCache's core consumer:

```
Python call chain:
  design_orbit.py
    → spice.enable_ephem_cache(bodies, frames, et0, et_end, dt=3600)
    → shooting_multiple(states, ..., parallel=True)
      → Rust: StrictGuard::new()        # strict on
      → Rust: rayon par_iter segments   # each segment reads cache independently
      → Python: spice.disable_ephem_cache()  # try/finally cleanup
```

**Strict mode** (`StrictGuard` RAII): on throughout shooting. Within scope,
any `lookup_*` miss returns `Err` (hard failure), eliminating silent cspice
fallbacks from force models — cspice is the source of kernel-pool corruption
in parallel regions. `StrictGuard` saves the prior value and restores it on
Drop; nestable.

**E2M2E_MS_PARALLEL environment variable**: `E2M2E_MS_PARALLEL=0` forces
serial segment integration to verify parallel/serial bit-level identity
(`par_iter` order-preserving + deterministic per-segment integration → same
results). Parallel by default.

**Concurrency safety mechanisms**:
- The global cache sits behind `RwLock<Option<EphemCache>>`: read locks run
  concurrently without blocking each other (segments read splines in
  parallel); write locks (enable/disable) exclude readers.
- Cubic-spline interpolation is pure numerics with no cspice FFI — thread-safe.

### Registration flow

```
Python side                                  Rust side
──────────────────────────────────────────────────────────────
spice.enable_ephem_cache(                     enable_ephem_cache()
  bodies=["EARTH","MOON","SUN"],                → EphemCache::build()
  et_start, et_end,                               sample spkpos/pxform/sxform at dt
  dt=3600,                                      → build cubic-spline tables
  observer="EARTH",                             → ephem_cache::enable(cache)
  frame_pairs=[                                   write global RwLock
    ("ITRF93","J2000"),
    ("MOON_PA","J2000"),
  ],
)

# During shooting:
#   force models call lookup_body_position / lookup_frame_matrix
#   → read lock + spline lookup → hit: interpolation / miss: Err(strict)

spice.disable_ephem_cache()                   disable_ephem_cache()
                                                → ephem_cache::disable()
                                                → write lock clears global cache
```

`enable_ephem_cache` is a PyO3-exported function
(`e2m2e-integrators/src/lib.rs`) forwarding parameters to `EphemCache::build`.
Python-side design callers pair `enable/disable` within `try/finally` to avoid
leaking caches into later calls.

### Relationship to ADR 0013 (verification strategy)

ADR 0013 requires completing tasks by definition: test assertions come from
analytic solutions and physical invariants, not external software output.
EphemCache's verification aligns:

1. **Cache-consistency tests**: the same shooting problem run with and without
   cache (`E2M2E_MS_PARALLEL=0` + serial); assert bit-identical final states.
   Verifies the cache changes no results, referencing nothing external.
2. **Spline accuracy tests**: against known analytic trajectories (two-body
   circular orbit), assert interpolation error within tolerance. By definition:
   judged by mathematics.
3. **Strict-mode behavior tests**: miss returns `Err` under strict,
   `Ok(None)` otherwise. Interface-contract testing.
4. **Parallel/serial consistency**: `E2M2E_MS_PARALLEL=0` verifies bit-level
   identity during development — regression means, no external software.

### Revision (2026-08-12, ADR 0020 decision 4)

**Cache-miss semantics: hard failure once enabled.** After explicit
`enable_ephem_cache`, misses (query outside sampled range / pair not in list)
always return `Err`; no more strict/non-strict distinction. Enablement signals
the user asked for caching, so a miss afterward is an error — never a silent
cspice FFI fallback (kernel-pool corruption risk in parallel regions).

**Not-enabled is not a miss**: when the global cache is `None`, `lookup_*`
returns `Ok(None)` and callers fall back to cspice (user didn't ask for
caching; legitimate path). `StrictGuard` (RAII, active in shooting parallel
regions) remains as extra insurance there: hard failures even if caching wasn't
enabled, guaranteeing zero cspice in parallel regions. The old non-strict
miss→`Ok(None)` behavior now applies only to the not-enabled case.

## 中文

**状态**：已采纳
**日期**：2026-08-02
**关联**：ADR 0011（五层架构）、ADR 0013（验证策略）

### 背景

e2m2e 的力模型（GravityField、SRP、SPK 加速度）在积分每一步都需要查 SPICE 星历：天体位置（`spkpos`）和参考帧旋转（`pxform`/`sxform`）。原始实现直接调 cspice FFI，存在两个问题：

1. **性能**：多年多圈打靶（如 2 年 50 圈 × 每圈 8 节点 × 50 迭代）中，每步每力都要跨 FFI 边界查 SPICE 内核，调用量级不可接受。
2. **并发安全**：cspice 内部维护全局状态（内核池），多线程并发调用会触发 `DAFFRNOTFOUND` 或 panic，并行打靶（rayon `par_iter` 段积分）无法直接使用。

EphemCache 在打靶前预采样星历到内存三次样条表，积分全程查表替代 cspice FFI。

### 缓存覆盖范围

EphemCache 缓存两类 SPICE 查询：

| 类型 | API | 缓存键 | 插值方法 |
|------|-----|--------|----------|
| 天体位置 | `spkpos` → `lookup_body_position` | `(target, observer, et)` | 三次样条（`[f64; 3]`） |
| 参考帧旋转 | `pxform`/`sxform` → `lookup_frame_matrix` / `lookup_sxform` | `(from, to, et)` | 逐元素三次样条（`[[f64; 3]; 3]` 或 `[[f64; 6]; 6]`） |

**预采样流程**：`EphemCache::build(bodies, frames, sxform_pairs, et_start, et_end, dt)` 按 `dt` 步长（默认 3600s）在 `[et_start, et_end]` 区间调 cspice 采样，构建三次样条表。区间外的查询视为 miss。

### 已知限制

**不缓存的力模型**：以下力模型不查星历，因此不涉及缓存：
- `PointMassGravity`：中心天体二体加速度用解析公式，不走 SPICE
- 推力类模型（`FiniteBurn` / `VariableMassFiniteBurn`）：无 SPICE 依赖

**不缓存的条件**：
- 未调用 `enable_ephem_cache` 时，全局缓存为 `None`，`lookup_*` 返回 `Ok(None)`，调用方回退 cspice（非 strict 模式）
- 查询时刻 `et` 超出预采样区间 `[et_start, et_end]` → miss
- 请求的 target/observer 或 frame 对不在预采样列表中 → miss
- `dt` 过大导致三次样条插值精度不足（当前默认 3600s 对轨道力学场景足够）

**cspice 回退**：非 strict 模式下 miss 返回 `Ok(None)`，力模型按既有模式回退 cspice FFI。strict 模式下 miss 返回 `Err(CacheMissError)`，由调用方 `?` 向上传播。

### 与 Parallel Shooting 的关系

并行打靶（`multiple_shooting.rs`）是 EphemCache 的核心消费者：

```
Python 调用链：
  design_orbit.py
    → spice.enable_ephem_cache(bodies, frames, et0, et_end, dt=3600)
    → shooting_multiple(states, ..., parallel=True)
      → Rust: StrictGuard::new()        # 开启 strict
      → Rust: rayon par_iter 段积分     # 每段独立查缓存
      → Python: spice.disable_ephem_cache()  # try/finally 清理
```

**Strict 模式**（`StrictGuard` RAII）：打靶全程开启。作用域内任何 `lookup_*` miss 返回 `Err`（硬失败），杜绝力模型静默回退 cspice：cspice 在并行区是内核池损坏的根源。`StrictGuard` 保存前值，Drop 时恢复，支持嵌套。

**E2M2E_MS_PARALLEL 环境变量**：`E2M2E_MS_PARALLEL=0` 强制串行段积分，用于验证并行/串行位级一致性（`par_iter` 保序 + 各段积分确定 → 并行与串行结果相同）。默认并行。

**并发安全机制**：
- 全局缓存用 `RwLock<Option<EphemCache>>` 保护：读锁并行不互相阻塞（段积分并发查三次样条），写锁（enable/disable）与读锁互斥
- 三次样条插值是纯数值计算，无 cspice FFI，线程安全

### 注册流程

```
Python 侧                                    Rust 侧
──────────────────────────────────────────────────────────────
spice.enable_ephem_cache(                     enable_ephem_cache()
  bodies=["EARTH","MOON","SUN"],                → EphemCache::build()
  et_start, et_end,                               按 dt 步长预采样
  dt=3600,                                        spkpos / pxform / sxform
  observer="EARTH",                             → 构建三次样条表
  frame_pairs=[                                 → ephem_cache::enable(cache)
    ("ITRF93","J2000"),                            写入全局 RwLock
    ("MOON_PA","J2000"),
  ],
)

# 打靶期间：
#   力模型调 lookup_body_position / lookup_frame_matrix
#   → 读锁查三次样条 → 命中返回插值 / miss 返回 Err(strict)

spice.disable_ephem_cache()                   disable_ephem_cache()
                                                → ephem_cache::disable()
                                                → 写锁清空全局缓存
```

`enable_ephem_cache` 是 PyO3 导出函数（`e2m2e-integrators/src/lib.rs`），参数透传给 `EphemCache::build`。Python 侧的设计调用者负责在 `try/finally` 中配对 `enable/disable`，避免缓存泄漏到后续调用。

### 与 ADR 0013（验证策略）的关系

ADR 0013 要求按定义完成任务：测试断言来自解析解和物理不变量，不依赖外部软件输出。EphemCache 的验证策略与此一致：

1. **缓存一致性测试**：同一打靶问题，分别在有缓存和无缓存（`E2M2E_MS_PARALLEL=0` + 串行）下运行，断言最终状态位级一致。这验证的是缓存不改变结果，不依赖外部参照。
2. **三次样条精度测试**：对已知解析轨迹（如二体圆轨道），断言样条插值与解析值的误差在容差内。这是按定义验证：插值精度由数学定义裁决。
3. **strict 模式行为测试**：断言 strict 下 miss 返回 `Err`、非 strict 下返回 `Ok(None)`。这是接口契约测试。
4. **并行/串行一致性**：`E2M2E_MS_PARALLEL=0` 用于开发期验证并行与串行位级一致，是回归测试手段，不依赖外部软件。

### 修订（2026-08-12，ADR 0020 决策 4）

**缓存 miss 语义：enable 后即硬失败**。`enable_ephem_cache` 显式开启后，miss（查询时刻超出预采样区间 / target/observer 或 frame 对不在预采样列表）一律返回 `Err`，不再区分 strict/非 strict。enable 是用户要求缓存的信号，enable 后 miss 就是错误，不静默回退 cspice FFI（并行区内核池损坏风险）。

**未启用缓存不是 miss**：全局缓存为 `None` 时 `lookup_*` 返回 `Ok(None)`，调用方回退 cspice（用户未要求缓存，合法路径）。`StrictGuard`（RAII，打靶并行区开启）保留为并行区额外保险：作用域内即使未启用缓存也硬失败，保证并行区零 cspice。原先非 strict 模式 miss 返回 `Ok(None)` 的行为仅剩未启用缓存一档。
