# ADR 0017：转移网格搜索纯数值内核下沉 Rayon

**状态**：已采纳
**日期**：2026-08-07
**关联**：ADR 0002（Rust 积分器内核）、ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0013（验证策略）、ADR 0016（EphemCache 架构）
**设计文档**：`archive/plans/transfer-grid-search-rust.md`

## 背景

DRO→RO 转移的「搜索—优化」两步法（Cui et al. 2025），第一步是在 α（切向速度比）× n_departure（出发时刻）网格上逐点前向积分 + 几何可行性筛选，产出候选解集合。搜索阶段每个网格点完全独立、无跨点状态，是典型的「喂进数字就迭代」纯数值内核。

此前搜索用 Python `ProcessPoolExecutor` / `ThreadPoolExecutor` 并行。瓶颈不在积分本身——积分早已是 Rust（`propagate_cr3bp`）——而在 Python 侧两点：

1. **进程间 pickle**：`processes` 后端把配置元组 + `departure_state` + `arrival_states` 数组 pickle 进每个子进程，按出发点数线性增长。
2. **per-α 的 Python 循环开销**：`search_single_departure` 内层 α 循环里，`detect_local_minimum` 的 Python for 循环、`dict` 组装、`np.concatenate`/`np.linalg.norm` 等逐点 Python 开销，是 `threads` 后端的真正负担——它们持 GIL，线程无法真正并行。

基准证实了这一点：大网格（320 评估）下 `threads` 反而比 `processes` 慢一半（GIL 把 Python 循环开销串行化，又叠加线程切换），见文末基准数据。

## 决策

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

**并行架构**照搬 `multiple_shooting.rs:355-412` 的保序并行范式：

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

- **GIL 释放**：`py.allow_threads` 包外层（`multiple_shooting.rs:664-676` 模板）。
- **直接调纯 Rust**：worker 内部调 `e2m2e_forces::cr3bp::propagate_cr3bp`，**不绕道 `propagate_cr3bp_py`**——后者持 GIL，会让 Rayon 形同串行（本设计最易踩的坑）。
- **保序位级一致**：`par_iter` + `collect` 保序，`E2M2E_SEARCH_PARALLEL=0` 强制串行模式对照（复用 `E2M2E_MS_PARALLEL` 的串行验证范式）。
- **CR3BP 无 SPICE FFI**：纯数学，不需要 `multiple_shooting` 的 `StrictGuard` / `ephem_cache`——rayon 安全前提比多重打靶简单。

## 先例

- **`multiple_shooting.rs:411`** `(0..n_seg).into_par_iter()`：本仓已落地的「数值层 `par_iter` + `allow_threads` + 环境变量开关」范式。本 ADR 的搜索下沉是同一范式的复用，把环境变量开关从 `E2M2E_MS_PARALLEL` 对称扩为 `E2M2E_SEARCH_PARALLEL`。
- **ADR 0002 修订 2（propagate_compiled）**：传播进入 Rust 是被 cspice 内核池单例约束推动的**特例**（SPICE 相关传播必须和力模型编进同一扩展）。本 ADR 是「纯数值原子下沉」的**另一类先例**——搜索评估单元无 SPICE，下沉动机是性能（消除 Python 调度与循环开销），不是并发安全约束。

两者共同把 ADR 0002 的 Rust 内核边界从「单步积分 / 单段打靶」扩展到「网格评估单元」。

## 边界（固化）

下沉只触及搜索阶段的纯数值评估单元，下列职责明确留在 Python：

- **编排留 Python**：`TransferSearch`（参数管理、`search`/`optimize` 入口、可行性过滤）、`dispatch_grid_search`（后端分发）、`set_parallel_backend`（backend 校验与路由）。
- **NLP 优化留 Python**：SLSQP / COPT 串行迭代是 Python 强项（`architecture-design-discussion.md` 共识）；多候选解并行用 `ProcessPoolExecutor`，不在本 ADR 范围。
- **几何方法 thin-wrapper 保留**：`TransferSearch` 上的 6 个 thin-wrapper（`_forward_integrate` / `_check_collision` / `_compute_distance_series` / `_detect_intersection` / `_detect_local_minimum` / `_compute_min_distance`）是 Python 端**唯一分发缝**，保留作 monkeypatch 兼容 + numpy 对照基准。
- **CR3BP 路径无 ephem_cache**：纯数学，不涉及星历缓存。星历路径（`EphemerisDynamics` / BCR4BP 搜索）需 `ephem_cache` + `StrictGuard`（ADR 0016），另开设计。
- **low-thrust / porkchop / nsga2 / wsb**：可复用本阶段基础设施（同模式 multiprocessing→rayon），后续单独迁移。

## 架构合规

- **ADR 0011（五层）**：搜索评估单元是数值层职责内的纯数值原子（积分 + 几何），不是算法编排下沉。编排（`TransferSearch`）仍留算法层。
- **ADR 0002（Rust 边界）**：下沉的是「喂进数字就迭代」的数值评估，不是编排。引用 `multiple_shooting` + `propagate_compiled` 两个先例，Rust 内核边界扩展到「网格评估单元」。
- **ADR 0012（依赖方向）**：Python 算法层（`algorithm/transfer/`）调 Rust 数值层（`crates/`），合法方向。
- **ADR 0013（验证策略）**：双后端共存——Python 算法单元测试保留（thin-wrapper 分发缝让 monkeypatch 测试零改动），Rust 用等价性对照（`test_rust_backend_equivalence` 逐候选逐字段、`test_geometry_rust_vs_numpy` 逐几何函数），不依赖外部软件。

## 双后端共存与 monkeypatch 缝

`search(parallel_backend='rust')` 生效需两个前提：Rust 扩展已构建、且几何方法未被 monkeypatch。任一不满足自动回退 Python 路径（结果正确，仅降速）：

- **monkeypatch 缝**：测试用 `monkeypatch.setattr(TransferSearch, "_forward_integrate", ...)` 注入合成轨迹。Rust 内核不经过 Python 方法分发，patch 不生效——`_geometry_methods_monkeypatched` 检测 `__qualname__` 偏离即回退 Python，保住测试语义。
- **Rust 扩展缺失**：`grid_search_rust` 抛 `RuntimeError`（`transfer_grid_search_py` 为 None），回退 `processes`。

现有 12 个搜索测试（含 4 个 monkeypatch）零改动通过。

## 基准数据

三档并行度统一 `n_workers=4`、每档每 backend 跑 3 次取中位 wall-time（48 核机器，CR3BP 地月，DOP853 rtol=atol=1e-9）。完整配置与脚本见 `scripts/benchmark_transfer_search.py` 与 `archive/plans/transfer-grid-search-rust-benchmark.md`。

| 规模 (dep×α) | 评估数 | processes(s) | threads(s) | rust(s) | rust vs processes |
|---|---|---|---|---|---|
| 小 (2×3)   | 6   | 0.028 | 0.029 | 0.005 | 5.50× |
| 中 (8×10)  | 80  | 0.112 | 0.202 | 0.015 | 7.52× |
| 大 (16×20) | 320 | 0.438 | 0.913 | 0.042 | 10.35× |

**读数**：

- rust 的加速比随网格规模增大（5.5× → 7.5× → 10.4×）。Python 侧的 per-α 循环开销按点数线性增长，rust 的 rayon 调度近零开销、无 Python 循环——点数越多，省下的绝对时间越多。
- `threads` 在中/大网格比 `processes` 还慢（0.55× / 0.48×）：per-α 的 Python 循环持 GIL，线程无法真正并行，反叠加切换开销。这反向印证瓶颈是 Python 循环而非积分。

## 结果

### 新增

- `crates/e2m2e-integrators/` 下搜索评估单元的 Rust 实现（`transfer_geometry` 几何核 + `transfer_grid_search` 网格分发 + Rayon 并行）。
- Python 侧 `grid_search_rust` / `grid_search_rust_serial` 薄封装（`e2m2e.integrators` 重新导出）。
- `dispatch_grid_search` 第三分支 `grid_search_rust_dispatch`（POD 输入展平 + monkeypatch 回退检测）。
- `set_parallel_backend` 校验集合加 `'rust'`。
- 等价性对照测试（`test_rust_backend_equivalence`、`test_geometry_rust_vs_numpy`、`test_rust_backend_via_search`）。

### 不变

- `TransferSearch` 公开 API、`search()` / `optimize()` 签名与行为。
- `processes` / `threads` / sequential 四种 Python 执行路径（测试与降级路径保留）。
- 12 个现有搜索测试（含 4 个 monkeypatch）零改动。
- NLP 优化阶段（SLSQP / COPT，留 Python）。

### 取舍

- **新增 Rust 维护面**：`transfer_geometry` + `transfer_grid_search` 是新的需维护的 Rust 代码，几何函数须与 numpy 参照保持等价（由 `test_geometry_rust_vs_numpy` 兜底）。
- **进度粒度退化**：rust 后端用出发点粒度回调（每完成一个 departure 触发，不逐 α），与 processes/threads 的逐 α tqdm 不同——逐 α 跨 FFI 会抵消吞吐。

## 修订（2026-08-12，ADR 0020 决策 4）

- **显式选 rust 但 Rust 不可用改报错**：原"Rust 扩展缺失时 `grid_search_rust` 抛 `RuntimeError`、回退 `processes`"改为直接报错（issue #378）——默认 `_default_parallel_backend` 恒为 `rust`，不因扩展缺失悄然改变并行模型；`processes`/`threads` 仅在调用方显式选择时使用。
- **monkeypatch 缝豁免**：几何方法被 monkeypatch（测试 `setattr` 注入合成轨迹）时回退 Python 路径保留，但限定在测试路径（`_geometry_methods_monkeypatched` 检测），生产路径不触发（ADR 0020 决策 4 测试注入缝豁免）。
