# 转移网格搜索 Rust + Rayon 并行化设计

> 状态：设计草案 · 2026-08-06
> 范围：`e2m2e/algorithm/transfer/` 网格搜索阶段（「搜索—优化」两步法的第一步）
> 不在范围：NLP 优化阶段（留 Python SLSQP/COPT）、low-thrust（待调试）、porkchop/nsga2/wsb（后续复用）

## 0. 背景与目标

DRO→RO 转移的「搜索—优化」两步法（Cui et al. 2025，docs/transfer/overview.rst）：
- **搜索阶段**：在 α（切向速度比）× n_departure（出发时刻）网格上逐点前向积分 + 几何可行性筛选，产出候选解集合
- **优化阶段**：取候选解作 NLP 初值，SLSQP/COPT 精化

搜索阶段的每个网格点（`search_single_departure` 内层 α 循环 6 步）**完全独立、无跨点状态**，是典型的"喂进数字就迭代"纯数值内核。现用 Python `ProcessPoolExecutor`/`ThreadPoolExecutor` 并行，瓶颈在进程间 pickle + Python 循环开销，不是积分本身（积分已走 Rust `propagate_cr3bp`）。

**目标**：把搜索阶段的 6 步评估单元 + 网格分发整体下沉 Rust，用 Rayon `par_iter` 替代 Python 进程/线程池——照搬本仓 `multiple_shooting.rs:411` 已落地的范式（`py.allow_threads` 释放 GIL + 内部直接调纯 Rust 函数 + 保序位级一致）。

**非目标**：不动 NLP 优化（SLSQP 串行迭代，Python 强项，architecture-design-discussion.md:85 共识）；不重写积分器（复用 `e2m2e_forces::cr3bp::propagate_cr3bp`）。

## 1. 测试套件评估（先看现状）

### 1.1 现有覆盖（12 个测试，两个文件）
- `test_dro_ro_search.py`（6 个）：几何核直测（`_compute_min_distance` 三元组与公开 API 一致性、`_compute_distance_series` 形状/数值、二者自洽）+ `_is_feasible` 判定分支（碰撞拦截/相交直通/局部极小/距离阈值）
- `test_first_feasibility.py`（6 个）：`_compute_distance_series` 正确性 + `TestSearchFirstFeasibilityFields`（4 个，**依赖 monkeypatch** 注入合成轨迹，验证 `first_intersection_idx/time`、`first_min_distance_idx/time`、`status` 字段语义）

### 1.2 monkeypatch hook 清单（Rust 化关键约束）
`test_first_feasibility.py:145-147` patch 三个类方法：
- `_forward_integrate` → 注入合成轨迹（绕过真实积分）
- `_check_collision` → 返回无碰撞
- `_detect_local_minimum` → 返回无局部极小

**关键**：`_compute_distance_series` 与 `_detect_intersection` **未被 patch**——它们对合成轨迹跑真实 numpy，这是测试能断言精确距离值的根因。

**失效条件**：若 Rust 后端让 `search_single_departure` 不再经过 `searcher._forward_integrate/_check_collision/_detect_local_minimum` 这三个 Python 方法（整个 per-α 内核下沉），这 4 个测试虽不报错（patch 仍能 setattr），但变成"测了 patch 函数本身"的空测试。

### 1.3 覆盖黑洞（与 Rust 化无关，本就该补）
- `search()` 顶层入口：**0 测试**
- 三条执行路径（sequential / parallel_processes / parallel_threads）：**0 测试**（含 `process_departure_worker` 的多进程 pickle 重建）
- 进度回调（progress_queue / tqdm pbar / aggregate）：**0 测试**
- `compute_distance_series_chunked`（n_traj×n_orbit > 1e7 分块）：**0 测试**
- 碰撞/相交几何正例（earth/moon 命中）：**0 测试**（只 patch 成 False）
- NLP 衔接（search 候选 → optimize 初值）：**0 测试**

### 1.4 API 更正（任务描述与代码不符）
`set_parallel_backend` **只接受 `'processes'` 或 `'threads'`**，不接受 `'sequential'`。sequential 模式由 `dispatch_grid_search` 中 `n_workers==1` 触发（与 backend 字符串无关）。实际是 2 后端 × {n_workers==1 串行, n_workers>1 并行} = 4 种执行路径。

## 2. 测试策略：双后端共存 + 分层等价性对照

### 2.1 分发缝保留（核心策略）
保留 `TransferSearch` 上的 6 个 thin-wrapper 作为 Python 端**唯一分发缝**：
```
_forward_integrate / _check_collision / _compute_distance_series
_detect_intersection / _detect_local_minimum / _compute_min_distance
```
`search_single_departure` **继续通过 `searcher._x(...)` 调用，不直接调 Rust**。新增内部开关 `_use_rust_backend`（由 Rust 扩展可导入性自动决定），thin-wrapper 内 `if` 分流：True→调 Rust，False→调 numpy。

**效果**：现有 12 个测试（含 4 个 monkeypatch）**零改动**——patch 的是类方法本身，patch 后内部分支无关。

### 2.2 Python 后端保底
CI 里现有 12 测试默认跑 Python 路径不动。另补一组 `search()` 端到端小网格测试，覆盖 n_workers=1 / parallel_threads(n=2) / parallel_processes(n=2)——这是当前最大覆盖黑洞，与 Rust 化无关也该补。多进程路径重点测 `process_departure_worker` 的 pickle 重建（Rust 后端状态必须可从纯标量配置元组重建，不携带 Python 句柄）。

### 2.3 Rust vs Python 等价性对照（新增 `test_rust_backend_equivalence.py`）
`pytest.importorskip('e2m2e._integrators')` 守卫。同一个小网格分别跑 Rust 后端与 Python sequential 后端，按 `(departure_time_index, alpha)` 排序后逐候选对照：
- **整数索引字段**（`first_intersection_idx`、`min_distance_idx` 等）：**精确相等**（argmin/argmax 不一致 = 算法分叉，非数值噪声）
- **布尔字段**（`success`、`intersection_found` 等）：精确相等
- **浮点字段**（`min_distance`、`dv_departure` 等）：`np.testing.assert_allclose(rtol=1e-9, atol=1e-12)`

### 2.4 几何函数 numpy 等价性单测（新增 `test_geometry_rust_vs_numpy.py`）
对 5 个下沉几何函数逐个单测，三档用例：合成已知答案 / 随机数组 / 边界（含 chunked 触发、单点轨道、全同点、碰撞正例 earth/moon 命中）。

### 2.5 进度回调测试
Rust+Rayon 下用**出发点粒度**回调（每完成一个 departure 触发，不逐 α——逐 α 跨 FFI 抵消吞吐）。测试断言：callback 调用次数 == n_departure、completed 单调不减、终值 == 总数。

## 3. 内层循环 6 步下沉清单

`search_single_departure`（search_parallel.py:81-230）每个 α 跑 6 步，**全部纯数值、无领域知识**，可整体下沉：

| 步 | 内容 | Python 位置 | Rust 要点 |
|---|---|---|---|
| 1 出发速度合成 | `compute_departure_velocity`（v_mag、t_hat、n_dir、alpha·v_mag·t_hat）| propulsion.py:38-71 | nalgebra，与 α 无关的预计算缓存 |
| 2 前向积分 | `forward_integrate` → `dynamics.propagate(with_stm=False)` | search_parallel.py:62-78 | **直接调 `e2m2e_forces::cr3bp::propagate_cr3bp`**（不重写） |
| 3 碰撞检测 | `check_collision`（到 earth/moon 中心距离）| search_geometry.py:86-103 | 向量化 |
| 4 距离序列 | `compute_distance_series`（n_traj×n_orbit 广播 + argmin）| search_geometry.py:15-28 | ndarray 并行 + fused multiply-add |
| 5 相交/局部极小 | `detect_intersection` + `detect_local_minimum`（Python for 循环）| search_geometry.py:59-83 | 局部极小向量化（比较左右邻居） |
| 6 组装 dict | 聚合标量/数组/状态 | search_parallel.py:189-222 | TransferPointResult pyclass |

**热点**：`detect_local_minimum` 的 Python for 循环（search_geometry.py:76-79）、距离矩阵广播——下沉后向量化收益最大。

**积分失败容错**：Rust `propagate_cr3bp` 步长塌缩时 raise RuntimeError，6 步下沉后需在 Rust 内 catch → 走 `dv=1e10` 惩罚分支（与 `_evaluate_all` 设计意图一致，前面 NLP 回归同一问题）。

## 4. 并行架构：Rayon 替代三档后端

照搬 `multiple_shooting.rs:355-412` 的保序并行范式：
```
#[pyfunction]
fn transfer_grid_search_py(...) -> Vec<TransferPointResult> {
    py.allow_threads(move || {
        (0..n_dep * n_alpha).into_par_iter()
            .map(|idx| evaluate_point(idx, ...))  // 直接调纯 Rust propagate_cr3bp
            .collect::<Vec<_>>()
    })
}
```
- **GIL 释放**：`py.allow_threads` 包外层（`multiple_shooting.rs:664-676` 模板）
- **直接调纯 Rust**：内部调 `e2m2e_forces::cr3bp::propagate_cr3bp`，**不绕道 `propagate_cr3bp_py`**（后者持 GIL，会让 Rayon 形同串行——这是最易踩的坑）
- **保序位级一致**：`par_iter` + `collect` 保序，E2M2E_MS_PARALLEL=0 串行模式对照（复用为 `E2M2E_SEARCH_PARALLEL`）
- **线程数**：`n_workers` → `rayon::ThreadPoolBuilder::num_threads(n_workers)`
- **CR3BP 无 SPICE FFI**：纯数学，不需要 multiple_shooting 的 StrictGuard/ephem_cache——rayon 安全前提比多重打靶简单

**❌ 必须避免**：从 Rust worker 回调 `propagate_cr3bp_py`（GIL 序列化、Rayon 形同虚设）。

## 5. 数据 schema：POD 边界

遵循 architecture-design-discussion.md 第五节"FFI 只暴露 POD"。

### 5.1 输入（Python→Rust，展平）
- `dep_states: Vec<f64>`（n_dep×6 展开）
- `dep_times: Vec<f64>`（n_dep）
- `arrival_states: Vec<f64>`（n_arrival×6）
- `arrival_times: Vec<f64>`（n_arrival）
- `alpha_grid: Vec<f64>`（n_alpha）
- 标量包（mu、max_transfer_time、integration_dt、intersection_threshold、min_distance_threshold、collision_earth_radius、collision_moon_radius、rtol、atol、max_step）

参考 `_process_pack_base`（search_parallel.py:331-350）的 15 元组展平。**Orbit 对象不穿透**，只传 states/times 数组。

### 5.2 输出（Rust→Python，TransferPointResult pyclass）
`#[pyclass(frozen, get_all)]`，字段对齐候选解 dict（~25 字段）。Python 侧 `grid_search_rust` 转 `list[dict]` 保持返回类型契约。

### 5.3 内存策略（关键）
`transfer_trajectory` 是大头（每评估 n×6 floats，n≈3000→18KB/评估），n_dep×n_alpha 评估总数据可达 GB 级。**建议**：失败/碰撞/无相交评估不回传轨迹（`Option<Vec<f64>>`），只对 `status=='success'` 回传；或 Rust 只回传摘要（min_dist/idx/intersection/local_min），轨迹按需重算。

## 6. set_parallel_backend 路由扩展

- `set_parallel_backend`（transfer_search.py:120-123）校验集合加 `'rust'`
- `dispatch_grid_search`（search_parallel.py:249-282）加第三分支：`pb=='rust'` → 新函数 `grid_search_rust(searcher, departure_orbit, arrival_orbit, ...)`
- `grid_search_rust`：网格输入展平 POD → 调 `transfer_grid_search_py` → 拿回 `Vec<TransferPointResult>` 转 `list[dict]`
- **默认 backend**：建议有 `_integrators` 构建时默认 `rust`，否则回退 `processes`
- 保留 `processes`/`threads`/sequential 作测试与降级路径

## 7. 进度回调：crossbeam channel + Python 轮询

现状三档：sequential（tqdm 每 α update）/ threads（slot_queue + 分槽 tqdm 或 aggregate+Lock）/ processes（`Manager().Queue()` + daemon 轮询线程）。

**Rust 方案**（与 processes 模式 UX 对称）：
- Rust 侧 `crossbeam::channel::unbounded()`，每个 departure 完成 `tx.send(1)`
- Python 侧传 callable 进度回调（持 GIL 时触发）+ daemon 线程排空 queue 更新 tqdm
- **粒度**：出发点粒度（每 departure 一次，不逐 α），避免抵消 Rayon 吞吐
- **batch 回调**：每 N 个 departure 回调一次，降 GIL 开销
- **约束**：`py.allow_threads` 内不碰 Python 对象，回调在 allow_threads 块外触发

## 8. monkeypatch 缝兼容（最大约束）

`test_first_feasibility.py` 4 个测试 patch `_forward_integrate/_check_collision/_detect_local_minimum`。**方案（推荐 A）**：
- **(A)** Rust backend 生效时，检测到 searcher 几何方法被 monkeypatch（或显式 `use_rust=False`）则**回退 Python 路径**——测试默认走 Python，生产走 Rust。在 `TransferSearch` 加 `_use_rust_kernel` 标志，`search()` 依据 `parallel_backend=='rust'` 且无 monkeypatch 痕迹时设 True
- (B) Rust 路径接收可选 Python callable 注入点——有 GIL 开销，违背下沉初衷，否决

**落地**：先 `grep` 全仓 `monkeypatch.setattr(TransferSearch, ...)` 盘点所有 patch 点。

## 9. 迁移路线（每步先测试后切换）

1. **下沉 6 步为 Rust 内层函数**，Python 侧仍串行调度（验证数值正确——几何 numpy 等价单测 + Rust vs Python sequential 等价）
2. **加 Rayon par_iter**（验证并行一致性——E2M2E_SEARCH_PARALLEL=0 对照）
3. **加 `set_parallel_backend='rust'` 路由 + 进度回调**（crossbeam channel）
4. **性能基准对照**（processes / threads / rust 三档，n_dep×n_alpha 网格规模梯度）
5. 默认 backend 切换 rust（可选，视性能数据）

## 10. 风险

| 风险 | 缓解 |
|---|---|
| GIL 陷阱（调 propagate_cr3bp_py 形同串行） | 直接调纯 Rust `propagate_cr3bp` + `py.allow_threads` |
| monkeypatch 测试变空测试 | thin-wrapper 分发缝 + 检测 monkeypatch 回退 Python |
| 多进程 pickle 重建挂（Rust 状态携 Python 句柄） | Rust 后端状态必须纯标量配置元组可重建 |
| 浮点容差（SIMD 归约 vs numpy 逐元素 ULP 差） | rtol=1e-9 + atol=1e-12 兜底；chunked 边界重点测 |
| 整数索引分叉（重复最小值取首/末） | 约定取首个，测试暴露 |
| transfer_trajectory 回传内存压力 | Option 只传成功评估 / 只回传摘要 |
| 进度粒度退化（逐 α → 逐 departure） | 文档标注行为差异 |
| 积分发散（步长塌缩）穿透 | Rust 内 catch → dv=1e10 惩罚（与 NLP 回归同修） |

## 附录 A：架构合规性（ADR 对齐）

- **ADR 0011 五层**：搜索下沉是算法层→数值层的"纯数值原子下沉"，参照 `multiple_shooting` 先例（数值层 par_iter），不违反分层
- **ADR 0002 Rust 边界**：下沉的是"喂进数字就迭代"的数值评估（积分+几何），不是编排（编排留 Python transfer_search）。与 propagate_compiled 先例一致
- **ADR 0012 依赖方向**：Python 算法层调 Rust 数值层，合法方向
- **ADR 0013 测试分层**：Python 算法单元测试保留（thin-wrapper 分发缝），Rust 用等价性对照
- **建议**：开 **ADR 0017** 论证"搜索阶段纯数值内核下沉 Rayon"的边界扩张（引用 multiple_shooting 先例），固化边界

## 附录 B：不在范围（明确边界）

- NLP 优化（SLSQP/COPT）：留 Python（多候选解并行用 ProcessPoolExecutor）
- low-thrust（shooting/collocation）：待调试
- porkchop / nsga2 / wsb：可复用第一阶段基础设施（同模式 multiprocessing→rayon），后续单独迁移
- 星历路径（EphemerisDynamics/BCR4BP 搜索）：需 ephem_cache + StrictGuard，另开设计
