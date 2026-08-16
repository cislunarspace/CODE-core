# Issue #447 一手资料调研：WSB/低能转移 Rust 下沉

**调研对象**：[cislunarspace/e2m2e#447](https://github.com/cislunarspace/e2m2e/issues/447)
**调研快照**：`4a0d530`，`issue/447`，2026-08-16。仅核对代码、ADR、迁移清单和测试；未运行构建或需要 CSPICE 的检查。

## 已确认决策

- #447 同时覆盖 WSB 与低能转移，但作为两个独立 Rust 数值工作包交付。
- #447 范围内的数值内核全部下沉 Rust；Python 仅保留高层编排、领域对象和结果组装，不扩展到 #448 的流形种子、STM 转运和管传播。
- 默认走 Rust。Python 路径只能由调用方显式指定作等价性对照，绝不作为自动回退；Rust 扩展或符号缺失必须报错。

## 结论

### 1. #447 是否同时覆盖 WSB 与 low-energy

**应同时覆盖，但拆成两个独立数值工作包。** Issue 正文的目标段落主要描述 low-energy 的配对/代价评估，标题则并列写 WSB/低能；仓库迁移清单明确把两者共同登记到 #447。因此实现前应在 issue 规格中把两包分别列出，避免正文与标题的范围歧义。[GitHub issue #447](https://github.com/cislunarspace/e2m2e/issues/447)

迁移清单明确把 `wsb.py` 与 `low_energy.py` 登记为同一 #447 条目，指出 WSB 是三维 BCR4BP 网格、low-energy 是流形管配对和拼接代价，二者传播已 Rust。[`docs/architecture/numerics-migration-status.md:160-164`](../architecture/numerics-migration-status.md#L160-L164)

不能把二者合并为一个物理内核：WSB 每个 `(sun_phase, departure_phase, tof)` 候选要传播、找月心近拱点、算 H2/高度/到达态/Δv；low-energy 则把两个截面态集合做笛卡尔积、计算加权代价并排序。[`e2m2e/algorithm/transfer/wsb.py:159-175`](../../e2m2e/algorithm/transfer/wsb.py#L159-L175) [`e2m2e/algorithm/transfer/wsb.py:320-402`](../../e2m2e/algorithm/transfer/wsb.py#L320-L402) [`e2m2e/algorithm/transfer/low_energy.py:52-91`](../../e2m2e/algorithm/transfer/low_energy.py#L52-L91)

ADR 0017 将 WSB 列为可复用 multiprocessing 到 Rayon 模式、但须后续单独迁移的对象；共享的是并行/FFI 结构，不是 CR3BP 网格的物理结果类型。[`docs/adr/0017-transfer-grid-search-rust-rayon.md:60-66`](../adr/0017-transfer-grid-search-rust-rayon.md#L60-L66)

**不应吞并 #448。** 流形特征分解、STM 转运和种子生成已单列为 #448；#447 的 low-energy 应消费已有流形管或截面态。[`docs/architecture/numerics-migration-status.md:44-46`](../architecture/numerics-migration-status.md#L44-L46) [`e2m2e/algorithm/manifold/manifolds.py:117-144`](../../e2m2e/algorithm/manifold/manifolds.py#L117-L144)

## 2. 数值边界

仓内边界是把“喂进数字就迭代”的纯数值评估下沉，编排和领域对象仍在 Python。[`docs/adr/0017-transfer-grid-search-rust-rayon.md:58-73`](../adr/0017-transfer-grid-search-rust-rayon.md#L58-L73)

### WSB 应下沉

Rust 批量核应接收扁平状态、标量参数和三维网格，并对每个候选执行：

1. TLI 方向、初始态和出发 Δv；
2. 直接调用 `e2m2e_forces::bcr4bp::propagate_bcr4bp`，不从 Rayon worker 回调 Python；
3. 复刻当前密采样后的月心 `r·v` 符号扫描、分段线性插值二分、首个穿越、近月高度和 H2；
4. 复刻近月点后的目标半径穿越插值、到达 Δv、总 Δv 和筛选。[`e2m2e/algorithm/transfer/wsb.py:320-402`](../../e2m2e/algorithm/transfer/wsb.py#L320-L402) [`crates/e2m2e-forces/src/bcr4bp.rs:253-380`](../../crates/e2m2e-forces/src/bcr4bp.rs#L253-L380)

第一阶段不得把事后截面替换成 event：当前逻辑使用密采样、线性插值和二分，并取 `crossings[0]`；BCR4BP 专用 Rust 传播不支持 events，通用 Rust event 路径与 scipy 语义也未完全对齐。[`e2m2e/algorithm/manifold/sections.py:85-138`](../../e2m2e/algorithm/manifold/sections.py#L85-L138) [`e2m2e/algorithm/dynamics/bcr4bp_dynamics.py:225-230`](../../e2m2e/algorithm/dynamics/bcr4bp_dynamics.py#L225-L230)

### low-energy 应下沉

Rust 入口只接收两个 `n x 6` 截面态数组和 `(w_r, w_v)`，返回索引、两态、`delta_r`、`delta_v`、`cost`。必须保持 `cost` 升序和相等 cost 时当前嵌套循环的 `i_a/i_b` 次序，因为外层只取 `candidates[0]`。[`e2m2e/algorithm/transfer/low_energy.py:52-91`](../../e2m2e/algorithm/transfer/low_energy.py#L52-L91) [`e2m2e/algorithm/transfer/low_energy.py:146-163`](../../e2m2e/algorithm/transfer/low_energy.py#L146-L163)

### 仍留 Python

- 参数/模型校验、系统构造、公开 API、Rust 数组与 `WsbCandidate`/`PatchCandidate`/`TransferSolution` 的重包。[`e2m2e/algorithm/transfer/wsb.py:159-280`](../../e2m2e/algorithm/transfer/wsb.py#L159-L280)
- low-energy 四个流形分支组合、截面对象/穿越态收集、最优管对、ThreeBodyLambert 闭合、单位换算和 `TransferArc` 组装。[`e2m2e/algorithm/transfer/low_energy.py:132-227`](../../e2m2e/algorithm/transfer/low_energy.py#L132-L227)
- `PoincareSection` 的 callable/event 适配和流形管对象遍历；`sections.py` 已被迁移清单定为有意留 Python。[`docs/architecture/numerics-migration-status.md:55-58`](../architecture/numerics-migration-status.md#L55-L58) [`e2m2e/algorithm/manifold/sections.py:141-262`](../../e2m2e/algorithm/manifold/sections.py#L141-L262)
- WSB 精化的 ThreeBodyLambert 调用及精化失败保留原候选的产品语义。[`e2m2e/algorithm/transfer/wsb.py:405-486`](../../e2m2e/algorithm/transfer/wsb.py#L405-L486)

## 3. 后端、回退和并行开关

已有 Rust 网格入口以 `parallel`、`n_workers` 和 `progress_callback` 控制执行；`E2M2E_SEARCH_PARALLEL=0` 强制串行，显式 worker 数建立一次性 Rayon 池并覆盖 `RAYON_NUM_THREADS`。[`crates/e2m2e-integrators/src/lib.rs:3104-3123`](../../crates/e2m2e-integrators/src/lib.rs#L3104-L3123) [`crates/e2m2e-integrators/src/lib.rs:3160-3229`](../../crates/e2m2e-integrators/src/lib.rs#L3160-L3229)

计算在 `py.allow_threads` 内，进度经 channel 和 drainer 聚合；现有语义是每个 departure 一次，不是每个候选一次。[`crates/e2m2e-integrators/src/lib.rs:2981-3022`](../../crates/e2m2e-integrators/src/lib.rs#L2981-L3022) [`crates/e2m2e-forces/src/transfer_grid_search.rs:370-453`](../../crates/e2m2e-forces/src/transfer_grid_search.rs#L370-L453)

`TransferSearch` 默认 Rust，显式允许 `rust`/`processes`/`threads`；Rust 缺失时不自动换进程。自动回退只保留给 monkeypatch 测试注入缝。[`e2m2e/algorithm/transfer/transfer_search.py:40-46`](../../e2m2e/algorithm/transfer/transfer_search.py#L40-L46) [`e2m2e/algorithm/transfer/transfer_search.py:128-141`](../../e2m2e/algorithm/transfer/transfer_search.py#L128-L141) [`e2m2e/algorithm/transfer/search_parallel.py:310-377`](../../e2m2e/algorithm/transfer/search_parallel.py#L310-L377)

#447 的分歧是：WSB 当前固定 `ProcessPoolExecutor(max_workers=os.cpu_count())` 和 `spawn`，low-energy 完全未并行；需决定是否扩展公开 backend/n_workers API，但 FFI 至少应有串并行对照开关。[`e2m2e/algorithm/transfer/wsb.py:223-258`](../../e2m2e/algorithm/transfer/wsb.py#L223-L258) [`e2m2e/algorithm/manifold/manifolds.py:146-167`](../../e2m2e/algorithm/manifold/manifolds.py#L146-L167)

issue 正文所说的“双后端作为降级”应服从较晚的 ADR 0020：资源缺失必须报错，能力缺失才由调用者显式选择 `backend="scipy"`/`"rust"`，禁止 `auto`。[`docs/adr/0020-failure-handling-policy.md:64-71`](../adr/0020-failure-handling-policy.md#L64-L71)

另有文档与代码分歧：ADR 0017 把 BCR4BP 搜索说成需 `ephem_cache`/`StrictGuard`，但当前 BCR4BP 实现和绑定明确为无 SPICE 的解析纯数学，并在 `allow_threads` 中执行。#447 应以当前代码为准按纯 Rust/Rayon 设计，并修正文档。[`docs/adr/0017-transfer-grid-search-rust-rayon.md:64-66`](../adr/0017-transfer-grid-search-rust-rayon.md#L64-L66) [`crates/e2m2e-forces/src/bcr4bp.rs:1-15`](../../crates/e2m2e-forces/src/bcr4bp.rs#L1-L15) [`crates/e2m2e-integrators/src/lib.rs:2297-2302`](../../crates/e2m2e-integrators/src/lib.rs#L2297-L2302)

## 4. 必须补充的验收条件

1. WSB 小三维网格的 Python 串行与 Rust 串行逐候选比较：数量、顺序、相位、近月时刻/态、高度、H2、到达态/时刻、Δv、总 Δv、状态三元组；当前 WSB 无候选会 skip，不能替代等价性测试。[`tests/algorithm/transfer/test_wsb.py:337-373`](../../tests/algorithm/transfer/test_wsb.py#L337-L373)
2. WSB 截面覆盖多次 `r·v=0`、无穿越、`r2-r1` 退化和首个穿越选择，残差对齐现有截面测试。[`tests/algorithm/manifold/test_sections.py:53-116`](../../tests/algorithm/manifold/test_sections.py#L53-L116) [`tests/algorithm/transfer/test_wsb.py:453-495`](../../tests/algorithm/transfer/test_wsb.py#L453-L495)
3. WSB 单点传播失败须继续计数并跳过；全部传播失败返回顶层 `DIVERGED`，其余无可行候选返回 `INFEASIBLE`，不能伪装成功。[`e2m2e/algorithm/transfer/wsb.py:268-280`](../../e2m2e/algorithm/transfer/wsb.py#L268-L280) [`docs/adr/0020-failure-handling-policy.md:27-47`](../adr/0020-failure-handling-policy.md#L27-L47)
4. low-energy 逐项比较 pair 数、索引、态、范数、权重、cost、空输入和相等 cost 排序，并验证真实流水线两段弧、总 Δv 和有限终态。[`tests/algorithm/transfer/test_low_energy.py:116-172`](../../tests/algorithm/transfer/test_low_energy.py#L116-L172)
5. Rayon `parallel=False/True`、`n_workers=1/None` 逐候选保序一致，测试环境变量覆盖和进度增量总和，沿用现有网格搜索断言。[`tests/algorithm/transfer/test_rust_backend_equivalence.py:314-365`](../../tests/algorithm/transfer/test_rust_backend_equivalence.py#L314-L365) [`tests/algorithm/transfer/test_rust_progress_workers.py:73-125`](../../tests/algorithm/transfer/test_rust_progress_workers.py#L73-L125)
6. 显式 Python 参照可用；显式/默认 Rust 缺失时抛错；只有 monkeypatch 注入缝自动回退。[`tests/algorithm/transfer/test_rust_backend_via_search.py:174-279`](../../tests/algorithm/transfer/test_rust_backend_via_search.py#L174-L279)

## 5. 风险

- 截面语义漂移是最高数值风险；第一版复刻事后检测，event 精度升级另开任务。[`e2m2e/algorithm/manifold/sections.py:85-138`](../../e2m2e/algorithm/manifold/sections.py#L85-L138)
- 排序并列会改变最优管对；当前流水线只取每组第一个。[`e2m2e/algorithm/transfer/low_energy.py:146-163`](../../e2m2e/algorithm/transfer/low_energy.py#L146-L163)
- 现有 Rust 网格失败时使用 `1e10` Δv 而 Python 保留真实值，测试只能跳过该字段；新 WSB 应用 `status`/`cause` 表达失败，不再制造哨兵分歧。[`crates/e2m2e-forces/src/transfer_grid_search.rs:34-39`](../../crates/e2m2e-forces/src/transfer_grid_search.rs#L34-L39) [`tests/algorithm/transfer/test_rust_backend_equivalence.py:153-168`](../../tests/algorithm/transfer/test_rust_backend_equivalence.py#L153-L168)
- low-energy 配对是 `n_a*n_b`，完整返回态会放大内存；应记录规模和峰值，不能未经 API 决策偷偷变成 top-k。[`e2m2e/algorithm/transfer/low_energy.py:73-91`](../../e2m2e/algorithm/transfer/low_energy.py#L73-L91)
- `InvariantManifold.propagate` 的 `n_workers` 仍是预留参数；不要以 #447 顺带迁移 #448 的种子、STM 或流形传播。[`e2m2e/algorithm/manifold/manifolds.py:146-193`](../../e2m2e/algorithm/manifold/manifolds.py#L146-L193) [`docs/architecture/numerics-migration-status.md:166-168`](../architecture/numerics-migration-status.md#L166-L168)

## 建议切片

1. low-energy 配对 Rust 串行核、薄封装和逐项等价，不碰 #448。
2. WSB Rust 串行候选核，严格复刻密采样后处理和失败语义。
3. 两核分别加入 Rayon、控制面和串并对照，之后才切默认 Rust 路由。
4. 同步迁移清单；若 ADR 0017 的 BCR4BP/SPICE 表述仍不符，随改动修订。[`tests/algorithm/transfer/test_rust_backend_equivalence.py:195-241`](../../tests/algorithm/transfer/test_rust_backend_equivalence.py#L195-L241)
