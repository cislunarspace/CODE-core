# ADR 0048: 星历 datum 与 GM 的配对，GM 跟随已加载星历内核

**状态**：已采纳（已实现）
**日期**：2026-09-26
**相关 issue**：#665（本决策的直接动机）、#670（DE421 外行星 GM 权威来源的缺口）、#377/ADR 0022（基准集来源，本 ADR 修订其决策 4）
**相关**：ADR 0022（决策 3“多基准并存”与决策 4“星历动力学默认 DE440”在此细化）、ADR 0013（按定义验证）、ADR 0039（共享内核叶）
**修订**：ADR 0022 决策 4（见文末）

## 背景

CE-5 精密定轨策略（孔静等 2022，表 1）的第三体引力（太阳、月球与大行星）用 JPL DE421 星历。e2m2e 无法按该口径复算算例，因为存在两处脱节：

1. **拿不到 DE421 内核**：`kernels/` 只有 de440s/de430，`load_design_kernels` 写死 `["de440s.bsp", "de430.bsp"]`，`kernels-v1` release 也没有 de421，而无差别拉取该 release 的 `download_kernels.py` 因此拿不到它。（内核来源已补：NAIF `spk/planets/a_old_versions/de421.bsp` 经 Colab 取回并上传至 `kernels-v1`，见 `scripts/colab_download_de421.py`。）
2. **GM 与星历 datum 未配对**：`SPICEManager.get_gm` 对已知天体无条件取 `gm_by_datum["DE440"]`，既无 datum 参数、也不感知已加载口径。于是口径是混合的：位置来自 de440s，GM 来自写死的 DE440 表。

动笔前先核实了一个决定性事实：SPK 内核本身不携带 GM。在 `de421.bsp` + `pck00010.tpc` 下对所有天体调用 `bodvrd(..., "GM")` 一律报 `KERNELVARNOTFOUND`（与仓库既有注释“DE430 bsp 不带 GM 数据”一致）。因此：

- “GM 从内核读”在 SPICE 层面不成立，GM 只能来自 `constants.toml` 的声明式 body 表。
- datum 只能由内核文件名推断，不可能从内核内容推断。

另一个实测约束：CSPICE 内核池是进程级全局的。两个 `SPICEManager` 实例同时加载 de421 与 de440s 时，后加载者对重叠覆盖段全局生效，两者会查到同一个内核，“同一进程内并存两套口径”是不成立的前提。这反过来强化了自动配对的价值：调用方几乎不可能手工维持一致。

## 决策

### 1. GM 口径跟随已加载星历内核（自动配对）

`SPICEManager` 记录已加载星历内核的 `(绝对路径, datum)` 列表。`ephemeris_datum` 取最后一个仍加载的内核的 datum，无星历内核时为 `DE440`（ADR 0022 决策 4 的默认值）。`get_gm(body, datum=None)` 默认用该口径，`datum` 可显式覆盖。

取“最后加载者”而非“首个”：与 SPICE 对重叠覆盖段“后加载者生效”的优先级规则一致，避免出现“位置按 de440s、GM 按 de421”的静默错配。

### 2. 内核到 datum 的映射是显式白名单，未收录者回退并告警

映射表只收录仓库确实有 GM 依据的基准：

| 内核 | datum |
|---|---|
| `de421.bsp` | DE421 |
| `de430/de435/de438/de440/de440s/de441/de442/de442s.bsp` | DE440 |

de441/de442 的 GM 与 DE440 确有差异，但补全它们需要的权威来源不在本仓库现有证据内，故一律按 DE440 处理并告警一次，绝不静默混用。同理，某基准下若无某天体的 GM 记录（当前：木星及以外无 DE421 行），回退 DE440 并按 `(天体, 基准)` 组合告警一次。外行星 DE421 GM 的补齐见 #670。

### 3. 显式请求的口径缺失时硬失败，不静默降级

`find_ephemeris_kernel(search_dir, preferred=...)` 与 `load_design_kernels(..., datum=...)` 在显式请求 DE421 而 `de421.bsp` 不存在时抛 `FileNotFoundError`，不悄悄退回 de440s/de435。降级会把“按 DE421 口径复算”变成“以为在复算、其实不是”，比失败更糟（ADR 0020 的失败显式化）。

缺省路径（不传 `datum`）保持 `de440s > de430` 的历史顺序逐位不变：`kernels/` 里出现可选内核不得让未声明口径的调用方静默换口径。

### 4. GM 的声明式单一来源不变，body 表与 datum 表必须同值

DE421 GM 仍只写在 `constants.toml`：`[datum.DE421]` 是聚合视图，`[body.X.gm].DE421` 是逐天体视图，二者由测试钉为同值。有卫星、两代星历 GM 确实不同的天体（SUN/EARTH/MOON/EMB）给各自 DE421 值。MERCURY/VENUS/MARS 的 GM 在两代间是不变量，两行同值且 `source` 保留真实的 `"DE440"`（值不错、出处不冒充）。

### 5. Rust 侧本轮不改

第三体 GM 本就从 Python 经 `to_rust_spec`（`ThirdBodyGravity` 等）与 `gm_values` 传入 Rust，故 Python 侧改口径即已生效。`gm_for_body()`（`crates/e2m2e-forces/src/forces/gravity_field.rs`，唯一调用点是固潮 Step1 的扰动体 GM）继续用 DE440 编译期常数，函数注释写明该取舍。

### 6. 不暴露到 MCP/CLI/sidecar

自动配对已覆盖常见用法（加载哪个星历就是哪个口径），`tool_inventory` 派生面不动。显式口径选择作为工具参数留给后续按需引入。

## 后果

- 加载 `de421.bsp` 后，第三体位置与 GM 同时为 DE421 口径，无需调用方手工配对。`Datum.DE421`（CR3BP/族生成默认）与星历动力学的口径自此可对齐。
- **行为变化**：`kernels/` 里存在 de421 且显式加载时，同一套代码的 GM 会从 DE440 切到 DE421（例：月球 GM 从 4902.800118 变为 4902.8005821478，相对差 9.5e-8）。未加载 de421 的环境行为逐位不变。
- GM 现在依赖运行环境里存在哪个内核。这是自动配对的固有代价，用“缺省不降级、未收录内核告警、测试守卫缺省路径”三重约束把意外面收窄。
- 口径差异可量化并固化在测试中（`tests/algorithm/design/test_de421_datum.py`）：2020–2026 弧段上 de421 与 de440s 的月球位置差米级（实测均值 3 m / 最大 4 m）、太阳百米级（均值 150 m / 最大 220 m），与两代星历的精度声明相符。
- 外行星在 DE421 口径下仍回退 DE440（有告警）。凡要用于论文口径复算的外行星项，必须先把权威 GM 补进 `constants.toml`（#670）。

## 被否方案

1. **强制调用方传 `datum`**（不给默认配对）。否：最需要口径一致的正是“加载 de421 就想复算论文”的调用方，多一个必填参数只是把出错的义务转给他们，且 ADR 0022 已确立基准是显式一等对象，配对规则可以是它的自动默认。
2. **从内核内容读 GM**。否：本 ADR 背景部分的实测已证伪，SPK 不含 GM。
3. **给 Rust 侧也加 datum 分支**。否：第三体 GM 不经过 Rust 常数表，加了是无消费者的死分支，固潮继续用 DE440 并已在注释中声明。
4. **同一进程内并存两套口径供对拍**。否：CSPICE 内核池进程级全局，后加载者覆盖，对拍只能按“换内核再查一遍”的顺序方式做（实现即如此）。

## 验证

- `tests/algorithm/design/test_de421_datum.py`：口径选择契约（显式加载、缺失报错、未知基准报错、缺省顺序不变）、GM 一致性（DE421 已发布值）、body 与 datum 表同值、`ThirdBodyGravity.to_rust_spec` 真的携带 DE421 GM、位置差量级守卫。
- `tests/data/kernels/test_spice_manager.py`：GM 默认口径、显式覆盖、缺失回退告警一次、加载与卸载切换口径、多内核“后加载者胜”。
- `tests/data/constants/test_constants.py`：body 表与 datum 表 DE421 GM 同值，外行星有意无 DE421 行（把“留空”钉成决定而非疏忽）。
- `tests/algorithm/design/test_kernel_future_coverage.py`：`kernels/` 存在 de421 时缺省口径仍为 de440s。

## 对 ADR 0022 的修订

决策 4 的“星历动力学默认 `DE440`”细化为：星历动力学的 GM 基准跟随已加载星历内核（本 ADR 决策 1），仅在无星历内核时仍默认 `DE440`。决策 3“多基准并存、按场景选”与决策 5“单一来源与生成期对齐”不变。

## 修订（2026-09-26）：簿记作用域与内核加载告警（#683 评审）

### 背景

PR #683 评审发现三处与本 ADR 自身声明不符的缺口。一是 datum 簿记（`_loaded_ephemeris`）是实例级的，而 CSPICE 内核池是进程级全局的（本 ADR 第 2 节），于是同一进程内第二个 manager 加载 `de440s.bsp` 后，位置被静默改成 de440s 口径，第一个 manager 仍报告 DE421 datum，正是本 ADR 要消除的“de440s 位置加 DE421 GM”混合口径。二是承诺的“de441/de442 回退 DE440 并告警”没有实现，`get_gm` 的告警分支永远不会触发，因为白名单把那些内核直接映射到 DE440。三是一次性告警集合是模块级全局变量，使测试模块之间相互耦合。

### 决策

1. **簿记改为类级。** `SPICEManager._loaded_ephemeris` 由所有实例共享，它镜像进程级全局池，因此 `ephemeris_datum`（以及由它决定的 `get_gm` 默认口径）总是反映进程中最后一次加载的内核。实例级簿记无法刻画全局池。
2. **加载期告警兑现“绝不静默混用”的契约。** 加载 GM 只是近似值的内核（de430/de435/de438/de441/de442/de442s 按 DE440 处理）时，每个内核名告警一次。加载不在白名单内的 DE 内核（例如 `de423.bsp`、改名的副本）同样告警一次，并保持当前 datum 不变。`de421` 与 `de440(s)` 携带权威 GM 表，不告警。
3. **告警去重按 manager 实例计**，不再是模块级全局，消除测试之间与调用方之间的耦合。
4. **datum 到内核的映射只有一份。** `SPICEManager.datum_kernel_name`（由 `_DATUM_KERNEL_PREFERENCE` 支撑，此时已包含 `DE440` 到 `de440s.bsp` 的条目）是唯一映射，`design_orbit.load_design_kernels` 消费它而不再另存一份，于是 `find_ephemeris_kernel(preferred="DE440")` 与 `load_design_kernels(datum="DE440")` 一致。

### 后果

- 原先假定同一进程内两套系统可持有不同 datum 的测试（`test_de421_datum.py::test_gm_follows_loaded_kernel`）改为断言共享 datum，DE440 对照通过显式 `datum=` 查询取得。
- 错误码、失败语义与 GM 数值均不变。

## 修订（2026-09-26，b）：撤回 SUN 的 DE421 行、datum 候选集与簿记锁（#683 复审）

- **撤回 `[body.SUN.gm].DE421` 行。** DE421 的太阳 GM 在本仓库没有权威出处（两个候选值相对差约 9e-8，且 JPL 的定义由 IAU 高斯常数与定义性 AU 决定，不是按代拟合），故删除该行，而不是静默返回猜测值：DE421 口径下查询 SUN 现在回退 DE440 并告警一次，与外行星同样处理，归 #670 跟踪。修订 (a) 记录了该缺口，却仍让这个值经由新增的 body 行静默露面。
- **datum 对应的是内核候选集，不是单个文件。** `_DATUM_KERNEL_PREFERENCE` 映射到有序元组（`DE440` 对应 `(de440.bsp, de440s.bsp)`），由 `SPICEManager.datum_kernel_names` 暴露，于是只含 `de440.bsp` 的目录也能满足 `preferred="DE440"`，不会报告 datum 缺失（那与管理器自身的默认优先级矛盾）。
- **簿记锁。** 类级列表在 `_bookkeeping_lock` 下变更（与模块的 `_leapseconds_lock` 约定一致），并按绝对路径去重，该语义已写明。

## 修订（2026-09-26，c）：furnsh 与簿记的临界区、失败回滚（#697）

### 背景

修订 (b) 只记录了“类级列表在 `_bookkeeping_lock` 下变更”，#683 之后仍有三处边界未闭合。其一，Python 侧 `furnsh` 发生在持锁记账之前，并发加载不同内核时，簿记末位可能晚于池里实际生效的末位，重新落回本 ADR 第 1 节要消除的“位置按新内核、GM 按旧簿记”错配。其二，双 `furnsh` 的异常路径没有回滚：Python 侧已生效而 Rust 侧失败时，池已变更而簿记未变更，卸载则反之，注释“仅在 furnsh 成功后登记，失败不污染当前口径”在双 `furnsh` 下不成立。

### 决策

1. **临界区覆盖双侧 furnsh/unload 与簿记三步。** `load_kernel` 的 Python `furnsh`、Rust `spice_furnsh` 与 `_loaded_ephemeris` 更新在同一个 `_bookkeeping_lock` 内完成，`unload_kernel` 对称。故簿记顺序与池的实际生效顺序一致。该锁只界定这三步的相对顺序，不串行化、也不承担 CSPICE 池内部状态的线程安全（池仍是进程级全局的，见第 2 节）。
2. **加载期告警在临界区之外。** `_warn_unregistered_kernel` 读 `ephemeris_datum`，而后者取同一把非可重入锁，放进临界区内即死锁。告警一律在锁外调用。
3. **失败路径回滚，并保留原异常。** Rust `furnsh` 失败时，撤销本次 Python 侧 `furnsh`（CSPICE 对同一文件按实例计数：重复 `furnsh` 使池内多一条，`unload` 移除最后一条，故一次 `unload` 恰为本次 `furnsh` 的逆操作，不会卸掉此前已加载的实例）且不登记簿记。Rust `unload` 失败时，恢复 Python 侧 `furnsh`（成员回到两池）并把该内核的簿记条目移到末位。恢复即重新加载，Python 池内它重新成为“后加载者”，簿记须跟着走，否则“位置按该内核、GM 按旧末位”再次失配。回滚与恢复本身失败只记 warning，不掩盖原异常。
4. **`spice_furnsh`/`spice_unload` 为 `None`（扩展未含 spice feature）时沿用 #382 的既有接缝：只喂 Python 池。** 本修订不改变该取舍，也不把它上升为本 ADR 的口径契约。

### 后果

- 并发或异常注入下，簿记末位与 Python 池的实际生效口径一致。登记成功即两侧同步。`furnsh` 失败回滚为调用前状态，簿记不动。`unload` 失败恢复 Python 侧并把该条目移到末位，恢复即重新加载。此时 Rust 池因 `unload` 失败保持原次序，两池次序本就无法兼顾，以 Python 池与簿记对齐为准。
- 守卫：`tests/data/kernels/test_spice_manager.py::TestKernelBookkeepingAtomicity`（Python `furnsh` 观察到簿记锁已持有、加载失败回滚、卸载失败恢复并重排簿记、重载失败不丢先前加载、告警在锁外）。
- 测试隔离随之收紧：会真实 `furnsh` 的用例改在 `try/finally` 中卸载，`de421.bsp` 不再从断言失败路径泄漏到同 worker 的后续用例。

## 修订（2026-09-26，d）：`de421.bsp` 纳入 git-lfs，不再只作 release 资产

### 背景

背景第 1 条把“拿不到 DE421 内核”列为本 ADR 的动机之一，并注明补法是从 NAIF 经 Colab 取回后上传 `kernels-v1`。那是分发手段而非入库：clone 得到的 `kernels/` 不含 de421，要按 ADR 口径复算必须先联网跑 `make kernels`。而 de430/de440s 自 #517 起就在 git-lfs 里（`.gitattributes` 的 `kernels/*.bsp` 规则），de421 是该规则下唯一的例外。

### 决策

1. **`kernels/de421.bsp` 按同一条 LFS 规则入库**（16 790 528 字节，sha256 `08b20db2ae22…`，与 `kernels-v1` 的 `de421.bsp` 资产逐字节相同）。`.gitattributes` 无需改动，`kernels/*.bsp` 已覆盖。
2. **release 路径保留且不是“备用降级”**：CI 的 `actions/checkout` 未开 `lfs: true`（检出的是 LFS 指针文件），任何未拉 LFS 的 clone 也如此，二者仍由 `make kernels`（`scripts/download_kernels.py`）供给，既有指针判据 `_is_lfs_pointer` 已覆盖本文件。两条路径的同名资产须保持同源。
3. **口径语义与守卫不动**：决策 1（GM 跟随仍加载的末位内核）、决策 3（显式请求 DE421 而内核缺失时 `FileNotFoundError`）以及缺省顺序 `de440s > de430` 逐位不变。本修订只改“de421.bsp 从哪来”，不改“加载后查到什么”。
4. **`de421.cmt` 不入库**：仓库没有 `.cmt` 消费者，该注释文本的 NAIF 取法仍记在 `scripts/colab_download_de421.py`。

### 后果

- 拉过 LFS 的本地环境无需联网即有 de421，`requires_de421_kernel` 守卫的用例（`tests/algorithm/design/test_de421_datum.py` 等）由 skip 转为实跑。
- 仓库 LFS 侧增加 16.8 MB。未拉 LFS 的 clone 只多一个指针文件，行为不变。

## 修订（2026-09-27，#713）：SUN 的 DE421 GM 两视图配对恢复

### 背景

第 4 节要求 body 与 datum 两视图同值。SUN 的 `[datum.DE421].sun_gm` 一直有值（1.32712428e11），但 `[body.SUN.gm]` 从未有过 DE421 行，聚合视图与逐天体视图矛盾。`tests/data/constants/test_constants.py` 的一致性测试把 SUN 归入“无出处、有意留空”一组而静默跳过，两视图漂移无守卫。

### 决策

1. **补 `[body.SUN.gm].DE421 = 1.32712428e11`，与 `[datum.DE421].sun_gm` 逐位同值，`source = "DE421"`**。该值的仓库内权威文献出处不在现有证据内，出处缺口归 #670。`source = "DE421"` 表示“DE421 口径”的标注而非文献出处，诚实性由本段文字承担。
2. **测试恢复 SUN 覆盖**：两视图同值断言纳入 `TestDatumBodyGmAlignment.test_earth_moon_emb_de421_agree`，Rust 对拍补 `body.SUN.gm.DE421/DE440`，“有意留空”断言收窄为 JUPITER 及以外。

### 后果

- 数值影响：DE421 口径下第三体太阳 GM 由 DE440 回退值 `1.32712440018e11` 切到 `1.32712428e11`（相对差约 9e-8），属第 4 节自动配对的预期行为。两视图同值后，按口径查询与按天体查询不再分歧。
- 出处证据缺口（#670）不变：补到权威出处时按 #670 更新 `source` 标注与本 ADR。
