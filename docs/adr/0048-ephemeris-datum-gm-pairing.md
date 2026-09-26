# ADR 0048: 星历 datum 与 GM 的配对——GM 跟随已加载星历内核

**状态**：已采纳（已实现）
**日期**：2026-09-26
**相关 issue**：#665（本决策的直接动机）、#670（DE421 外行星 GM 权威来源的缺口）、#377/ADR 0022（基准集来源，本 ADR 修订其决策 4）
**相关**：ADR 0022（决策 3「多基准并存」与决策 4「星历动力学默认 DE440」在此细化）、ADR 0013（按定义验证）、ADR 0039（共享内核叶）
**修订**：ADR 0022 决策 4（见文末）

## 背景

CE-5 精密定轨策略（孔静等 2022，表 1）的第三体引力（太阳、月球与大行星）用
JPL **DE421** 星历。e2m2e 无法按该口径复算算例，因为存在两处脱节：

1. **拿不到 DE421 内核**：`kernels/` 只有 de440s/de430，`load_design_kernels`
   写死 `["de440s.bsp", "de430.bsp"]`；`kernels-v1` release 也没有 de421，
   而无差别拉取该 release 的 `download_kernels.py` 因此拿不到它。
   （内核来源已补：NAIF `spk/planets/a_old_versions/de421.bsp` 经 Colab 取回并
   上传至 `kernels-v1`，见 `scripts/colab_download_de421.py`。）
2. **GM 与星历 datum 未配对**：`SPICEManager.get_gm` 对已知天体无条件取
   `gm_by_datum["DE440"]`，既无 datum 参数、也不感知已加载口径。于是口径是
   混合的：位置来自 de440s，GM 来自写死的 DE440 表。

动笔前先核实了一个决定性事实：**SPK 内核本身不携带 GM**。在
`de421.bsp` + `pck00010.tpc` 下对所有天体调用 `bodvrd(..., "GM")` 一律报
`KERNELVARNOTFOUND`（与仓库既有注释「DE430 bsp 不带 GM 数据」一致）。因此：

- "GM 从内核读" 在 SPICE 层面不成立，**GM 只能来自 `constants.toml` 的声明式
  body 表**；
- datum 只能由内核**文件名**推断，不可能从内核内容推断。

另一个实测约束：**CSPICE 内核池是进程级全局的**。两个 `SPICEManager` 实例
同时加载 de421 与 de440s 时，后加载者对重叠覆盖段全局生效，两者会查到同一个
内核——"同一进程内并存两套口径"是不成立的前提。这反过来强化了自动配对的价值：
调用方几乎不可能手工维持一致。

## 决策

### 1. GM 口径跟随已加载星历内核（自动配对）

`SPICEManager` 记录已加载星历内核的 `(绝对路径, datum)` 列表；`ephemeris_datum`
取**最后一个仍加载**的内核的 datum，无星历内核时为 `DE440`（ADR 0022 决策 4 的
默认值）。`get_gm(body, datum=None)` 默认用该口径，`datum` 可显式覆盖。

取"最后加载者"而非"首个"：与 SPICE 对重叠覆盖段「后加载者生效」的优先级规则
一致，避免出现"位置按 de440s、GM 按 de421"的静默错配。

### 2. 内核 → datum 是显式白名单，未收录者回退并告警

映射表只收录仓库确实有 GM 依据的基准：

| 内核 | datum |
|---|---|
| `de421.bsp` | DE421 |
| `de430/de435/de438/de440/de440s/de441/de442/de442s.bsp` | DE440 |

de441/de442 的 GM 与 DE440 **确有差异**，但补全它们需要的权威来源不在本仓库
现有证据内，故一律按 DE440 处理并**告警一次**，绝不静默混用。同理，某基准下
若无某天体的 GM 记录（当前：木星及以外无 DE421 行），回退 DE440 并按
`(天体, 基准)` 组合告警一次。外行星 DE421 GM 的补齐见 #670。

### 3. 显式请求的口径缺失时硬失败，不静默降级

`find_ephemeris_kernel(search_dir, preferred=...)` 与
`load_design_kernels(..., datum=...)` 在显式请求 DE421 而 `de421.bsp` 不存在时
抛 `FileNotFoundError`，不悄悄退回 de440s/de435——降级会把"按 DE421 口径复算"
变成"以为在复算、其实不是"，比失败更糟（ADR 0020 的失败显式化）。

缺省路径（不传 `datum`）保持 `de440s > de430` 的历史顺序**逐位不变**：
`kernels/` 里出现可选内核不得让未声明口径的调用方静默换口径。

### 4. GM 的声明式单一来源不变，body 表与 datum 表必须同值

DE421 GM 仍只写在 `constants.toml`：`[datum.DE421]` 是聚合视图，
`[body.X.gm].DE421` 是逐天体视图，二者由测试钉为同值。有卫星、两代星历 GM
确实不同的天体（SUN/EARTH/MOON/EMB）给各自 DE421 值；MERCURY/VENUS/MARS 的 GM
在两代间是不变量，两行同值且 `source` 保留真实的 `"DE440"`（值不错、出处不冒充）。

### 5. Rust 侧本轮不改

第三体 GM 本就从 Python 经 `to_rust_spec`（`ThirdBodyGravity` 等）与
`gm_values` 传入 Rust，故 Python 侧改口径即已生效。`gm_for_body()`
（`crates/e2m2e-forces/src/forces/gravity_field.rs`，唯一调用点是固潮 Step1 的
扰动体 GM）继续用 DE440 编译期常数，函数注释写明该取舍。

### 6. 不暴露到 MCP/CLI/sidecar

自动配对已覆盖常见用法（加载哪个星历就是哪个口径），`tool_inventory` 派生面
不动。显式口径选择作为工具参数留给后续按需引入。

## 后果

- 加载 `de421.bsp` 后，第三体**位置**与 **GM** 同时为 DE421 口径，无需调用方
  手工配对；`Datum.DE421`（CR3BP/族生成默认）与星历动力学的口径自此可对齐。
- **行为变化**：`kernels/` 里存在 de421 且显式加载时，同一套代码的 GM 会从
  DE440 切到 DE421（例：月球 GM 4902.800118 → 4902.8005821478，相对差
  9.5e-8）。未加载 de421 的环境行为逐位不变。
- GM 现在**依赖运行环境里存在哪个内核**。这是自动配对的固有代价，用"缺省不
  降级 + 未收录内核告警 + 测试守卫缺省路径"三重约束把意外面收窄。
- 口径差异可量化并固化在测试中（`tests/algorithm/design/test_de421_datum.py`）：
  2020–2026 弧段上 de421 与 de440s 的月球位置差米级（实测均值 3 m / 最大
  4 m）、太阳百米级（均值 150 m / 最大 220 m），与两代星历的精度声明相符。
- 外行星在 DE421 口径下仍回退 DE440（有告警）。凡要用于论文口径复算的外行星
  项，必须先把权威 GM 补进 `constants.toml`（#670）。

## 被否方案

1. **强制调用方传 `datum`**（不给默认配对）。否：最需要口径一致的正是"加载
   de421 就想复算论文"的调用方，多一个必填参数只是把出错的义务转给他们；且
   ADR 0022 已确立基准是显式一等对象，配对规则可以是它的自动默认。
2. **从内核内容读 GM**。否：本 ADR 背景部分的实测已证伪——SPK 不含 GM。
3. **给 Rust 侧也加 datum 分支**。否：第三体 GM 不经过 Rust 常数表，加了是
   无消费者的死分支；固潮继续用 DE440 并已在注释中声明。
4. **同一进程内并存两套口径供对拍**。否：CSPICE 内核池进程级全局，后加载者
   覆盖；对拍只能按"换内核再查一遍"的顺序方式做（实现即如此）。

## 验证

- `tests/algorithm/design/test_de421_datum.py`：口径选择契约（显式加载/缺失报错/
  未知基准报错/缺省顺序不变）、GM 一致性（DE421 已发布值）、body↔datum 表同值、
  `ThirdBodyGravity.to_rust_spec` 真的携带 DE421 GM、位置差量级守卫。
- `tests/data/kernels/test_spice_manager.py`：GM 默认口径、显式覆盖、缺失回退
  告警一次、加载/卸载切换口径、多内核"后加载者胜"。
- `tests/data/constants/test_constants.py`：body 表与 datum 表 DE421 GM 同值；
  外行星**有意**无 DE421 行（把"留空"钉成决定而非疏忽）。
- `tests/algorithm/design/test_kernel_future_coverage.py`：
  `kernels/` 存在 de421 时缺省口径仍为 de440s。

## Revision to ADR 0022

Decision 4 的「ephemeris dynamics defaults to `DE440`」细化为：星历动力学的 GM
基准**跟随已加载星历内核**（本 ADR 决策 1），仅在无星历内核时仍默认 `DE440`。
决策 3「多基准并存、按场景选」与决策 5「单一来源 + 生成期对齐」不变。
