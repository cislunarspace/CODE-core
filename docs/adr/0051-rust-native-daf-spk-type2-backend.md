# ADR 0051: 纯 Rust DAF + SPK Type 2 星历读取后端（#639 Phase A）

**状态**：已采纳（已实现）
**日期**：2026-09-26
**相关 issue**：#684（本决策的直接动机）、#639（调研结论与原型数据）
**相关**：ADR 0016（EphemCache 架构）、ADR 0020（失败显式化）、ADR 0039（共享内核叶）、
ADR 0048（星历 datum 与 GM 配对）

## 背景

Rust 侧星历几何查询目前全部跨 CSPICE FFI：`spice_ffi::spkezr` 直调 `spkezr_c`，
`crates/cspice`（vendor）的 `spk::easier_reader` 经进程级 `SPICE_LOCK`（ReentrantMutex）
串行化。CSPICE 内核池是进程级全局单例且非线程安全，因此 Rust 并行区（Rayon 打靶、
网格搜索）必须先经 `EphemCache` 预采样 + `StrictGuard` 把 miss 变成硬错误，否则有内核
池并发损坏风险（`SPICE(DAFFRNOTFOUND)` 或直接 panic）。

#639 调研的实测结论（决策依据）：

1. 仓库 6 个二进制内核（de421/de430/de440s.bsp、earth_latest_high_prec.bpc、
   SPICEEarthPredictedKernel.bpc、SPICELunaCurrentKernel.bpc）**91/91 段全部为
   Type 2**（SPK 43 段 + BPC 容器 48 段）。
2. 原型（DAF 解析 + SPK Type 2 求值 + `spkgeo` 首公共节点链式求值）与 CSPICE 在
   182 个目标对 × 2143 采样点（390,026 组状态）上**逐位一致**（`f64::to_bits` 全等，
   非容差），零跳过。
3. 性能（MOON 相对 EARTH，N=100k）：原型纯求值 ~1.28e6 qps，为 Rust 内循环
   `spkezr`（`batch_body_states_py`）的 3.6×。
4. 许可路径：**引用 NAIF `daf.req`/`spk.req` 规格自研**；anise（MPL-2.0）仅作验证
   oracle 与架构参照，不拷代码、不加依赖。

## 决策

### 1. 纯 Rust DAF + SPK Type 2 读取器落在 `e2m2e-spice` 新模块

新模块 `crates/e2m2e-spice/src/native_spk/`（`daf.rs` 容器解析、`spk2.rs` 求值、
`mod.rs` 注册表与链式求值），纯 std、无条件编译、零新依赖。`crates/cspice` 保持
「只做 FFI 包装」的 vendor 定位，**不改动**。

行为地面真值是 vendored CSPICE 的 C 源（`chbint.c`/`spke02.c`/`spkr02.c`/
`dafrfr_c.c`/`spkgeo.c`）：Chebyshev–Clenshaw 递推**逐操作移植**，操作顺序不得重排，
保证与 CSPICE 逐位一致。

关键规格点（易错处，作为实现契约）：

- `FWARD`/`BWARD`/摘要记录首字 `next` 是 **Fortran 1 基记录号**，不是字地址；换算
  `字地址 = (记录号-1)*128 + 1`。只有 `FREE` 本身是字地址。
- 摘要记录控制区是 3 个 double（next/prev/nsum），摘要从记录内第 24 字节（字 4）起；
  段数用 `nsum` 逐记录链读取，不用 `(128-2)/SS` 推算。
- 整数分量按 2 个 i32 打包进 1 个 f64、低字在前（`to_bits` 后低 32 位为序号小的
  整数分量）。
- SPK Type 2 段尾 `[INIT, INTLEN, RSIZE, N]`；`recno = (integer)((et-INIT)/INTLEN)+1`
  （CSPICE 只做 `min(recno,N)` 钳位，越界由选段层 `dc0 <= et <= dc1` 拦截）。
- 链式求值复刻 `spkgeo.c` 的**首个公共节点**语义：target 与 observer 各自沿段图
  `center` 向上累加状态到首个公共节点后相减——不是各自归算 SSB 再相减（浮点加法
  顺序不同即逐位不同）。
- 选段优先级复刻 `spksfs` 的「后加载者生效」：注册表按**逆加载序**、段按**逆文件内
  顺序**扫描，首个命中段胜出。

### 2. 替换面与调用契约

- `spice_ffi::spkezr` 签名不变，函数体改为：空注册表 → 项目语境错误（原文案）；
  名字→ID 解析走 `BODY_ALIASES` + "SOLAR SYSTEM BARYCENTER" + 纯数字串
  （覆盖面窄于 CSPICE `bodn2c` 内置表，未收录名硬报错，见边界）；`frame == "J2000"`
  且 `abcorr == "NONE"` 才继续；求值走 `native_spk::state`。
  **lt 恒为 0.0**：状态分量与 CSPICE 逐位一致，但 lt 不再计算（CSPICE 在
  abcorr=NONE 下返回几何单向光时）；仓库内 Rust 调用方全部丢弃 lt，唯一可观察面是
  Python 诊断口 `spice_spkezr`。
- `cspice::spk::easier_reader` 的全部生产调用点（`spk_accel` 3 处、`ephem_cache` 采样、
  `e2m2e-integrators` 的 `spice_poc_body_position`）改为经 `spice_ffi::spkezr`；
  `spice_ffi` 测试内的对拍 oracle 保持直连 CSPICE。
- 删除 `spice_ffi::bodvrd`（全仓库零调用方；GM 不在内核内，见 ADR 0048）。
- 新增 `e2m2e_spice::furnish_kernel`：先 `native_spk::load`（`NotDaf` = 文本内核，跳过
  native 登记但不算错误），再 `cspice::data::furnish`——Rust 注册表与 CSPICE 内核池
  **双登记**，任一真实错误上抛，不静默丢内核。`spice_unload` 对称双卸载（native 侧
  幂等）。六入口与 `spice_ext` 四符号签名不变；ABI 戳与「不静默降级」契约不变。

### 3. 硬报错，不回退（沿用 ADR 0020）

遇到不支持的 data type、frame、abcorr、走不到公共节点或无覆盖段：返回携带上下文
（target/observer/et 等）的显式错误。**任何情况不得回退 CSPICE**——回退会把「不支持」
变成「静默换了另一套语义」。

### 4. Phase A 边界

- 只做 SPK Type 2、`frame == 1`（J2000）、`abcorr == NONE`；容器只支持
  ND=2 的 SPK（NI=6）与 BPC（NI=5）记录头，其余布局硬报错。
- 天体名解析覆盖 `BODY_ALIASES` + SSB + 数字串；`bodn2c` 内置表的其余名字
  （"PLUTO"/"LUNA"/各质心名）不在 Phase A 覆盖内。
- 链深 ≥20 时本实现截断（CSPICE 继续上溯并折叠到第 20 节点，"settle for the
  last common node"）；真实内核链深 ≤3，不可达，记为已接受简化。
- BPC（DAF `NI=5` 容器）**只解析描述符**（供对拍与注册表登记，`center: None` 使其
  天然不参与 `state()` 选段），不求值（Phase B）。
- 不承诺移除 cspice 依赖：`pxform`/`sxform`/`et2utc`/`ktotal` 仍走 cspice-sys；
  `EphemCache`/`StrictGuard` 去留另行评估（#639 §9.2）。
- 不动 Python 侧 `SPICEManager`（继续 spiceypy；Python↔Rust 一致性由 #638 对拍框架
  守护）。时间语义不变：`et` 为 TDB 秒直接进求值，LSK 继续只服务 CSPICE 侧。

## 后果

- Rust 星历查询路径零 FFI、零可变全局内核池：`native_spk` 查询在注册表快照（`Arc`
  克隆）上进行，求值期间不持锁，Rayon 并行安全；`FFI_CALLS` 不再计入 spkezr 路径。
- `spice_spkezr`/`batch_body_states_py`/`spice_poc_body_position` 等暴露入口的行为值
  逐位不变（与 CSPICE 对拍为证），错误面变化：不支持的 frame/abcorr/type 从 CSPICE
  错误码变为项目语境错误，消息含 `SPK_NATIVE_UNSUPPORTED_*`。
- 加载/卸载经 `furnish_kernel`/`spice_unload` 的路径两侧登记同步；直接调
  `cspice::data::furnish` 的新调用方将导致 native 注册表缺内核，属须避免的旁路
  （测试 helper 已全部切换）。

## 验证

- `native_spk_parity.rs` 七组用例（另有换源后原样全绿的既有
  `ephem_cache`/`spk_accel`/`nbody_stm` 回归）：91 段描述符逐位一致（含地球预测
  BPC 跨摘要记录链 2 记录 37 段）；de440s/de430 全 ID 对 × 2143 点状态逐位一致
  零跳过；四链拓扑显式用例（(199,1)/(10,0) 直连、(301,399) 2 跳、(10,399) 3 跳）；
  「FWARD 当字地址」负控回归；`UnsupportedType/Frame/Abcorr` 硬报错；残缺链
  拓扑（target 链断裂、observer 链命中公共节点，钉住 spkgeo 的 `found` 重置）；
  native ≥ 3× FFI 性能（断言仅优化构建生效：CSPICE 为预编译 -O2 静态库，debug
  比值失真，验收证据取 release 运行日志）。
- `cargo test --workspace -- --test-threads=1`、`uv run --no-sync python -m pytest
  -m "not spice"`、`make check` 全绿。

## 被否方案

1. **引入 anise 作实现**。否：新增依赖违反 Phase A 约束；anise 仅作 oracle。
2. **各自归算 SSB 再相减**。否：浮点求和顺序与 `spkgeo` 不同，逐位一致不成立。
3. **Phase A 顺带删 cspice/`EphemCache`**。否：pxform/sxform/et2utc 仍需 CSPICE；
   缓存去留属 #639 §9.2 后续评估，本决策只替换星历求值路径。
